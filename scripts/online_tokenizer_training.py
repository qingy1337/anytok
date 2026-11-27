"""Deterministic policy rewards and exact joint training updates on a shared batch."""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from types import MethodType

import torch

from anytok_online import Generator
from online_reward_scoring import BATCH_SIZE, MICROBATCH_SIZE, digest, paired_batch
from rift_eval_memory import inference_attention, inference_layer


class JointScorer:
    def __init__(self, trainer, inputs, checkpoint, recipe):
        from pool_training_memory import apply_gpu_memory_fix

        self.trainer = trainer
        self.canonical = trainer.AutoTokenizer.from_pretrained(inputs / 'model', local_files_only=True)
        self.lm = trainer.AutoModelForCausalLM.from_pretrained(
            inputs / 'model', local_files_only=True, dtype=torch.float32, low_cpu_mem_usage=True).to('cuda')
        trainer.separate_input_embeddings_and_freeze_output(self.lm)
        self.source = self.lm.get_input_embeddings().weight.detach()
        original = self.source.clone()
        self.lm.load_state_dict(torch.load(checkpoint / 'lm.pt', map_location='cpu', weights_only=True, mmap=True))
        if not (torch.equal(self.source, original) and torch.equal(self.lm.get_output_embeddings().weight, original)):
            raise ValueError('Canonical input or output weights changed')
        del original
        self.hn = trainer.ZettHypernet.from_pretrained(checkpoint / 'hypernet', local_files_only=True).to('cuda')
        self.fallback = trainer.enable_byte_fallbacks(self.hn, self.canonical)
        trainer.freeze_rescalers(self.hn)
        self.hn.model.activation_checkpointing = True
        self.hn.activation_checkpoint_input = True
        apply_gpu_memory_fix(self.hn)
        self.params = [p for model in (self.hn, self.lm) for p in model.parameters() if p.requires_grad]
        self.generator = Generator(inputs / 'corpus', recipe)

    @contextmanager
    def inference(self):
        """Use the same forward operators as the frozen search, then restore training."""
        self.lm.eval()
        self.hn.eval()
        methods = [(layer, inference_layer) for layer in self.hn.model.layers]
        methods += [(attention, inference_attention) for layer in self.hn.model.layers
                    for attention in (layer.attention, layer.inter_token_attention_down, layer.inter_token_attention_up)]
        saved = [(module, module.forward) for module, _ in methods]
        try:
            for module, function in methods:
                module.forward = MethodType(function, module)
            with torch.inference_mode():
                yield
        finally:
            for module, forward in saved:
                module.forward = forward
            self.hn.train()
            self.lm.train()

    def prepare(self, settings, shared):
        sample = self.generator.sample(**settings)
        if (len(sample.vocabulary) != settings['vocab_size']
                or sum(sample.bucket_quotas.values()) != 200_000_000
                or not 0 <= 200_000_000 - sample.bytes_used <= 3 * len(sample.bucket_quotas)):
            raise ValueError('Unexpected realized vocabulary or byte budget')
        native, forms, priors, truncated = sample.training_inputs(
            self.canonical, self.fallback, self.trainer.build_surface_forms)
        examples, controls, proofs, rejected, prefix_bytes = paired_batch(self.trainer, self.canonical, native, shared)
        metrics = dict(settings=settings, tokenizer_sha256=hashlib.sha256(native.to_str().encode()).hexdigest(),
                       actual_quotas=sample.bucket_quotas, used_bytes=sample.bytes_used, valid_examples=len(examples),
                       rejected_examples=rejected, scored_bytes=sum(r[2] for r in examples),
                       scored_tokens=sum(len(r[1]) for r in examples), total_prefix_bytes=sum(prefix_bytes),
                       maximum_positions=max(r['maximum_positions'] for r in proofs), paired_batch_sha256=digest(proofs),
                       truncated_forms=truncated)
        return (examples, controls, torch.as_tensor(forms, device='cuda', dtype=torch.long),
                torch.as_tensor(priors, device='cuda'), metrics)

    def reward(self, batch):
        examples, controls, forms, priors, metrics = batch
        trainer = self.trainer
        with self.inference():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                table, output, _ = self.hn(forms, priors, self.source, None, deterministic=True)
            if output is not None or table.shape != (len(forms), self.source.shape[1]) or not torch.isfinite(table).all():
                raise ValueError('Invalid full-vocabulary hypernetwork output')
            canonical = trainer.canonical_loss_nats(self.lm, controls, self.source,
                                                    MICROBATCH_SIZE, trainer.make_lm_microbatch)
            transferred = 0.
            for start in range(0, BATCH_SIZE, MICROBATCH_SIZE):
                micro = examples[start:start + MICROBATCH_SIZE]
                embeds, mask, labels, positions = trainer.make_lm_microbatch(micro, table, self.source, 'cuda')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = self.lm(inputs_embeds=embeds, attention_mask=mask, labels=labels,
                                   position_ids=positions, use_cache=False).loss
                transferred += float(loss) * sum(len(targets) for _, targets, _ in micro)
                del embeds, mask, labels, positions, loss
        raw_bpb, canonical_bpb, gap_bpb = trainer.paired_gap_bpb(transferred, canonical, metrics['scored_bytes'])
        return dict(raw_bpb=raw_bpb, canonical_bpb=canonical_bpb, gap_bpb=gap_bpb,
                    transferred_nats=transferred, canonical_nats=canonical)

    def update(self, batch, optimizer, step):
        examples, _, forms, priors, metrics = batch
        trainer = self.trainer
        self.lm.train()
        self.hn.train()
        hn_lr, lm_lr = trainer.set_learning_rates(optimizer, 20000 + step - 1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            generated, output, _ = self.hn(forms, priors, self.source, None, deterministic=False)
        if output is not None or not torch.isfinite(generated).all():
            raise ValueError('Invalid training hypernetwork output')
        leaf = generated.detach().float().requires_grad_(True)
        loss_nats = 0.
        for start in range(0, BATCH_SIZE, MICROBATCH_SIZE):
            micro = examples[start:start + MICROBATCH_SIZE]
            embeds, mask, labels, positions = trainer.make_lm_microbatch(micro, leaf, self.source, 'cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = self.lm(inputs_embeds=embeds, attention_mask=mask, position_ids=positions,
                               labels=labels, use_cache=False).loss
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite student training loss')
            count = sum(len(r[1]) for r in micro)
            loss_nats += float(loss.detach()) * count
            (loss * count / metrics['scored_tokens']).backward()
            del embeds, mask, labels, positions, loss
        if leaf.grad is None or not torch.isfinite(leaf.grad).all():
            raise ValueError('Invalid student embedding gradient')
        lexical = trainer.input_lexical_loss(generated, forms, self.source, self.hn.config.pad_token_id, 0)
        if not torch.isfinite(lexical):
            raise ValueError('Nonfinite lexical loss')
        trainer.backward_hypernetwork(generated, leaf.grad.detach(), lexical, optimized=True)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, .1, foreach=False)
        if not torch.isfinite(grad_norm):
            raise ValueError('Nonfinite student gradient norm')
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return dict(training_loss_nats=loss_nats, lexical_loss=float(lexical.detach()),
                    gradient_norm=float(grad_norm), hn_lr=hn_lr, lm_lr=lm_lr)


def rollback_uncommitted_steps(output, restart_step):
    """Preserve outputs after the last verified restart before replaying them."""
    checkpoints = [path for path in output.glob('step-*')
                   if path.is_dir() and int(path.name.removeprefix('step-')) > restart_step]
    events = output / 'events.jsonl'
    original = events.read_bytes() if events.exists() else b''
    retained = []
    lines = original.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except ValueError:
            if index == len(lines) - 1 and not line.endswith(b'\n'):
                break
            raise
        if row.get('step', 0) <= restart_step:
            retained.append(line if line.endswith(b'\n') else line + b'\n')
    replay = b''.join(retained)
    if not checkpoints and original == replay:
        return
    archive_root = output / 'superseded'
    archive_root.mkdir(exist_ok=True)
    archive = Path(tempfile.mkdtemp(prefix=f'after-{restart_step:06d}-', dir=archive_root))
    if original != replay:
        (archive / events.name).write_bytes(original)
        with tempfile.NamedTemporaryFile(dir=output, prefix='.events-', delete=False) as handle:
            handle.write(replay)
            staged = Path(handle.name)
        staged.replace(events)
    for path in checkpoints:
        path.rename(archive / path.name)


def timestamp():
    return datetime.now(timezone.utc).isoformat()
