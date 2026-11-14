"""One isolated, exact-artifact joint-training memory diagnostic."""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from local_memory_variants import apply_variant, BoundedOffload, move_optimizer_state
from pool_generalization import save_json, file_sha

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'modal_results/local_memory'


def load_trainer():
    """Stage the actual immutable training source, including its Rust helper."""
    code = STATE / 'code'
    base = ROOT / 'runtime/anytok_pool_20k'
    if not code.exists():
        shutil.copytree(base / 'hf_hypernet', code / 'hf_hypernet')
        (code / 'scripts').mkdir()
        (code / 'rust_utils/target/release').mkdir(parents=True)
        shutil.copy2(base / 'rust_utils/target/release/librust_utils.so', code / 'rust_utils/target/release/librust_utils.so')
        shutil.copy2(ROOT / 'runtime/anytok_pool_settings_rl_20k/hf_hypernet/settings_policy.py', code / 'hf_hypernet/settings_policy.py')
        shutil.copy2(ROOT / 'runtime/anytok_pool_regret_rl_20k/hf_hypernet/paired_reward.py', code / 'hf_hypernet/paired_reward.py')
        shutil.copy2(ROOT / 'runtime/anytok_pool_regret_rl_20k/scripts/train_joint_smollm_input.py', code / 'scripts/train_joint_smollm_input.py')
    path = code / 'scripts/train_joint_smollm_input.py'
    if file_sha(path) != file_sha(ROOT / 'runtime/anytok_pool_regret_rl_20k/scripts/train_joint_smollm_input.py'):
        raise ValueError('Staged training code changed')
    spec = importlib.util.spec_from_file_location('memory_production_trainer', path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class Measurements:
    def __init__(self, output):
        self.output = output;self.rows = [];self.current = 'initializing';self.maximum = 0
        self.offload=None

    @contextmanager
    def phase(self, name):
        self.current = name
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start = time.monotonic()
        entered = torch.cuda.memory_allocated()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated();self.maximum = max(self.maximum, peak)
            self.rows.append(dict(phase=name, seconds=time.monotonic()-start,
                entered_allocated_bytes=entered, exit_allocated_bytes=torch.cuda.memory_allocated(),
                peak_allocated_bytes=peak, peak_reserved_bytes=torch.cuda.max_memory_reserved()))
            save_json(self.output / 'phases.json', self.rows)
            print(json.dumps(self.rows[-1]), flush=True)


def gradient_reference(models, path, write):
    tensors = {prefix + name: parameter.grad for prefix, model in models for name, parameter in model.named_parameters()
               if parameter.grad is not None}
    if write:
        save_file({name:g.detach().cpu() for name,g in tensors.items()}, str(path))
        return dict(parameters=len(tensors), written=True)
    worst = 0.;difference_sq = 0.;reference_sq = 0.;equal = True;nonfinite = False
    worst_name = None
    with safe_open(path, framework='pt', device='cpu') as reader:
        if set(reader.keys()) != set(tensors):raise ValueError('Gradient parameter coverage changed')
        for name, gradient in tensors.items():
            reference = reader.get_tensor(name);actual = gradient.detach().cpu()
            nonfinite |= not bool(torch.isfinite(actual).all())
            equal &= torch.equal(reference, actual)
            # Process one full parameter at a time; never subsample elements.
            delta = actual.double() - reference.double()
            maximum = float(delta.abs().max())
            if maximum > worst:worst = maximum;worst_name = name
            difference_sq += float(delta.square().sum());reference_sq += float(reference.double().square().sum())
    return dict(parameters=len(tensors), bitwise_equal=equal, maximum_absolute_difference=worst,
        worst_parameter=worst_name, relative_l2=(difference_sq / max(reference_sq, 1e-300))**.5, nonfinite=nonfinite)


def run(args, measure):
    if torch.__version__.split('+')[0] != '2.15.0':raise ValueError('Match the running Modal torch==2.15.0')
    torch.set_num_threads(1);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    trainer = load_trainer();artifacts = args.artifacts;checkpoint = artifacts / 'checkpoint'
    receipt = json.loads((artifacts / 'download_receipt.json').read_text())
    config = json.loads((checkpoint / 'run_config.json').read_text())
    if receipt['tokenizers'] != 1000 or config['policy']['entropy_weight'] != 0:raise ValueError('Wrong complete regret experiment')
    if file_sha(STATE / 'code/scripts/train_joint_smollm_input.py') != config['training_script_sha256']:
        raise ValueError('Trainer differs from the copied checkpoint')
    canonical = trainer.AutoTokenizer.from_pretrained(artifacts / 'input/model', local_files_only=True)
    with measure.phase('load model and complete optimizer'):
        lm = trainer.AutoModelForCausalLM.from_pretrained(artifacts / 'input/model', local_files_only=True,
            dtype=torch.float32, low_cpu_mem_usage=True).to('cuda')
        trainer.separate_input_embeddings_and_freeze_output(lm)
        lm.load_state_dict(torch.load(checkpoint / 'lm.pt', map_location='cpu', weights_only=True, mmap=True))
        hn = trainer.ZettHypernet.from_pretrained(checkpoint / 'hypernet', local_files_only=True).to('cuda')
        fallback = trainer.enable_byte_fallbacks(hn, canonical);trainer.freeze_rescalers(hn)
        hn.model.activation_checkpointing = True;hn.activation_checkpoint_input = True
        source = lm.get_input_embeddings().weight.detach()
        hn_params = [p for p in hn.parameters() if p.requires_grad]
        lm_params = [p for p in lm.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainer.make_optimizer_groups(hn, 0.) + trainer.make_optimizer_groups(lm, 0.),
                                      betas=(.9, .95), eps=1e-8, fused=True)
        optimizer.load_state_dict(torch.load(checkpoint / 'optimizer.pt', map_location='cpu', weights_only=True, mmap=True))
        state = torch.load(checkpoint / 'state.pt', map_location='cpu', weights_only=True)
        pool = trainer.TokenizerPool(artifacts / 'input/pool')
        policy = trainer.SettingsTokenizerBandit([r['config'] for r in pool.entries], seed=43, entropy_weight=0.)
        policy.load_state_dict(state['bandit'])
        py_rng = random.Random();py_rng.setstate(state['python_rng'])
        corpus = trainer.BalancedTextStream(artifacts / 'input/recipe.json', artifacts / 'input/corpus', state['corpus'])
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        lm.train();hn.train();gc.collect()
    optimizer_bytes=0
    if args.offload_optimizer:
        with measure.phase('offload inactive optimizer state'):
            optimizer_bytes=move_optimizer_state(optimizer,'cpu')
    original_action = policy.sample()  # Preserve the production sampling RNG consumption.
    target, vocab = pool.load(args.tokenizer)
    names = py_rng.choices(corpus.names, weights=corpus.weights, k=128)
    texts = [corpus.next_text(name, chars=2048) for name in names]
    examples = [];controls = [];proofs = [];rejected = 0
    for name, raw in zip(names, texts):
        paired = trainer.make_paired_prefix_example(raw[:512], target, canonical, min_chars=32)
        while paired is None and rejected < 128:
            rejected += 1;raw = corpus.next_text(name, chars=2048)
            paired = trainer.make_paired_prefix_example(raw[:512], target, canonical, min_chars=32)
        if paired is None:raise ValueError('Cannot construct the original full paired batch')
        a,b,proof = paired;examples.append(a);controls.append(b);proofs.append(proof)
    forms_np, truncated = pool.forms(args.tokenizer, canonical, fallback)
    forms = torch.from_numpy(forms_np.astype(np.int64)).to('cuda')
    priors = torch.tensor([score for _,score in vocab], device='cuda', dtype=torch.float32)
    scored = sum(len(x[1]) for x in examples);byte_count = sum(x[2] for x in examples)
    if args.variant.startswith('chunked_norm'):
        from lossless_layer_norm import apply_chunked_norm
        if args.variant in ['chunked_norm_nested','chunked_norm_source','chunked_norm_compact','chunked_norm_deep','chunked_norm_mlp','chunked_norm_reentrant']:
            from chunked_hypernetwork import recompute
            apply_variant(hn, 'split_norm_offload')
            for layer in hn.model.layers:
                if args.variant == 'chunked_norm_deep':
                    from types import MethodType
                    from lossless_layer_norm import deep_checkpoint_layer
                    layer.forward=MethodType(deep_checkpoint_layer,layer)
                if args.variant in ['chunked_norm_mlp','chunked_norm_reentrant']:
                    from types import MethodType
                    from lossless_layer_norm import native_mlp_layer
                    layer.forward=MethodType(native_mlp_layer,layer)
                original_forward = layer.forward
                def wrapped(*values, original=original_forward, reentrant=args.variant=='chunked_norm_reentrant'):
                    if reentrant:
                        from torch.utils.checkpoint import checkpoint
                        return checkpoint(original,*values,use_reentrant=True,preserve_rng_state=True)
                    return recompute(original, *values)
                layer.forward = wrapped
        elif args.variant == 'chunked_norm':
            apply_variant(hn, 'split_norm_offload')
        apply_chunked_norm(hn, args.chunk_size)
        if args.variant in ['chunked_norm_source','chunked_norm_compact','chunked_norm_deep','chunked_norm_mlp','chunked_norm_reentrant']:
            from lossless_layer_norm import apply_chunked_source
            apply_chunked_source(hn,2048)
        if args.variant == 'chunked_norm_compact':
            original_output=hn.output_projection.forward
            hn.output_projection.forward=lambda x: original_output(x.contiguous())
    elif args.variant == 'chunked':
        from chunked_hypernetwork import apply_chunked
        apply_chunked(hn, args.chunk_size)
    else:
        apply_variant(hn, args.variant)
    ledger = BoundedOffload(hn_params + lm_params + [source], int((args.activation_offload_gib or args.host_budget_gib) * 2**30),
                            spill_directory=args.output/'spill' if args.variant.endswith('_disk') else None)
    measure.offload=ledger
    trainer.set_learning_rates(optimizer, state['step']);optimizer.zero_grad(set_to_none=True)
    metadata = dict(variant=args.variant, tokenizer_index=args.tokenizer, vocabulary=len(vocab), batch=128,
        microbatch=32, maximum_positions=max(p['maximum_positions'] for p in proofs),
        scored_tokens=scored, scored_bytes=byte_count, truncated_forms=truncated,
        context_sha256=hashlib.sha256(json.dumps(proofs, sort_keys=True).encode()).hexdigest(),
        checkpoint_step=state['step'], original_next_policy_action=original_action,
        diagnostic_forced_tokenizer=args.tokenizer, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        allocator_configuration=os.environ.get('PYTORCH_ALLOC_CONF','default'),
        host_budget_gib=args.host_budget_gib,
        offloaded_optimizer_bytes=optimizer_bytes,
        release_embedding_leaf=args.release_embedding_leaf,
        direct_embedding_backward=args.direct_embedding_backward,
        activation_offload_gib=args.activation_offload_gib,
        offload_lm_gradients=args.offload_lm_gradients,
        gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        source_policy_sha256=config['policy_source_sha256'], source_trainer_sha256=config['training_script_sha256'])
    metadata.update(diagnostic_variant_sha256=file_sha(ROOT/'scripts/local_memory_variants.py'),
                    diagnostic_trial_sha256=file_sha(Path(__file__)))
    if args.variant == 'chunked':
        metadata.update(chunk_size=args.chunk_size,
                        chunked_source_sha256=file_sha(ROOT/'scripts/chunked_hypernetwork.py'))
    if args.variant.startswith('chunked_norm'):
        metadata.update(chunk_rows=args.chunk_size,
                        chunked_source_sha256=file_sha(ROOT/'scripts/lossless_layer_norm.py'))
        if args.variant in ['chunked_norm_mlp','chunked_norm_reentrant']:metadata['mlp_source_sha256']=file_sha(ROOT/'scripts/lossless_mlp.py')
    save_json(args.output / 'inputs.json', metadata)
    with measure.phase('canonical no-gradient forward'):
        canonical_nats = trainer.canonical_loss_nats(lm, controls, source, 32, trainer.make_lm_microbatch)
    loss_nats = 0.
    with ledger.context('offload' in args.variant or args.variant.endswith('_disk') or args.activation_offload_gib>0):
        with measure.phase('hypernetwork forward'):
            with torch.autocast('cuda', dtype=torch.bfloat16):generated, output, _ = hn(forms, priors, source, None, deterministic=False)
            if output is not None or not torch.isfinite(generated).all():raise ValueError('Invalid hypernetwork output')
            leaf = generated.detach().float().requires_grad_(True)
        with measure.phase('four LM microbatches forward and backward'):
            for start in range(0, 128, 32):
                micro = examples[start:start+32]
                embeds,mask,labels,positions = trainer.make_lm_microbatch(micro, leaf, source, 'cuda')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = lm(inputs_embeds=embeds,attention_mask=mask,position_ids=positions,labels=labels,use_cache=False).loss
                count = sum(len(x[1]) for x in micro);loss_nats += float(loss.detach()) * count
                (loss * count / scored).backward()
                del embeds,mask,labels,positions,loss
        staged_gradients=[]
        if args.offload_lm_gradients:
            with measure.phase('offload completed LM gradients'):
                for parameter in lm_params:
                    if parameter.grad is not None:
                        staged_gradients.append((parameter,parameter.grad.to('cpu')))
                        parameter.grad=None
                metadata['offloaded_lm_gradient_bytes']=sum(g.numel()*g.element_size() for _,g in staged_gradients)
                save_json(args.output/'inputs.json',metadata)
        with measure.phase('lexical loss'):
            lexical = trainer.input_lexical_loss(generated,forms,source,hn.config.pad_token_id,0)
            lexical_value = float(lexical.detach())
        embedding_gradient=leaf.grad.detach()
        if args.release_embedding_leaf:del leaf
        with measure.phase('hypernetwork backward'):
            if args.direct_embedding_backward:
                torch.autograd.backward((generated,lexical),(embedding_gradient,lexical.new_tensor(.5)))
            else:
                ((generated.float() * embedding_gradient).sum() + .5 * lexical).backward()
    if staged_gradients:
        with measure.phase('restore complete LM gradients'):
            for parameter,gradient in staged_gradients:parameter.grad=gradient.to(parameter.device)
            staged_gradients.clear()
    rng_digest = hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()
    raw_bpb,canonical_bpb,gap = trainer.paired_gap_bpb(loss_nats,canonical_nats,byte_count)
    numeric = dict(raw_bpb=raw_bpb,canonical_bpb=canonical_bpb,gap_bpb=gap,lexical_loss=lexical_value,
                   cuda_rng_sha256=rng_digest)
    if args.reference:
        reference = args.reference
        if args.write_reference:
            reference.mkdir(parents=True,exist_ok=True)
            save_file({'generated':generated.detach().cpu()},str(reference/'generated.safetensors'))
            save_json(reference/'numeric.json', numeric);save_json(reference/'inputs.json', metadata)
        else:
            before = json.loads((reference/'inputs.json').read_text())
            for key in ['context_sha256','checkpoint_step','vocabulary','scored_tokens','scored_bytes']:
                if before[key] != metadata[key]:raise ValueError('Reference data changed: '+key)
            with safe_open(reference/'generated.safetensors',framework='pt',device='cpu') as reader:
                expected=reader.get_tensor('generated');actual=generated.detach().cpu()
                numeric['output_bitwise_equal'] = torch.equal(expected,actual)
                numeric['output_maximum_absolute_difference'] = float((expected-actual).abs().max())
            numeric['reference_numeric'] = json.loads((reference/'numeric.json').read_text())
        numeric['gradients'] = gradient_reference([('hn.',hn),('lm.',lm)],reference/'gradients.safetensors',args.write_reference)
    with measure.phase('gradient clipping and complete optimizer update'):
        norm = torch.nn.utils.clip_grad_norm_(hn_params+lm_params,.1,foreach=False)
        if not torch.isfinite(norm):raise FloatingPointError('Invalid global gradient')
        if args.offload_optimizer:move_optimizer_state(optimizer,'cuda')
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        policy_metrics = policy.update(args.tokenizer,gap)
    corpus.close()
    return dict(outcome='complete',inputs=metadata,numeric=numeric,offload=ledger.statistics(),
        global_gradient_norm=float(norm),peak_allocated_bytes=measure.maximum,phases=measure.rows,
        policy_updated=policy_metrics['policy_updates'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,default=STATE/'artifacts')
    parser.add_argument('--variant',choices=['baseline','offload','fine','fine_offload','split_norm_offload','split_norm_disk','chunked','chunked_norm','chunked_norm_baseline','chunked_norm_nested','chunked_norm_source','chunked_norm_compact','chunked_norm_deep','chunked_norm_mlp','chunked_norm_reentrant'],required=True)
    parser.add_argument('--chunk-size',type=int,default=2048)
    parser.add_argument('--tokenizer',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--write-reference',action='store_true')
    parser.add_argument('--host-budget-gib',type=float,default=64.)
    parser.add_argument('--offload-optimizer',action='store_true')
    parser.add_argument('--release-embedding-leaf',action='store_true')
    parser.add_argument('--direct-embedding-backward',action='store_true')
    parser.add_argument('--activation-offload-gib',type=float,default=0.)
    parser.add_argument('--offload-lm-gradients',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);measure=Measurements(args.output)
    torch.cuda.memory._record_memory_history(max_entries=20000,stacks='python')
    start=time.monotonic()
    try:result=run(args,measure)
    except torch.OutOfMemoryError as error:
        torch.cuda.memory._dump_snapshot(str(args.output/'oom_snapshot.pickle'))
        result=dict(outcome='out_of_memory',phase=measure.current,error=str(error),phases=measure.rows,
                    peak_allocated_bytes=max(measure.maximum,torch.cuda.max_memory_allocated()))
    except Exception as error:
        save_json(args.output/'result.json',dict(outcome='error',phase=measure.current,error=str(error)));raise
    if 'inputs' not in result and (args.output/'inputs.json').is_file():
        result['inputs']=json.loads((args.output/'inputs.json').read_text())
    if measure.offload is not None:result['offload']=measure.offload.statistics()
    snapshot = torch.cuda.memory._snapshot()
    allocations = [r for device in snapshot.get('device_traces', []) for r in device if r.get('action') in ['alloc','oom']]
    largest = []
    for row in sorted(allocations,key=lambda r:r.get('size',0),reverse=True)[:20]:
        largest.append(dict(action=row['action'], bytes=row.get('size'),
            frames=[f for f in row.get('frames',[]) if 'anytok' in f.get('filename','')]))
    result['largest_individual_allocations'] = largest
    result['elapsed_seconds']=time.monotonic()-start;save_json(args.output/'result.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
