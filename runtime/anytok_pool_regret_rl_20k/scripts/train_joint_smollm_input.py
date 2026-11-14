"""Jointly train SmolLM2 and an input-only ZeTT hypernetwork on fresh tokenizers.

Each optimizer update samples one saved pool tokenizer using adversarial REINFORCE.
The continuation and output head always use canonical SmolLM2 tokens.
"""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
rust_library = REPO / "rust_utils" / "target" / "release" / "librust_utils.so"
if not rust_library.is_file():
    raise FileNotFoundError(f"Build ZeTT's Rust sampler first: {rust_library}")
rust_spec = importlib.util.spec_from_file_location("rust_utils", rust_library)
rust_utils = importlib.util.module_from_spec(rust_spec)
sys.modules["rust_utils"] = rust_utils
rust_spec.loader.exec_module(rust_utils)

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import AutoModelForCausalLM, AutoTokenizer

from hf_hypernet.pool_sampling import TokenizerPool
from hf_hypernet.settings_policy import SettingsTokenizerBandit
from hf_hypernet.paired_reward import make_paired_prefix_example, canonical_loss_nats, paired_gap_bpb
from hf_hypernet.identity_warmup import freeze_rescalers
from hf_hypernet.joint_training import (
    BalancedTextStream, INITIAL_QUEUE_BATCHES, QUEUE_BATCH_SIZE, SPLIT_REGEX,
    VOCAB_SIZE, draw_tokenizer, input_lexical_loss, make_lm_microbatch,
    prime_sampler, zett_main_learning_rate,
)
from hf_hypernet.modeling_hypernet import ZettHypernet
from hf_hypernet.smollm_setup import separate_input_embeddings_and_freeze_output
from hf_hypernet.tokenizer_inputs import build_surface_forms, enable_byte_fallbacks

SPECIAL_TOKEN = "<|endoftext|>"
SPECIAL_ID = 0
LEGACY_MICROBATCH_4_SCRIPT_SHA256 = "a4bf402ab2280ebb8fd4256de55e6727479c2f8d9eba2a2c2d003204ee4ce631"
LEGACY_MIN_CHARS_64_SCRIPT_SHA256 = "37fe7c82e1459e922ad9a98ae09e52a666554d6925804ea7e6149faae68feca5"


def make_optimizer_groups(module, learning_rate):
    decay, no_decay = [], []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith("bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [{"params": decay, "lr": learning_rate, "weight_decay": 0.01},
            {"params": no_decay, "lr": learning_rate, "weight_decay": 0.0}]


def set_learning_rates(optimizer, update):
    hn_lr = zett_main_learning_rate(update, 6e-5)
    lm_lr = zett_main_learning_rate(update, 1e-6)
    for group in optimizer.param_groups[:2]:
        group["lr"] = hn_lr
    for group in optimizer.param_groups[2:]:
        group["lr"] = lm_lr
    return hn_lr, lm_lr


def tokenizer_from_vocab(vocab):
    import tokenizers
    tok = Tokenizer(models.Unigram([tuple(row) for row in vocab]))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(tokenizers.Regex(SPLIT_REGEX), "removed", invert=True),
        pre_tokenizers.ByteLevel(False, False),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def build_eval_set(eval_dir, target_tokenizer, source_tokenizer, per_bucket=4):
    """Use identical raw prefix/continuation boundaries for both input methods."""
    generated_examples, canonical_examples = [], []
    stats = {}
    for path in sorted(Path(eval_dir).glob("*.jsonl.gz")):
        if path.stem.removesuffix(".jsonl") not in {
            "web-en", "web-es", "web-fr", "web-ru", "web-zh", "web-ja",
            "web-ar", "web-hi", "wiki-en", "wiki-es", "wiki-fr", "wiki-ru",
            "wiki-zh", "wiki-ja", "wiki-ar", "wiki-hi", "code", "papers-en",
            "math-en", "qa-en", "books-en", "medical-en", "legal-en",
        }:
            continue
        accepted = 0
        with __import__("gzip").open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                text = json.loads(line)["text"]
                if len(text) < 80:
                    continue
                target_example = None
                for chars in (240, 200, 160, 120, 80):
                    raw = text[:chars]
                    if len(raw) < 80:
                        break
                    paired = make_paired_prefix_example(raw, target_tokenizer, source_tokenizer,
                                                        max_positions=128, min_chars=len(raw))
                    if paired is None:
                        continue
                    target_example, canonical_example, _ = paired
                    prefix_ids = canonical_example[0]
                    break
                if target_example is None:
                    continue
                generated_examples.append(target_example)
                canonical_examples.append((prefix_ids, target_example[1], target_example[2]))
                accepted += 1
                if accepted == per_bucket:
                    break
        if accepted < per_bucket:
            raise ValueError(f"Only {accepted} usable held-out examples in {path}")
        stats[path.name] = accepted
    if len(stats) != 23:
        raise ValueError(f"Expected 23 held-out buckets, found {len(stats)}")
    return generated_examples, canonical_examples


@torch.inference_mode()
def evaluate(lm, hn, source, source_tokenizer, eval_specs, fallback, eval_dir):
    lm.eval()
    hn.eval()
    results = []
    device = source.device
    for vocab in eval_specs:
        tok = tokenizer_from_vocab(vocab)
        examples, canonical = build_eval_set(eval_dir, tok, source_tokenizer)
        forms_np, trunc = build_surface_forms(
            [piece for piece, _ in vocab], source_tokenizer, 10, fallback,
            {SPECIAL_TOKEN: SPECIAL_ID},
        )
        forms = torch.from_numpy(forms_np.astype(np.int64)).to(device)
        priors = torch.tensor([score for _, score in vocab], device=device,
                              dtype=torch.float32)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            generated, _, _ = hn(forms, priors, source, None, deterministic=True)
        if not torch.isfinite(generated).all():
            raise FloatingPointError("Non-finite held-out embeddings")
        nats = [0.0, 0.0]
        bytes_total = sum(x[2] for x in examples)
        for idx, sequence_examples in enumerate((examples, canonical)):
            table = generated if idx == 0 else source
            for start in range(0, len(sequence_examples), 4):
                micro = sequence_examples[start:start + 4]
                embeds, mask, labels, positions = make_lm_microbatch(
                    micro, table, source, device
                )
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    loss = lm(inputs_embeds=embeds, attention_mask=mask,
                              position_ids=positions, labels=labels,
                              use_cache=False).loss
                nats[idx] += float(loss) * sum(len(x[1]) for x in micro)
        results.append({"generated_bpb": nats[0] / bytes_total / math.log(2),
                        "canonical_bpb": nats[1] / bytes_total / math.log(2),
                        "examples": len(examples), "bytes": bytes_total,
                        "truncated_forms": trunc})
        del generated
    lm.train()
    hn.train()
    return results


def prune_checkpoints(output, protected_steps, before_save=False):
    """Keep milestone restarts plus at most one disposable restart point."""
    completed = sorted(
        p for p in Path(output).glob("step-*")
        if p.is_dir() and (p / "state.pt").is_file()
    )
    if not protected_steps:
        if not before_save:
            for stale in completed[:-2]:
                shutil.rmtree(stale)
        return
    protected = [p for p in completed
                 if int(p.name.removeprefix("step-")) in protected_steps]
    disposable = [p for p in completed if p not in protected]
    # Once a milestone exists, it is a safe restart point if the next write
    # fails. Free the previous disposable checkpoint before allocating ~21 GB.
    stale = disposable if before_save and protected else disposable[:-1]
    for path in stale:
        shutil.rmtree(path)


def save_checkpoint(output, step, lm, hn, optimizer, corpus, queue_batches,
                    py_rng, noise_rng, metrics, protected_steps=(), bandit=None,
                    full_restart=True):
    output = Path(output)
    final = output / f"step-{step:06d}"
    if final.exists():
        raise FileExistsError(final)
    with tempfile.TemporaryDirectory(prefix=".writing-", dir=output) as temporary:
        stage = Path(temporary)
        hn.save_pretrained(stage / "hypernet", safe_serialization=True)
        torch.save(lm.state_dict(), stage / "lm.pt")
        torch.save(bandit.weights_dict(), stage / 'policy.pt')
        state = {
            "step": step, "corpus": corpus.state_dict(),
            "queue_batches": queue_batches,
            "python_rng": py_rng.getstate(),
            "numpy_rng": noise_rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "metrics": metrics,
            "bandit": bandit.state_dict(),
        }
        if full_restart:
            torch.save(optimizer.state_dict(), stage / "optimizer.pt")
            torch.save(state, stage / "state.pt")
        os.replace(stage, final)


def record(output, event):
    line = json.dumps(event, sort_keys=True)
    print(line, flush=True)
    with (Path(output) / "events.jsonl").open("a") as handle:
        handle.write(line + "\n")


def verify_corpus(data_dir, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    if len(manifest) != 23:
        raise ValueError("Expected checksums for all 23 corpus buckets")
    for name, item in manifest.items():
        path = Path(data_dir) / name / "train.jsonl.gz"
        if not path.is_file() or path.stat().st_size != item["compressed_bytes"]:
            raise ValueError(f"Missing or incomplete staged training file: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != item["sha256"]:
            raise ValueError(f"Training file checksum mismatch: {path}")


def check_resume_settings(saved, requested, allow_microbatch_migration,
                          checkpoint_step, allow_min_chars_migration=False):
    """Allow only the two reviewed changes at their exact checkpoints."""
    if saved == requested:
        return None
    same_except_min_chars_and_script = (
        set(requested) == set(saved) | {"min_data_chars"}
        and all(saved.get(key) == value for key, value in requested.items()
                if key not in {"min_data_chars", "training_script_sha256"})
    )
    if (allow_min_chars_migration and checkpoint_step == 3000
            and saved.get("microbatch_size") == 32
            and saved.get("min_data_chars") is None
            and requested.get("min_data_chars") == 32
            and saved.get("training_script_sha256")
            == LEGACY_MIN_CHARS_64_SCRIPT_SHA256
            and same_except_min_chars_and_script):
        return {"stage": "settings_change", "step": checkpoint_step,
                "min_data_chars_before": 64, "min_data_chars_after": 32,
                "training_script_sha256_before": saved["training_script_sha256"],
                "training_script_sha256_after": requested["training_script_sha256"],
                "reason": "Keep short non-English examples within 128 token positions"}
    same_except_pass_size_and_script = all(
        saved.get(key) == value for key, value in requested.items()
        if key not in {"microbatch_size", "training_script_sha256"}
    ) and set(saved) == set(requested)
    if not (allow_microbatch_migration and checkpoint_step == 1000
            and saved.get("microbatch_size") == 4
            and requested["microbatch_size"] == 32
            and saved.get("training_script_sha256") == LEGACY_MICROBATCH_4_SCRIPT_SHA256
            and same_except_pass_size_and_script):
        raise ValueError("Resume settings differ from the original run")
    return {"stage": "settings_change", "step": checkpoint_step,
            "microbatch_size_before": 4, "microbatch_size_after": 32,
            "training_script_sha256_before": saved["training_script_sha256"],
            "training_script_sha256_after": requested["training_script_sha256"],
            "reason": "Same 128-example update in four GPU passes instead of 32"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-pool", type=Path, required=True)
    parser.add_argument("--eval-tokenizers", type=Path, required=True,
                        help="Frozen sanity tokenizer specs from the uniform run")
    parser.add_argument("--policy-lr", type=float, default=0.001)
    parser.add_argument("--policy-entropy", type=float, default=0.0)
    parser.add_argument("--policy-baseline-decay", type=float, default=0.99)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-1.7B")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--microbatch-size", type=int, default=4)
    parser.add_argument("--allow-microbatch-migration", action="store_true",
                        help="Permit the audited 4-to-32 change at checkpoint 1000")
    parser.add_argument("--allow-min-chars-migration", action="store_true",
                        help="Permit the audited 64-to-32 minimum at checkpoint 3000")
    parser.add_argument("--max-updates-this-run", type=int, default=None)
    parser.add_argument("--stop-at-update", type=int, default=None,
                        help="Absolute update to stop at, including after a resume")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--protected-steps", type=int, nargs="*", default=[],
                        help="Full checkpoints never pruned (e.g. 5000 10000 15000)")
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("A CUDA GPU is required for this training run")
    if args.save_every < 1 or args.eval_every < 1:
        parser.error("Checkpoint and evaluation intervals must be positive")
    if (any(step < 1 for step in args.protected_steps)
            or len(set(args.protected_steps)) != len(args.protected_steps)):
        parser.error("Protected steps must be unique positive updates")
    if any(step % args.save_every for step in args.protected_steps):
        parser.error("Protected steps must be divisible by --save-every")
    if args.stop_at_update and any(step > args.stop_at_update
                                   for step in args.protected_steps):
        parser.error("Protected steps must not exceed --stop-at-update")
    if args.max_updates_this_run is not None and args.max_updates_this_run < 1:
        parser.error("--max-updates-this-run must be positive")
    if args.stop_at_update is not None and not 1 <= args.stop_at_update <= 190_000:
        parser.error("--stop-at-update must be between 1 and 190000")
    if args.max_updates_this_run is not None and args.stop_at_update is not None:
        parser.error("Choose only one update limit")
    if args.microbatch_size < 1 or 128 % args.microbatch_size:
        parser.error("--microbatch-size must divide the 128-example update")
    if args.allow_microbatch_migration and not args.resume:
        parser.error("--allow-microbatch-migration requires --resume")
    if args.allow_min_chars_migration and not args.resume:
        parser.error("--allow-min-chars-migration requires --resume")
    if args.resume:
        if not args.output.is_dir():
            parser.error("--resume requires an existing run directory")
    elif args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty unless --resume is set")
    args.output.mkdir(parents=True, exist_ok=True)
    lock_handle = (args.output / ".run.lock").open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Another process is already training in this output directory") from exc
    stop_requested = False

    def request_stop(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print(json.dumps({"stage": "stop_requested", "signal": signum}), flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    verify_corpus(args.data_dir, args.data_manifest)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    py_rng = random.Random(args.seed)
    noise_rng = np.random.default_rng(args.seed)
    pool = TokenizerPool(args.tokenizer_pool)
    bandit = SettingsTokenizerBandit([entry['config'] for entry in pool.entries],
                                    args.seed+1, args.policy_lr,
                                    args.policy_entropy, args.policy_baseline_decay)
    recipe_hash = hashlib.sha256(args.recipe.read_bytes()).hexdigest()
    manifest_hash = hashlib.sha256(args.data_manifest.read_bytes()).hexdigest()
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    helpers_hash = hashlib.sha256((REPO / "hf_hypernet" / "joint_training.py").read_bytes()).hexdigest()
    config = {"recipe_sha256": recipe_hash, "model": args.model,
              "data_manifest_sha256": manifest_hash,
              "training_script_sha256": script_hash,
              "joint_helpers_sha256": helpers_hash,
              "warmup_checkpoint": str(args.warmup_checkpoint.resolve()),
              "main_updates": 190_000, "batch_size": 128,
              "microbatch_size": args.microbatch_size,
              "tokenizer_sampling": "adversarial_shared_settings_mlp_per_optimizer_update",
              "policy": bandit.settings,
              "policy_reward": "paired_pre_model_update_transferred_bpb_minus_current_canonical_bpb",
              "pairing_protocol": "nearest_valid_canonical_midpoint_boundary_joint_128_positions_v1",
              "paired_reward_source_sha256": hashlib.sha256((REPO / "hf_hypernet/paired_reward.py").read_bytes()).hexdigest(),
              "policy_source_sha256": hashlib.sha256((REPO / 'hf_hypernet/settings_policy.py').read_bytes()).hexdigest(),
              "fixed_eval_sha256": hashlib.sha256(args.eval_tokenizers.read_bytes()).hexdigest(),
              "pool_plan_sha256": pool.plan_hash, "pool_size": len(pool.entries),
              "pool_index_sha256": pool.index_hash,
              "pool_sampling_sha256": hashlib.sha256((REPO / "hf_hypernet/pool_sampling.py").read_bytes()).hexdigest(),
              "pool_priors": "native_unigram_scores_otherwise_zero",
              "max_pieces": 10,
              "hn_peak_lr": 6e-5, "lm_peak_lr": 1e-6,
              "lr_ramp_updates": 10_000, "lr_final_fraction": 0.1,
              "lexical_loss_weight": 0.5, "sequence_length": 128,
              "sampler_text_chars": 2048, "data_snippet_chars": 512,
              "min_data_chars": 32,
              "seed": args.seed}
    if args.protected_steps:
        config["protected_steps"] = sorted(args.protected_steps)
    config_path = args.output / "run_config.json"
    baseline = json.loads((args.tokenizer_pool.parent/'uniform_run_config.json').read_text())
    paired_keys = ['recipe_sha256','data_manifest_sha256','main_updates','batch_size',
                   'microbatch_size','pool_plan_sha256','pool_size','pool_index_sha256',
                   'pool_sampling_sha256','pool_priors','max_pieces','hn_peak_lr','lm_peak_lr',
                   'lr_ramp_updates','lr_final_fraction','lexical_loss_weight','sequence_length',
                   'sampler_text_chars','data_snippet_chars','min_data_chars','seed']
    if any(config[key] != baseline[key] for key in paired_keys):
        raise ValueError('RL settings differ from the uniform baseline')
    if args.resume:
        completed = sorted(p for p in args.output.glob("step-*")
                           if p.is_dir() and (p / "state.pt").is_file())
        if not completed:
            raise ValueError("No complete checkpoint to resume")
        checkpoint_step = int(completed[-1].name.removeprefix("step-"))
        change = check_resume_settings(
            json.loads(config_path.read_text()), config,
            args.allow_microbatch_migration, checkpoint_step,
            args.allow_min_chars_migration,
        )
        if change is not None:
            with tempfile.NamedTemporaryFile(mode="w", dir=args.output,
                                             prefix=".run-config-", delete=False) as handle:
                json.dump(config, handle, indent=2, sort_keys=True)
                handle.write("\n")
                pending_config = Path(handle.name)
            os.replace(pending_config, config_path)
            record(args.output, change)
    else:
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    source_tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if source_tokenizer.convert_tokens_to_ids(SPECIAL_TOKEN) != SPECIAL_ID:
        raise ValueError("This runner expects SmolLM2's end-of-text token at ID zero")
    lm = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to("cuda")
    if lm.config.eos_token_id != SPECIAL_ID or lm.config.bos_token_id != SPECIAL_ID:
        raise ValueError("SmolLM2 model special IDs differ from the tokenizer")
    separate_input_embeddings_and_freeze_output(lm)
    completed = sorted(p for p in args.output.glob("step-*")
                       if p.is_dir() and (p / "state.pt").is_file())
    if args.resume and not completed:
        raise ValueError("No complete checkpoint to resume")
    last = completed[-1] if args.resume else None
    hn = ZettHypernet.from_pretrained(
        last / "hypernet" if last else args.warmup_checkpoint,
        local_files_only=True,
    ).to("cuda")
    fallback = enable_byte_fallbacks(hn, source_tokenizer)
    freeze_rescalers(hn)
    hn.model.activation_checkpointing = True
    hn.activation_checkpoint_input = True
    if not hn.config.hn_add_inter_token_attention or not hn.config.hn_embed_target_priors:
        raise ValueError("Expected vocabulary attention and target priors")
    assert hn.config.hn_surface_maxlen == 10
    assert not lm.get_input_embeddings().weight.requires_grad
    assert not lm.get_output_embeddings().weight.requires_grad
    source = lm.get_input_embeddings().weight.detach()
    hn_params = [p for p in hn.parameters() if p.requires_grad]
    lm_params = [p for p in lm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        make_optimizer_groups(hn, 0.0) + make_optimizer_groups(lm, 0.0),
        betas=(0.9, 0.95), eps=1e-8, fused=True,
    )
    corpus_state = None
    queue_batches = []
    start_update = 0
    if last:
        lm.load_state_dict(torch.load(last / "lm.pt", map_location="cpu",
                                      weights_only=True))
        optimizer.load_state_dict(torch.load(
            last / "optimizer.pt", map_location="cpu", weights_only=True
        ))
        state = torch.load(last / "state.pt", map_location="cpu", weights_only=True)
        start_update = state["step"]
        corpus_state = state["corpus"]
        queue_batches = state["queue_batches"]
        py_rng.setstate(state["python_rng"])
        noise_rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        bandit.load_state_dict(state['bandit'])
        if bandit.updates != start_update:
            raise ValueError('Policy and model update counters differ')
    corpus = BalancedTextStream(args.recipe, args.data_dir, corpus_state)
    sampler = rust_utils.TokenizerSampler()
    if last:
        prime_sampler(sampler, queue_batches)
    else:
        for _ in range(INITIAL_QUEUE_BATCHES):
            names = py_rng.choices(corpus.names, weights=corpus.weights,
                                   k=QUEUE_BATCH_SIZE)
            batch = [corpus.next_text(name, chars=2048) for name in names]
            queue_batches.append(batch)
        prime_sampler(sampler, queue_batches)

    specs_path = args.output / "heldout_tokenizers.json"
    initial_eval_path = args.output / "initial_eval.json"
    if last:
        eval_specs = json.loads(specs_path.read_text())
        initial_results = json.loads(initial_eval_path.read_text())
    else:
        eval_sampler = rust_utils.TokenizerSampler()
        prime_sampler(eval_sampler, queue_batches)
        eval_specs = []
        for _ in range(2):
            _, vocab, _ = draw_tokenizer(eval_sampler, queue_batches[0], noise_rng,
                                         SPECIAL_TOKEN)
            eval_specs.append(vocab)
        # Keep the original RNG consumption, then use the baseline's fixed panel.
        eval_specs = json.loads(args.eval_tokenizers.read_text())
        specs_path.write_text(json.dumps(eval_specs, ensure_ascii=False) + "\n")
    lm.train()
    hn.train()
    if not last:
        initial_results = evaluate(lm, hn, source, source_tokenizer,
                                   eval_specs, fallback, args.eval_dir)
        initial_eval_path.write_text(json.dumps(initial_results, indent=2) + "\n")
        record(args.output, {"stage": "initial_eval",
                             "results": initial_results})
    if args.stop_at_update is not None:
        stop_update = args.stop_at_update
    elif args.max_updates_this_run is not None:
        stop_update = min(190_000, start_update + args.max_updates_this_run)
    else:
        stop_update = 190_000
    if stop_update <= start_update:
        raise ValueError("Requested stopping update has already been reached")
    if not last:
        save_checkpoint(args.output, 0, lm, hn, optimizer, corpus, queue_batches,
                        py_rng, noise_rng, {'stage':'initial_checkpoint','step':0},
                        args.protected_steps, bandit=bandit, full_restart=True)
    for update in range(start_update, stop_update):
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        names = py_rng.choices(corpus.names, weights=corpus.weights, k=128)
        texts = [corpus.next_text(name, chars=2048) for name in names]
        pool_index = bandit.sample()
        action_probability = float(bandit.probabilities()[pool_index])
        target_tokenizer, vocab = pool.load(pool_index)
        noise = None
        queue_batches.pop(0)
        queue_batches.append(texts)
        examples = [];canonical_examples = [];pairing_evidence = []
        rejected = 0
        for name, raw in zip(names, texts):
            paired = make_paired_prefix_example(raw[:512], target_tokenizer, source_tokenizer, min_chars=32)
            while paired is None and rejected < 128:
                rejected += 1
                raw = corpus.next_text(name, chars=2048)
                paired = make_paired_prefix_example(raw[:512], target_tokenizer, source_tokenizer, min_chars=32)
            if paired is None:
                raise ValueError("Too many non-roundtripping or oversized paired examples")
            example, canonical_example, evidence = paired
            examples.append(example);canonical_examples.append(canonical_example);pairing_evidence.append(evidence)
        forms_np, truncated = pool.forms(pool_index, source_tokenizer, fallback)
        forms = torch.from_numpy(forms_np.astype(np.int64)).to("cuda")
        priors = torch.tensor([score for _, score in vocab], device="cuda",
                              dtype=torch.float32)
        total_scored = sum(len(x[1]) for x in examples)
        hn_lr, lm_lr = set_learning_rates(optimizer, update)
        optimizer.zero_grad(set_to_none=True)
        if any(a[1:] != b[1:] for a, b in zip(examples, canonical_examples)):
            raise ValueError('Canonical and native targets or continuation bytes differ')
        # The control forward uses current pre-update LM weights, the exact same
        # targets/bytes, and no gradients. Its RNG consumption is restored.
        canonical_nats = canonical_loss_nats(lm, canonical_examples, source,
                                             args.microbatch_size, make_lm_microbatch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated, output_embeddings, _ = hn(
                forms, priors, source, None, deterministic=False
            )
        if output_embeddings is not None or not torch.isfinite(generated).all():
            raise FloatingPointError("Hypernetwork produced invalid input embeddings")
        leaf = generated.detach().float().requires_grad_(True)
        loss_nats = 0.0
        for start in range(0, 128, args.microbatch_size):
            micro = examples[start:start + args.microbatch_size]
            embeds, mask, labels, positions = make_lm_microbatch(
                micro, leaf, source, "cuda"
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = lm(inputs_embeds=embeds, attention_mask=mask,
                          position_ids=positions, labels=labels,
                          use_cache=False).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite LM loss")
            count = sum(len(x[1]) for x in micro)
            loss_nats += float(loss.detach()) * count
            (loss * count / total_scored).backward()
            del embeds, mask, labels, positions, loss
        if leaf.grad is None or not torch.isfinite(leaf.grad).all():
            raise FloatingPointError("Non-finite accumulated embedding gradient")
        lexical = input_lexical_loss(
            generated, forms, source, hn.config.pad_token_id,
            SPECIAL_ID,
        )
        if not torch.isfinite(lexical):
            raise FloatingPointError("Non-finite input lexical loss")
        lexical_value = float(lexical.detach())
        ((generated.float() * leaf.grad.detach()).sum() + 0.5 * lexical).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(hn_params + lm_params, 0.1,
                                                    foreach=False)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("Non-finite global gradient norm")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scored_bytes = sum(x[2] for x in examples)
        train_bpb, canonical_bpb, reward_bpb = paired_gap_bpb(loss_nats, canonical_nats, scored_bytes)
        policy_metrics = bandit.update(pool_index, reward_bpb)
        del generated, leaf, lexical, forms, priors
        step = update + 1
        metrics = {"stage": "train", "step": step,
                   "loss_nats_per_token": loss_nats / total_scored,
                   "input_lexical_loss": lexical_value,
                   "scored_tokens": total_scored,
                   "scored_bytes": scored_bytes,
                   "train_bpb": train_bpb,
                   "paired_canonical_bpb": canonical_bpb,
                   "paired_gap_bpb": reward_bpb,
                   "paired_valid_examples": len(pairing_evidence),
                   "paired_maximum_positions": max(r['maximum_positions'] for r in pairing_evidence),
                   "paired_batch_sha256": hashlib.sha256(json.dumps(pairing_evidence, sort_keys=True).encode()).hexdigest(),
                   "policy_action_probability": action_probability,
                   **policy_metrics,
                   "hn_lr": hn_lr, "lm_lr": lm_lr,
                   "gradient_norm_before_clip": float(grad_norm),
                   "seconds": round(time.perf_counter() - started, 3),
                   "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                   "tokenizer_pool_index": pool_index, "tokenizer_config_index": pool.entries[pool_index]["config"]["index"],
                   "tokenizer_vocab_size": len(vocab), "truncated_forms": truncated,
                   "rejected_examples": rejected}
        record(args.output, metrics)
        should_eval = step % args.eval_every == 0 or step == stop_update or stop_requested
        should_save = step % args.save_every == 0 or step == stop_update or stop_requested
        if should_eval:
            results = evaluate(lm, hn, source, source_tokenizer, eval_specs,
                               fallback, args.eval_dir)
            record(args.output, {"stage": "eval", "step": step,
                                 "results": results})
            for metric in ("generated_bpb", "canonical_bpb"):
                if all(result[metric] > 2 * baseline[metric]
                       for result, baseline in zip(results, initial_results)):
                    raise RuntimeError(
                        f"Both held-out tokenizers more than doubled {metric}; "
                        "stopping before overwriting the last good checkpoint"
                    )
        if should_save:
            save_checkpoint(args.output, step, lm, hn, optimizer, corpus,
                            queue_batches, py_rng, noise_rng, metrics,
                            args.protected_steps, bandit=bandit,
                            full_restart=(step % 2000 == 0 or step == stop_update or stop_requested))
            record(args.output, {"stage": "checkpoint", "step": step})
        if stop_requested:
            stop_update = step
            break
    corpus.close()
    record(args.output, {"stage": "complete" if stop_update == 190_000 else "paused_at",
                         "step": stop_update})


if __name__ == "__main__":
    main()
