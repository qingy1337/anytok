"""Continue Uniform 20k with 2,000 directly proposed tokenizers on an H200."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
import torch

import anytok_online
from direct_tokenizer_policy import TokenizerPolicy
from natural_transfer import score_rows
from online_reward_scoring import BATCH_SIZE, TRIMMED_PANEL_SHA, SharedBatch, confirm, digest, domain_bpb, heldout_panel
from online_tokenizer_training import JointScorer, rollback_uncommitted_steps
from package_training_inputs import atomic_json, sha256
from pool_generalization import digest as panel_digest
from run_raw_bpb_training import prepare_code


PREFIXES = (100, 250, 500, 1000, 2000)
INITIAL_STEP = 20000
CHECKPOINT_SHA = 'badfeb74959db03947ade9023c94bd6b8e5565a22086f20a17b7f7aba4f0e35e'
INPUT_SHA = 'c33c871a858033fba1325d9fcc811e8eec43a711f983bf36e3be7ef76a75216c'
INITIAL_FILES = ('lm.pt', 'hypernet/config.json', 'hypernet/model.safetensors', 'optimizer.pt')
PANEL_REPLAY = Path(__file__).resolve().parents[1] / 'docs/online-tokenizer-handoff/heldout-panel-replay.json'


def read(path):
    return json.loads(path.read_text())


def verify_files(folder, entries):
    for name, record in entries.items():
        path = folder / name
        if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise ValueError(f'Artifact checksum mismatch: {path}')


def verify_inputs(inputs, checkpoint):
    manifest = read(checkpoint / 'manifest.json')
    if digest(manifest) != CHECKPOINT_SHA:
        raise ValueError('Expected the original Uniform 20k checkpoint manifest')
    verify_files(checkpoint, {name: manifest['files'][name] for name in INITIAL_FILES})
    files = {name: row for name, row in read(inputs.parent / 'BUNDLE_MANIFEST.json')['files'].items()
             if name.startswith(('input/model/', 'input/corpus/')) or name == 'input/recipe.json'}
    if panel_digest(files) != INPUT_SHA:
        raise ValueError('Expected the original 23-bucket corpus, recipe and base model')
    verify_files(inputs, {name.removeprefix('input/'): row for name, row in files.items()})


def load_trainer(code):
    sys.path[:0] = [str(code), str(code / 'scripts')]
    spec = importlib.util.spec_from_file_location('direct_joint_base', code / 'scripts/train_joint_smollm_input.py')
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    return trainer


class PolicyState:
    def __init__(self):
        self.model = TokenizerPolicy()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.001)
        self.baseline = None
        self.seed_rng = random.Random(45)

    def observe(self, log_prob, raw_bpb):
        advantage = 0. if self.baseline is None else raw_bpb - self.baseline
        loss = -advantage * log_prob
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not loss.isfinite() or any(p.grad is not None and not p.grad.isfinite().all() for p in self.model.parameters()):
            raise ValueError('Nonfinite direct policy update')
        self.optimizer.step()
        self.baseline = raw_bpb if self.baseline is None else .99 * self.baseline + .01 * raw_bpb
        return advantage

    def weights_dict(self):
        return self.model.state_dict()

    def state_dict(self):
        return dict(model=self.weights_dict(), optimizer=self.optimizer.state_dict(), baseline=self.baseline,
                    seed_rng=self.seed_rng.getstate())

    def restore(self, state):
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.baseline = state['baseline']
        self.seed_rng.setstate(state['seed_rng'])


def propose(scorer, policy, action_rng, shared):
    seed = policy.seed_rng.getrandbits(64)
    rejected, log_probs = [], []
    for _ in range(128):
        settings, log_prob = policy.model.draw(action_rng)
        settings['seed'] = seed
        log_probs.append(log_prob)
        try:
            batch = scorer.prepare(settings, shared)
        except ValueError as exc:
            if not any(marker in str(exc) for marker in ('bytes; this corpus provides', 'distinct eligible pieces for this request')):
                raise
            rejected.append(dict(settings=settings, reason=str(exc)))
        else:
            return batch, torch.stack(log_probs).sum(), rejected
    raise ValueError('128 consecutive policy draws were infeasible')


def prepare_update(scorer, policy, action_rng, corpus, data_rng, smoke=False):
    started = time.perf_counter()
    shared = SharedBatch(corpus, data_rng)
    if smoke:
        settings = dict(mixture=dict.fromkeys(anytok_online.DOMAINS, 1.), vocab_size=128000,
                        pretokenizer='no-regex', seed=71)
        batch, log_prob, rejected = scorer.prepare(settings, shared), None, []
    else:
        batch, log_prob, rejected = propose(scorer, policy, action_rng, shared)
    return batch, log_prob, rejected, shared.sha256, time.perf_counter() - started


def can_prefetch(step, stop_after):
    return step < stop_after and step not in PREFIXES and step % 500 != 0


def report(output, stage, step, **fields):
    platform = 'Tillicum · H200'
    account = os.environ.get('SLURM_JOB_ACCOUNT') or os.environ.get('USER', 'tillicum')
    row = dict(at_utc=datetime.now(timezone.utc).isoformat(), stage=stage, step=step, total=2000,
               platform=platform, profiles=[account], **fields)
    row['workers'] = [] if stage in ('complete', 'stopped', 'failed') else [dict(
        id=os.environ.get('SLURM_JOB_ID', str(os.getpid())), profile=account, platform=platform, state=stage)]
    atomic_json(output / 'status.json', row)
    with (output / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
    print(json.dumps(row, allow_nan=False), flush=True)


def confirm_milestone(scorer, panel, step, winner):
    with scorer.inference():
        canonical = domain_bpb(panel, score_rows(scorer.lm, scorer.source, scorer.source, panel))
        heldout = confirm(scorer, panel, winner['settings'])
    if heldout['tokenizer_sha256'] != winner['tokenizer_sha256'] or len(heldout['losses_nats']) != 156:
        raise ValueError('The held-out result changed the selected tokenizer or passage panel')
    gaps = {name: value - canonical[name] for name, value in heldout['domain_bpb'].items()}
    return dict(prefix=step, student_step=INITIAL_STEP + step, winner=winner, heldout=heldout,
                heldout_canonical_bpb=float(np.mean(list(canonical.values()))),
                heldout_raw_bpb=float(np.mean(list(heldout['domain_bpb'].values()))),
                heldout_gap_bpb=float(np.mean(list(gaps.values()))), heldout_domain_gaps=gaps)


def verify_optimizer(scorer, optimizer):
    # Uniform supplies external source embeddings and no language ID, so
    # these two embedding parameters have never acquired AdamW moments.
    unused = {scorer.hn.model.word_embeddings.weight, scorer.hn.model.lang_embedding.weight}
    for parameter in scorer.params:
        state = optimizer.state.get(parameter)
        if state is None and parameter in unused:
            continue
        if state is None or any(state[name].shape != parameter.shape for name in ('exp_avg', 'exp_avg_sq')):
            raise ValueError('AdamW state does not match the student trainable parameters')


def train(args, code, panel):
    torch.set_num_threads(1)
    if torch.__version__ != '2.15.0+cu132' or 'H200' not in torch.cuda.get_device_name():
        raise ValueError('Expected a Tillicum H200 and PyTorch 2.15.0+cu132')
    output = args.output
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    files = ('direct_tokenizer_policy.py', 'online_tokenizer_training.py', 'online_reward_scoring.py',
             'rift_eval_memory.py', 'natural_transfer.py', 'pool_generalization.py')
    identity = dict(initial_checkpoint=CHECKPOINT_SHA, initial_student_step=INITIAL_STEP,
                    final_student_step=INITIAL_STEP + PREFIXES[-1], panel=TRIMMED_PANEL_SHA,
                    trainer=sha256(code / 'source-manifest.json'), launcher=sha256(__file__),
                    sources={name: sha256(Path(__file__).with_name(name)) for name in files},
                    generator=[sha256(anytok_online.__file__), sha256(anytok_online.anytok_online_native.__file__)],
                    seed=42, action_seed=43, generation_seed=45, reward='deterministic pre-update raw BPB',
                    entropy=0, policy='direct REINFORCE', smoke=args.smoke,
                    overlap_preparation=args.overlap_preparation)
    config = output / 'run-config.json'
    if args.resume:
        if not config.is_file() or read(config) != identity:
            raise ValueError('Run configuration differs from the original')
    else:
        atomic_json(config, identity)
    latest = max((p for p in output.glob('step-*') if (p / 'state.pt').is_file()), default=None)
    state = None
    if latest:
        manifest = read(latest / 'manifest.json')
        if manifest['kind'] != 'full_restart_checkpoint' or manifest['step'] != int(latest.name.removeprefix('step-')):
            raise ValueError('Invalid joint checkpoint manifest')
        verify_files(latest, manifest['files'])
        state = torch.load(latest / 'state.pt', map_location='cpu', weights_only=True)
        if state['step'] != manifest['step']:
            raise ValueError('Checkpoint step differs from its recorded state')
    start = state['step'] if state else 0
    if start >= args.stop_after:
        raise ValueError('The requested update has already been committed')
    milestones = output / 'milestones.json'
    confirmed = state['metrics']['confirmed'] if state else []
    if args.resume:
        rollback_uncommitted_steps(output, start)
        atomic_json(milestones, confirmed)
    report(output, 'Loading Uniform student and optimizer', start)
    trainer = load_trainer(code)
    scorer = JointScorer(trainer, args.inputs, latest or args.checkpoint, read(args.inputs / 'recipe.json'))
    optimizer = torch.optim.AdamW(trainer.make_optimizer_groups(scorer.hn, 0.) + trainer.make_optimizer_groups(scorer.lm, 0.),
                                  betas=(.9, .95), eps=1e-8, fused=True)
    optimizer.load_state_dict(torch.load((latest or args.checkpoint) / 'optimizer.pt', map_location='cpu',
                                         weights_only=True, mmap=True))
    verify_optimizer(scorer, optimizer)
    policy = PolicyState()
    data_rng, action_rng = random.Random(42), np.random.default_rng(43)
    corpus = trainer.BalancedTextStream(args.inputs / 'recipe.json', args.inputs / 'corpus', state['corpus'] if state else None)
    best = state['metrics']['best'] if state else None
    if state:
        policy.restore(state['bandit'])
        data_rng.setstate(state['python_rng'])
        action_rng.bit_generator.state = state['numpy_rng']
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    stopping = False

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, request_stop)
    durations = []
    pending = None
    prepare_args = (scorer, policy, action_rng, corpus, data_rng, args.smoke)
    with ThreadPoolExecutor(max_workers=1) if args.overlap_preparation else nullcontext() as worker:
        for step in range(start + 1, args.stop_after + 1):
            began = time.perf_counter()
            prepared = pending.result() if pending else prepare_update(*prepare_args)
            pending = None
            batch, log_prob, rejected, raw_batch_sha, preparation_seconds = prepared
            preparation_wait = time.perf_counter() - began
            batch = scorer.to_device(batch)
            torch.cuda.reset_peak_memory_stats()
            reward_started = time.perf_counter()
            score = scorer.reward(batch)
            reward_seconds = time.perf_counter() - reward_started
            reward_peak = torch.cuda.max_memory_reserved() / 2**30
            advantage = None if args.smoke else policy.observe(log_prob, score['raw_bpb'])
            # The next draw depends on this reward, which is already known. Its
            # CPU preparation reads no student parameters or CUDA state.
            if worker and not stopping and can_prefetch(step, args.stop_after):
                pending = worker.submit(prepare_update, *prepare_args)
            cache_started = time.perf_counter()
            torch.cuda.empty_cache()
            cache_seconds = time.perf_counter() - cache_started
            torch.cuda.reset_peak_memory_stats()
            train_started = time.perf_counter()
            trained = scorer.update(batch, optimizer, step)
            train_seconds = time.perf_counter() - train_started
            elapsed = time.perf_counter() - began
            result = dict(step=step, student_step=INITIAL_STEP + step, raw_batch_sha256=raw_batch_sha,
                          rejected_settings=rejected, policy_advantage_bpb=advantage, reward_reserved_gib=reward_peak,
                          train_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                          train_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                          preparation_seconds=preparation_seconds, preparation_wait_seconds=preparation_wait,
                          reward_seconds=reward_seconds, cache_release_seconds=cache_seconds,
                          training_seconds=train_seconds, step_seconds=elapsed, **batch[-1], **score, **trained)
            del batch, prepared
            if best is None or result['raw_bpb'] > best['raw_bpb']:
                best = result
            durations.append(elapsed)
            report(output, 'train', step, result=result, best_step=best['step'], best_bpb=best['raw_bpb'],
                   eta_seconds=(2000 - step) * float(np.mean(durations[-40:])))
            if step in PREFIXES or args.smoke:
                confirmed.append(confirm_milestone(scorer, panel, step, best))
                atomic_json(milestones, confirmed)
            # Finish an already-started preparation before recording a restart,
            # so the saved RNGs and corpus cursor describe the committed step.
            restart = step == args.stop_after or (stopping and pending is None)
            if not args.smoke and (step % 500 == 0 or restart):
                trainer.save_checkpoint(output, step, scorer.lm, scorer.hn, optimizer, corpus, [], data_rng,
                                        action_rng, dict(best=best, confirmed=confirmed), bandit=policy, full_restart=restart)
            if restart:
                break
    corpus.close()
    report(output, 'complete' if args.smoke or step == 2000 else 'stopped', step, best_step=best['step'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True, help='Verified original input/ directory')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Downloaded Uniform step-020000 directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stop-after', type=int, default=2000, help='Updates since Uniform 20k')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--overlap-preparation', action='store_true',
                        help='Build the next tokenizer on CPU during the current GPU update')
    parser.add_argument('--smoke', action='store_true', help='Run one full 128k training update in an isolated directory')
    args = parser.parse_args()
    if args.smoke and args.resume:
        parser.error('--smoke needs a fresh output directory')
    if args.smoke:
        args.stop_after = 1
    if not 1 <= args.stop_after <= 2000:
        parser.error('--stop-after must be between 1 and 2000')
    verify_inputs(args.inputs, args.checkpoint)
    from transformers import AutoTokenizer
    canonical = AutoTokenizer.from_pretrained(args.inputs / 'model', local_files_only=True)
    panel = heldout_panel(args.inputs, canonical, PANEL_REPLAY)
    code = prepare_code(args.output.parent)
    if args.prepare_only:
        print(json.dumps(dict(stage='ready', input_sha256=INPUT_SHA, checkpoint_sha256=CHECKPOINT_SHA,
                              panel_sha256=TRIMMED_PANEL_SHA, documents=len(panel))))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.resume and any(p.name != '.run.lock' for p in args.output.iterdir()):
            parser.error('Choose an empty output directory or use --resume')
        try:
            train(args, code, panel)
        except Exception as exc:
            current = read(args.output / 'status.json')['step'] if (args.output / 'status.json').is_file() else 0
            report(args.output, 'failed', current, error=str(exc))
            raise


if __name__ == '__main__':
    main()
