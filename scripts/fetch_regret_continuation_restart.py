"""Fetch the regret run's step-8,000 full restart from Hugging Face into a resumable run folder.

Runs on the cloud target (a Tillicum login node), so the 23.5 GB checkpoint moves straight from
Hugging Face to Tillicum and only metadata passes through the caller. Every file is checked
against the checkpoint manifest, and the manifest against the repository's latest-restart pointer.

The event log is cut back to the committed history: the source run logged update 8,001 after
its step-8,000 checkpoint and then stopped, so rows after step 8,000 are kept only in a separate
copy (events-source-through-<last>.jsonl), like the source run's own Modal-to-Tillicum handoff.

    python fetch_regret_continuation_restart.py --state /gpfs/scrubbed/$USER/anytok-runs/regret-rl-20k
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

REPO_ID = 'cminst/anytok-regret-rl-20k'
REVISION = '89bbf8d4cb9f5fb40399d4c4ea5ed51083499b51'  # main on 2026-11-20; holds step 8,000
STEP = 8000
PREFIX = f'checkpoints/step-{STEP:06d}'
POINTER_MANIFEST_SHA256 = '3829bc297ffdeab836c97de036e78a5ea81ead0e60aa2bf444a0005c9a02bda3'
METADATA = ['run_config.json', 'run_config.modal.json', 'initial_eval.json', 'heldout_tokenizers.json',
            'events.jsonl']


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def status(state, stage, **fields):
    atomic_json(state / 'fetch-status.json', dict(stage=stage, at_utc=datetime.now(timezone.utc).isoformat(), **fields))
    print(json.dumps(dict(stage=stage, **fields)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    args = parser.parse_args()
    state = args.state.resolve()
    run = state / 'run'
    download = state / 'download'
    if (run / f'step-{STEP:06d}' / 'manifest.json').exists():
        raise SystemExit('Run folder already holds step 8,000; refusing to overwrite it')
    state.mkdir(parents=True, exist_ok=True)
    status(state, 'downloading', repo_id=REPO_ID, revision=REVISION)
    from huggingface_hub import snapshot_download
    snapshot_download(REPO_ID, repo_type='model', revision=REVISION, local_dir=download,
                      allow_patterns=[PREFIX + '/*', 'metadata/*', 'latest_restart.json'], max_workers=8)

    status(state, 'verifying')
    pointer = json.loads((download / 'latest_restart.json').read_text())
    checkpoint = download / PREFIX
    if (pointer['step'], pointer['kind'], pointer['prefix']) != (STEP, 'full_restart_checkpoint', PREFIX):
        raise ValueError('Latest-restart pointer does not name the step-8,000 full restart')
    if pointer['manifest_sha256'] != POINTER_MANIFEST_SHA256 or sha256(checkpoint / 'manifest.json') != POINTER_MANIFEST_SHA256:
        raise ValueError('Step-8,000 manifest differs from the published pointer')
    manifest = json.loads((checkpoint / 'manifest.json').read_text())
    if manifest['step'] != STEP or manifest['kind'] != 'full_restart_checkpoint':
        raise ValueError('Manifest is not the step-8,000 full restart')
    present = {str(p.relative_to(checkpoint)) for p in checkpoint.rglob('*') if p.is_file()} - {'manifest.json'}
    if present != set(manifest['files']):
        raise ValueError('Checkpoint files differ from the manifest: ' + ', '.join(sorted(present ^ set(manifest['files']))))
    files = {}
    for name, expected in sorted(manifest['files'].items()):
        path = checkpoint / name
        if path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
            raise ValueError('Checksum mismatch: ' + name)
        files[name] = expected
    config = json.loads((download / 'metadata/run_config.json').read_text())
    if (config.get('reward_objective'), config['policy'].get('entropy_weight'), config.get('seed'),
            config['policy'].get('seed')) != ('regret', 0.0, 42, 43):
        raise ValueError('Saved run config is not the entropy-zero regret run')

    status(state, 'arranging')
    run.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (download / 'metadata/events.jsonl').read_text().splitlines() if line.strip()]
    kept = [row for row in rows if row.get('step', 0) <= STEP]
    last = max(row.get('step', 0) for row in rows)
    source_copy = run / f'events-source-through-{last}.jsonl'
    shutil.copy2(download / 'metadata/events.jsonl', source_copy)
    (run / 'events.jsonl').write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in kept))
    for name in METADATA[:-1]:
        shutil.copy2(download / 'metadata' / name, run / name)
    target = run / f'step-{STEP:06d}'
    os.replace(checkpoint, target)
    for path in [target, *target.rglob('*')]:
        mode = path.stat().st_mode
        path.chmod(mode & ~0o222)  # the trainer only reads the starting checkpoint
    receipt = dict(source_repo=REPO_ID, revision=REVISION, prefix=PREFIX, step=STEP, latest_restart=pointer,
                   files=files, total_bytes=sum(row['bytes'] for row in files.values()),
                   metadata={name: sha256(run / name) for name in METADATA[:-1]},
                   source_events=dict(path=source_copy.name, rows=len(rows), last_step=last,
                                      sha256=sha256(source_copy)),
                   events=dict(rows=len(kept), dropped_after_step=len(rows) - len(kept),
                               sha256=sha256(run / 'events.jsonl')),
                   at_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(state / 'restart-receipt.json', receipt)
    status(state, 'complete', step=STEP, bytes=receipt['total_bytes'], dropped_rows=len(rows) - len(kept))


if __name__ == '__main__':
    main()
