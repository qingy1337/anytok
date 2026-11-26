"""Continue the entropy-zero regret RL run from its step-8,000 full restart on Tillicum (account stf).

The source run (Hugging Face cminst/anytok-regret-rl-20k) trained updates 1-5,288 on Modal and
5,289-8,000 on Tillicum with scripts/tillicum_regret_job.py from qingy1337/anytok (zett-modal-20k,
fdbd3ea), then paused. This driver continues it on another Tillicum account. The trainer, inputs and
every training setting are the source run's; operations change: paths, the Slurm account, the
Hugging Face destination (Dekiofsa/RLtokenizers, folder regret-rl-20k), a cap of two approved
training allocations, and queueing the next allocation before the final uploads so a walltime stop
cannot break the chain.

Layout on scrubbed (STATE): run/ is the run folder (fetched by fetch_regret_continuation_restart.py),
check-run/ the resume check's copy, backups/ the upload receipts, code/ these scripts.

  --check        resume a hard-linked copy of run/ to update 8,002 (debug QoS) with no uploads and
                 no resubmission; verify the handoff and that update 8,001 reproduces the source
                 run's logged tokenizer and batch.
  (default)      train run/ to update 20,000, backing up every new checkpoint to Hugging Face and
                 verifying it there. Requires a passed check.
  --backup-only  no training: publish and verify any checkpoints not yet on Hugging Face (a recovery
                 tool that also runs on a login node), and mark the run complete when it is.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

USER = os.environ['USER']
HERE = Path(__file__).resolve().parent
STATE = Path(f'/gpfs/scrubbed/{USER}/anytok-runs/regret-rl-20k')
RUN = STATE / 'run'
CHECK_RUN = STATE / 'check-run'
BACKUPS = STATE / 'backups'
REPO = Path(f'/gpfs/projects/stf/{USER}/anytok')
INPUT = Path(f'/gpfs/projects/stf/{USER}/anytok-data/input')
CODE = Path(f'/gpfs/projects/stf/{USER}/anytok-runs/.raw-bpb-code-d875965262f96b29')
TRAINER_SHA256 = 'c411e683c2251047ad497512aea1ca948ffb0bece398ce5fd21992782e18ce8b'
REPO_ID = 'Dekiofsa/RLtokenizers'
FOLDER = 'regret-rl-20k'
SOURCE = dict(repo_id='cminst/anytok-regret-rl-20k', revision='89bbf8d4cb9f5fb40399d4c4ea5ed51083499b51',
              prefix='checkpoints/step-008000',
              manifest_sha256='3829bc297ffdeab836c97de036e78a5ea81ead0e60aa2bf444a0005c9a02bda3')
START, STOP, CHECK_STOP = 8000, 20000, 8002
MAX_ALLOCATIONS = 2
METADATA = ['run_config.json', 'run_config.modal.json', 'initial_eval.json', 'heldout_tokenizers.json',
            'events.jsonl']

sys.path.insert(0, str(REPO / 'scripts'))
from run_raw_bpb_training import command, run_and_monitor, source_files  # noqa: E402

backup_lock = threading.Lock()
verified_locally = set()  # (folder name, manifest sha256) already checked against its manifest


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{threading.get_ident()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def redact(text):
    return re.sub(r'(?<=/)' + re.escape(USER) + r'(?![\w.-])', '<user>', text)


def note(stage, **fields):
    record = dict(stage=stage, at_utc=now(), job_id=os.environ.get('SLURM_JOB_ID'), **fields)
    atomic_json(STATE / 'driver-status.json', record)
    print(json.dumps(record), flush=True)


def full_restarts(run):
    return sorted(int(p.name.removeprefix('step-')) for p in run.glob('step-*')
                  if (p / 'state.pt').is_file() and (p / 'manifest.json').is_file())


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def verify_inputs():
    """The model and warmup at this account's paths must match the pinned input bundle."""
    rows = json.loads((INPUT / 'manifest.json').read_text())['files']
    checked = 0
    for name, row in sorted(rows.items()):
        if name.startswith(('model/', 'warmup/')):
            path = INPUT / name
            if path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
                raise ValueError('Input differs from its bundle manifest: ' + name)
            checked += 1
    if checked != 5:
        raise ValueError('Expected five model and warmup files in the bundle manifest')
    return sha256(INPUT / 'manifest.json')


def verify_code():
    """Check, without writing, that the staged tree is the source run's code and matches the checkout."""
    manifest = json.loads((CODE / 'source-manifest.json').read_text())
    sources = source_files()
    if set(manifest) != set(sources):
        raise ValueError('Staged code lists different files than the launcher')
    for name, source in sorted(sources.items()):
        if not (sha256(source) == sha256(CODE / name) == manifest[name]):
            raise ValueError('Staged code differs from the checkout: ' + name)
    revision = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if CODE.name != '.raw-bpb-code-' + revision[:16] or manifest['scripts/train_joint_smollm_input.py'] != TRAINER_SHA256:
        raise ValueError('Staged code is not the source run\'s trainer')
    return CODE


def trainer_env():
    env = {k: v for k, v in os.environ.items() if k not in {'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN'}}
    env.update(TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
               ANYTOK_RUN_PLATFORM='Tillicum · H200', ANYTOK_RUN_PROFILE='stf')
    return env


def trainer_command(code, output, stop, check):
    args = SimpleNamespace(inputs=INPUT, output=output, stop_at_update=stop, resume=True,
                           reward_objective='regret', memory_execution='optimized')
    cmd = command(args, code)
    wrapper = [sys.executable, '-u', str(HERE / 'resume_regret_continuation.py'), '--code', str(code)]
    return args, wrapper + (['--check'] if check else []) + ['--', *cmd[3:]]


def check(code, bundle):
    if CHECK_RUN.exists():
        raise SystemExit('check-run/ already exists; move it aside before checking again')
    source_row = next(row for row in read_rows(RUN / 'events-source-through-8001.jsonl')
                      if row.get('stage') == 'train' and row.get('step') == START + 1)
    CHECK_RUN.mkdir()
    for name in METADATA:
        shutil.copy2(RUN / name, CHECK_RUN / name)
    source = RUN / f'step-{START:06d}'
    target = CHECK_RUN / source.name
    target.mkdir()
    for path in sorted(source.rglob('*')):
        destination = target / path.relative_to(source)
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, destination)  # read-only inodes; the trainer only reads its restart
    before = len(read_rows(CHECK_RUN / 'events.jsonl'))
    args, cmd = trainer_command(code, CHECK_RUN, CHECK_STOP, check=True)
    note('checking', input_bundle_manifest_sha256=bundle)
    result = run_and_monitor(cmd, args, trainer_env())
    rows = read_rows(CHECK_RUN / 'events.jsonl')[before:]
    train = {row['step']: row for row in rows if row.get('stage') == 'train'}
    handoffs = [row for row in rows if row.get('stage') == 'execution_migration']
    first = train.get(START + 1, {})
    checks = dict(
        trainer_exit_zero=result == 0,
        handoff_recorded=(len(handoffs) == 1 and handoffs[0].get('step') == START
                          and set(handoffs[0].get('changed', [])) <= {'model', 'warmup_checkpoint', 'protected_steps'}),
        updates_8001_8002=sorted(train) == [START + 1, START + 2],
        policy_counter=[train[s].get('policy_updates') for s in sorted(train)] == [START + 1, START + 2],
        valid_examples=bool(train) and all(r.get('paired_valid_examples') == 128 for r in train.values()),
        maximum_positions=bool(train) and all(r.get('paired_maximum_positions', 999) <= 128 for r in train.values()),
        regret_reward=bool(train) and all(r.get('reward_objective') == 'regret'
                                          and r.get('policy_reward_bpb') == r.get('paired_gap_bpb') for r in train.values()),
        same_tokenizer_as_source=first.get('tokenizer_pool_index') == source_row['tokenizer_pool_index'],
        same_batch_as_source=first.get('paired_batch_sha256') == source_row['paired_batch_sha256'],
        restart_saved=(CHECK_RUN / f'step-{CHECK_STOP:06d}' / 'manifest.json').is_file(),
    )
    keys = ['tokenizer_pool_index', 'paired_batch_sha256', 'paired_gap_bpb', 'paired_canonical_bpb',
            'policy_reward_bpb', 'train_bpb', 'loss_nats_per_token', 'policy_action_probability', 'seconds', 'peak_gib']
    comparison = {key: dict(source=source_row.get(key), check=first.get(key)) for key in keys}
    passed = all(checks.values())
    atomic_json(STATE / 'check-receipt.json', dict(passed=passed, checks=checks, update_8001=comparison,
                                                   handoff=handoffs[0] if handoffs else None,
                                                   input_bundle_manifest_sha256=bundle,
                                                   job_id=os.environ.get('SLURM_JOB_ID'), at_utc=now()))
    note('check_passed' if passed else 'check_failed', checks=checks)
    if not passed:
        raise SystemExit(1)


def hf_api():
    from huggingface_hub import HfApi
    token = (Path.home() / '.cache/huggingface/token').read_text().strip()
    return HfApi(token=token)


def publish_source(api, bundle):
    """Record provenance once; this also proves upload access before any GPU time is spent."""
    marker = BACKUPS / 'source.json'
    if marker.exists():
        return
    from huggingface_hub import CommitOperationAdd
    fetched = json.loads((STATE / 'restart-receipt.json').read_text())
    record = dict(continuation_of=SOURCE['repo_id'], source_revision=SOURCE['revision'],
                  source_prefix=SOURCE['prefix'], source_manifest_sha256=SOURCE['manifest_sha256'],
                  source_files=fetched['files'], source_metadata_sha256=fetched['metadata'],
                  source_events=fetched['source_events'], events_at_handoff=fetched['events'],
                  start_step=START, stop_step=STOP, trainer_sha256=TRAINER_SHA256,
                  input_bundle_manifest_sha256=bundle,
                  check=json.loads((STATE / 'check-receipt.json').read_text()))
    readme = (
        '# Regret RL continuation, updates 8,000 to 20,000\n\n'
        'Continuation of the entropy-zero regret run in '
        f'[{SOURCE["repo_id"]}](https://huggingface.co/{SOURCE["repo_id"]}), resumed from its verified '
        'step-8,000 full restart on one Tillicum H200. The trainer (SHA-256 '
        f'`{TRAINER_SHA256}`), inputs and every training setting are unchanged; only input locations differ, '
        'and the handoff is recorded in `metadata/events.jsonl`. Updates up to 8,000 are in the source '
        'repository.\n\n'
        'Weights every 500 updates, full resumable state every 2,000 and at segment ends. Each '
        '`manifest.json` lists file sizes and SHA-256 hashes; `latest_restart.json` names the newest full '
        'restart and is published only after its payload is verified. `SOURCE.json` records the starting '
        'checkpoint and the resume check.\n')
    commit = api.create_commit(repo_id=REPO_ID, repo_type='model', operations=[
        CommitOperationAdd(path_in_repo=f'{FOLDER}/SOURCE.json',
                           path_or_fileobj=(json.dumps(redact_value(record), indent=2, sort_keys=True) + '\n').encode()),
        CommitOperationAdd(path_in_repo=f'{FOLDER}/README.md', path_or_fileobj=readme.encode())],
        commit_message='Start the regret RL continuation from step 8,000')
    atomic_json(marker, dict(revision=commit.oid, at_utc=now()))


def redact_value(value):
    return json.loads(redact(json.dumps(value)))


def metadata_bytes(name):
    text = (RUN / name).read_text(encoding='utf-8')
    if name == 'events.jsonl':
        text = text[:text.rfind('\n') + 1]  # never publish a line the trainer is still appending
    return redact(text).encode('utf-8')


def backup(api, folder):
    with backup_lock:
        step = int(folder.name.removeprefix('step-'))
        receipt_path = BACKUPS / (folder.name + '.json')
        if receipt_path.exists():
            return
        manifest_sha256 = sha256(folder / 'manifest.json')
        manifest = json.loads((folder / 'manifest.json').read_text())
        if manifest['step'] != step:
            raise ValueError('Checkpoint step mismatch')
        if (folder.name, manifest_sha256) not in verified_locally:
            for name, expected in manifest['files'].items():
                if (folder / name).stat().st_size != expected['bytes'] or sha256(folder / name) != expected['sha256']:
                    raise ValueError('Checkpoint changed before backup: ' + name)
            verified_locally.add((folder.name, manifest_sha256))
        prefix = f'{FOLDER}/checkpoints/{folder.name}'
        commit = api.upload_folder(repo_id=REPO_ID, repo_type='model', folder_path=folder, path_in_repo=prefix,
                                   commit_message=f'Regret continuation checkpoint {step}: {manifest["kind"]}')
        info = api.model_info(REPO_ID, revision=commit.oid, files_metadata=True)
        siblings = {row.rfilename: row for row in info.siblings}
        for name, expected in manifest['files'].items():
            stored = siblings[prefix + '/' + name]
            if stored.size != expected['bytes']:
                raise ValueError('Backup size mismatch: ' + name)
            if stored.lfs and stored.lfs.sha256 != expected['sha256']:
                raise ValueError('Backup hash mismatch: ' + name)
        receipt = dict(step=step, kind=manifest['kind'], revision=commit.oid, repo_id=REPO_ID, prefix=prefix,
                       manifest_sha256=manifest_sha256, at_utc=now())
        from huggingface_hub import CommitOperationAdd
        operations = [CommitOperationAdd(path_in_repo=f'{FOLDER}/metadata/{name}', path_or_fileobj=metadata_bytes(name))
                      for name in METADATA if (RUN / name).exists()]
        previous = [json.loads(p.read_text()) for p in BACKUPS.glob('step-*.json')]
        latest = max((r['step'] for r in previous if r['kind'] == 'full_restart_checkpoint'), default=0)
        if manifest['kind'] == 'full_restart_checkpoint' and step >= latest:
            operations.append(CommitOperationAdd(path_in_repo=f'{FOLDER}/latest_restart.json',
                                                 path_or_fileobj=(json.dumps(receipt, indent=2) + '\n').encode()))
        api.create_commit(repo_id=REPO_ID, repo_type='model', operations=operations,
                          commit_message=f'Verified metadata for regret continuation checkpoint {step}')
        atomic_json(receipt_path, receipt)
        atomic_json(STATE / 'backup-status.json', dict(stage='verified', **receipt))
        print(json.dumps(dict(stage='backup_verified', step=step, kind=manifest['kind'])), flush=True)


def pending_backups():
    return [p for p in sorted(RUN.glob('step-*')) if p.name != f'step-{START:06d}'
            and (p / 'manifest.json').is_file() and not (BACKUPS / (p.name + '.json')).exists()]


def backup_all(api):
    """Back up every pending checkpoint; one failing folder does not block the others."""
    errors = []
    for folder in pending_backups():
        try:
            backup(api, folder)
        except Exception as error:  # noqa: BLE001 - recorded and re-raised together
            errors.append(f'{folder.name}: {redact(str(error))[:300]}')
    if errors:
        raise RuntimeError('; '.join(errors))


def finish_uploads(api, tries=6):
    for attempt in range(tries):
        try:
            backup_all(api)
            return True
        except Exception as error:  # noqa: BLE001
            atomic_json(STATE / 'backup-status.json', dict(stage='retrying', error=redact(str(error))[:500],
                                                           attempt=attempt + 1, at_utc=now()))
            if attempt + 1 < tries:
                time.sleep(min(600, 30 * 2 ** attempt))
    return not pending_backups()


def mark_complete_if_done(api):
    step = full_restarts(RUN)[-1]
    if step < STOP:
        return False
    if finish_uploads(api):
        atomic_json(STATE / 'complete.json', dict(step=step, repo_id=REPO_ID, folder=FOLDER, at_utc=now()))
        note('complete', step=step)
        return True
    note('needs_upload', step=step, pending=[p.name for p in pending_backups()])
    return False


def set_aside(job):
    """Move checkpoints newer than the newest full restart (and unfinished writes) out of the trainer's way."""
    newest = full_restarts(RUN)[-1]
    moved = []
    for path in sorted(RUN.iterdir()):
        stale_step = path.name.startswith('step-') and int(path.name.removeprefix('step-')) > newest
        if stale_step or path.name.startswith('.writing-'):
            target = RUN / f'interrupted-{job}'
            target.mkdir(exist_ok=True)
            os.rename(path, target / path.name)
            receipt = BACKUPS / (path.name + '.json')
            if receipt.exists():
                (BACKUPS / f'interrupted-{job}').mkdir(parents=True, exist_ok=True)
                os.rename(receipt, BACKUPS / f'interrupted-{job}' / receipt.name)
            moved.append(path.name)
    if moved:
        note('set_aside', newest_full_restart=newest, moved=moved)
    return newest


def train(code, bundle):
    job = os.environ['SLURM_JOB_ID']
    receipt = STATE / 'check-receipt.json'
    if not receipt.exists() or not json.loads(receipt.read_text())['passed']:
        raise SystemExit('The resume check has not passed; refusing to train')
    path = STATE / 'allocations.json'
    allocations = json.loads(path.read_text()) if path.exists() else []
    api = hf_api()
    publish_source(api, bundle)
    if mark_complete_if_done(api):
        return
    if job not in allocations and len(allocations) >= MAX_ALLOCATIONS:
        note('needs_approval', reason='Both approved training allocations are used')
        raise SystemExit('Both approved training allocations are used; ask before continuing')
    start = set_aside(job)
    args, cmd = trainer_command(code, RUN, STOP, check=False)
    stop = threading.Event()

    def uploads():
        try:
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 10)  # per-thread nice on Linux
        except OSError:
            pass
        failures = 0
        while not stop.is_set():
            try:
                backup_all(api)
                failures = 0
            except Exception as error:  # noqa: BLE001
                failures += 1
                atomic_json(STATE / 'backup-status.json', dict(stage='retrying', error=redact(str(error))[:500],
                                                               failures=failures, at_utc=now()))
            stop.wait(20 if failures == 0 else min(1800, 20 * 2 ** failures))

    def wrap_up_uploads():
        stop.set()
        worker.join(timeout=600)
        if not finish_uploads(api, tries=3):
            note('needs_upload', pending=[p.name for p in pending_backups()])

    if job not in allocations:
        allocations.append(job)
        atomic_json(path, allocations)
    worker = threading.Thread(target=uploads, daemon=True)
    worker.start()
    note('training', start_step=start, allocations=allocations)
    result = run_and_monitor(cmd, args, trainer_env())
    step = full_restarts(RUN)[-1]
    if result or step <= start:
        reason = f'Trainer exited {result}' if result else 'No committed training progress'
        note('failed', returncode=result, step=step, reason=reason)
        wrap_up_uploads()
        raise RuntimeError(reason + '; keep every checkpoint file and investigate')
    if step < STOP:
        if len(allocations) < MAX_ALLOCATIONS:
            try:
                submitted = subprocess.run(['sbatch', '--parsable', f'--dependency=afterany:{job}',
                                            str(HERE / 'continue_regret_rl.sbatch')],
                                           text=True, capture_output=True, check=True, timeout=300)
                next_job = submitted.stdout.strip().split(';')[0]
                atomic_json(STATE / 'current-job.json', dict(job_id=next_job, resume_step=step, queued_by=job,
                                                              reason='Graceful walltime handoff', at_utc=now()))
                note('handed_off', step=step, next_job=next_job)
            except (subprocess.SubprocessError, OSError) as error:
                note('handoff_failed', step=step, error=redact(str(getattr(error, 'stderr', '') or error))[:500])
        else:
            note('needs_approval', step=step, reason='Both approved training allocations are used')
    wrap_up_uploads()  # if the walltime cuts this short, the next allocation finishes the uploads
    mark_complete_if_done(api)


def backup_only():
    if not (STATE / 'check-receipt.json').exists() or not json.loads((STATE / 'check-receipt.json').read_text())['passed']:
        raise SystemExit('The resume check has not passed')
    api = hf_api()
    publish_source(api, verify_inputs())
    if not mark_complete_if_done(api) and not finish_uploads(api):
        note('needs_upload', pending=[p.name for p in pending_backups()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--backup-only', action='store_true')
    args = parser.parse_args()
    if not RUN.is_dir() or not (RUN / f'step-{START:06d}' / 'manifest.json').is_file():
        raise SystemExit('Fetch the step-8,000 restart first')
    lock = (STATE / 'job.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (STATE / 'pause-request.json').exists():
        note('paused', reason='pause-request.json present; explicit resume required')
        return
    if args.backup_only:
        backup_only()
        return
    bundle = verify_inputs()
    code = verify_code()
    if args.check:
        check(code, bundle)
    else:
        train(code, bundle)


if __name__ == '__main__':
    main()
