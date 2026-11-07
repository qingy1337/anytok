"""Run uniform-pool training to 20k, handing off through Hugging Face.

    python scripts/run_anytok_pool_20k.py

Uses B300 only, with no CPU/memory reservation, and estimated credit guards.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal
import fcntl
import json
import os
from pathlib import Path
import subprocess
import statistics
import time

import modal
from huggingface_hub import HfApi, get_token
from modal_credits import billing_row
from anytok_pool_config import PROFILES, VOLUME_NAME
from modal_tokenizer_pool_transfer import source_client

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'modal_results/anytok_pool_20k'
REPO = 'qingy2024/anytok-pool-smollm2-20k'
TRAINER = 'scripts/modal_anytok_pool_20k.py'
HANDOFF = 'scripts/modal_anytok_pool_20k_hf.py'


def emit(stage, **fields):
    old = json.loads((STATE / 'status.json').read_text()) if (STATE / 'status.json').exists() else {}
    value = {**old, 'at_utc': datetime.now(timezone.utc).isoformat(), 'stage': stage, **fields}
    temporary = STATE / 'status.tmp'
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(STATE / 'status.json')
    print(json.dumps(value), flush=True)


def call(profile, script, arguments, label):
    log = STATE / f'{profile}-{label}.log'
    with log.open('a') as output:
        process = subprocess.Popen(['modal', 'run', '--detach', script, *arguments],
            cwd=ROOT, env={**os.environ, 'MODAL_PROFILE': profile}, stdout=output, stderr=subprocess.STDOUT)
        last = 0
        while process.poll() is None:
            if label == 'train' and time.monotonic() - last > 15:
                last = time.monotonic()
                lines = log.read_text(errors='replace').splitlines()
                records = []
                for line in lines[-300:]:
                    try:
                        item = json.loads(line[line.index('{'):])
                    except (ValueError, TypeError):
                        continue
                    if item.get('stage') == 'train':
                        records.append(item)
                if records:
                    record = records[-1]
                    timing = statistics.median(r['seconds'] for r in records[-100:])
                    emit('training', profile=profile, step=record['step'],
                         seconds_per_step=timing, eta_seconds=((20000-record['step'])*timing if len(records)>=20 else None),
                         elapsed_seconds=time.time()-START,
                         workers=[{'id':'B300', 'profile':profile, 'platform':'Modal · B300',
                                   'task':f"Step {record['step']:,} / 20,000", 'state':'Training',
                                   'elapsed_seconds':time.time()-START}])
            time.sleep(5)
        if process.returncode:
            raise RuntimeError(f'{profile} {label} failed; inspect {log}')


def latest(profile):
    client, _ = source_client(profile)
    volume = modal.Volume.from_name(VOLUME_NAME, version=2, client=client)
    entries = list(volume.iterdir('run', recursive=True))
    return max([int(Path(e.path).parent.name.removeprefix('step-')) for e in entries
                if e.path.endswith('/state.pt') and Path(e.path).parent.name.startswith('step-')] or [0])


START = time.time()


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    lock = (STATE / 'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (STATE / 'status.json').exists():
        raise RuntimeError('Existing run state requires checkpoint reconciliation before another controller starts')
    HfApi(token=get_token()).create_repo(REPO, repo_type='model', private=True, exist_ok=True)
    emit('preparing', step=0, final_step=20000, profiles=list(PROFILES), repo=REPO)
    previous = None
    previous_step = 0
    archives = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for profile in PROFILES:
            budget = billing_row(profile, 'this month', Decimal(30))
            if budget['status'] != 'ok' or float(budget['estimated_remaining_usd']) < 10:
                continue
            try:
                client, _ = source_client(profile)
                with modal.App('anytok-pool-training-capacity').run(client=client):
                    pass
            except Exception as exc:
                emit('profile_unavailable', profile=profile, error=str(exc))
                continue
            emit('preparing_inputs', profile=profile)
            call(profile, 'scripts/modal_anytok_pool_inputs.py', [], 'inputs')
            if previous:
                call(profile, HANDOFF, ['--action','download'], 'download')
                call(previous, TRAINER, ['--action','compact-source'], 'compact')
                archives.append(pool.submit(call, previous, HANDOFF, ['--action','archive'], 'archive'))
            budget = billing_row(profile, 'this month', Decimal(30))
            if budget['status'] != 'ok':
                raise RuntimeError('Billing unavailable before GPU launch')
            available = float(budget['estimated_remaining_usd']) - 1.5
            seconds = min(12600, int(available / 7.5 * 3600))
            if seconds < 900:
                raise RuntimeError('Insufficient credit margin after input preparation')
            emit('starting_training', profile=profile, max_seconds=seconds)
            call(profile, TRAINER, ['--action','train','--max-seconds',str(seconds)], 'train')
            step = latest(profile)
            if step <= previous_step:
                raise RuntimeError('Training segment made no checkpoint progress')
            emit('publishing_checkpoint', profile=profile, step=step, workers=[])
            call(profile, HANDOFF, ['--action','upload'], 'upload')
            if step >= 20000:
                call(profile, TRAINER, ['--action','compact-source'], 'compact')
                archives.append(pool.submit(call, profile, HANDOFF, ['--action','archive'], 'archive'))
                for future in archives:
                    future.result()
                emit('complete', profile=profile, step=20000, workers=[], eta_seconds=0,
                     elapsed_seconds=time.time()-START)
                return
            previous = profile
            previous_step = step
        raise RuntimeError('Verified profile credits exhausted before 20k; saved checkpoint remains resumable')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if STATE.exists():
            emit('failed', error=str(exc), workers=[])
        raise
