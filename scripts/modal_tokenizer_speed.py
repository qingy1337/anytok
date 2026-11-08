"""Benchmark baseline/accelerated generation on nine full-range tokenizer cells.

Default command is a read-only plan. A small remote smoke test precedes the
matched 200 MB benchmark. All compute is CPU-only with no CPU/memory reservation.
Source: gildmember/anytok-pool-20k/input (read only). Outputs, logs, tokenizer
files and wheel provenance: gildmember/anytok-fast-tokenizers. Only metadata
returns to the local caller. Call IDs permit reattachment; never restart a live
coordinator after an observation timeout.

    MODAL_PROFILE=gildmember modal run scripts/modal_tokenizer_speed.py
    MODAL_PROFILE=gildmember modal run scripts/modal_tokenizer_speed.py --action smoke
    MODAL_PROFILE=gildmember modal run --detach scripts/modal_tokenizer_speed.py --action benchmark
"""
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import modal
from pool_generalization import save_json, file_sha
from tokenizer_pool import digest

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = os.environ.get('ANYTOK_SPEED_CANDIDATE', 'v4')
if CANDIDATE not in ['v4', 'v5', 'v6']: raise ValueError('Unknown tokenizer speed candidate')
STATE = ROOT / ('modal_results/tokenizer_speed' + ('_'+CANDIDATE if CANDIDATE != 'v4' else ''))
PLAN = ROOT / ('plans/tokenizer-speed-v1-implementation-only.json' if CANDIDATE != 'v4' else 'plans/tokenizer-speed-v1.json')
BASE = ROOT / 'build/pool-v2/tokenizers-0.22.2-cp39-abi3-manylinux_2_34_x86_64.whl'
FAST = ROOT / f'build/fast-{CANDIDATE}/tokenizers-0.22.2-cp39-abi3-manylinux_2_34_x86_64.whl'
app = modal.App('anytok-tokenizer-speed' + ('-'+CANDIDATE if CANDIDATE != 'v4' else ''))
source = modal.Volume.from_name('anytok-pool-20k', version=2)
outputs = modal.Volume.from_name('anytok-fast-tokenizers', create_if_missing=True, version=2)
image = (modal.Image.debian_slim(python_version='3.12')
         .add_local_file(BASE, '/tmp/base.whl', copy=True)
         .add_local_file(FAST, '/tmp/fast.whl', copy=True)
         .run_commands("python -c \"import zipfile; zipfile.ZipFile('/tmp/base.whl').extractall('/opt/base'); zipfile.ZipFile('/tmp/fast.whl').extractall('/opt/fast')\"")
         .env({'RAYON_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'true'})
         .add_local_python_source('tokenizer_pool', 'tokenizer_speed', 'pool_generalization'))


def now(): return datetime.now(timezone.utc).isoformat()


def status(stage, completed=0, total=18, **kwargs):
    row = {'at_utc': now(), 'stage': stage, 'completed': completed, 'total': total,
           'profile': 'gildmember', **kwargs}
    save_json(STATE / 'status.json', row)
    with (STATE / 'events.jsonl').open('a') as stream: stream.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


@app.function(image=image, volumes={'/train': source.with_mount_options(read_only=True), '/out': outputs},
              max_containers=9, timeout=21600, single_use_containers=True)
def benchmark_case(case, provenance, smoke=False):
    import tokenizer_speed
    config = dict(case['config']); recipe = json.loads(Path('/train/input/recipe.json').read_text())
    for bucket in recipe['buckets']: bucket['source_path'] = '/train/input/corpus/' + bucket['name']
    if smoke:
        from tokenizer_pool import allocate
        config['vocab_size'] = 1024
        config['bucket_bytes'] = dict(zip(config['bucket_bytes'], allocate(1_000_000, list(config['bucket_bytes'].values()))))
        config['domain_bytes'] = dict(zip(config['domain_bytes'], allocate(1_000_000, list(config['domain_bytes'].values()))))
    job = provenance['id'] + ('-smoke' if smoke else '')
    folder = Path('/out') / job / case['key']; folder.mkdir(parents=True, exist_ok=True)
    config_path = folder / 'config.json'; recipe_path = folder / 'recipe.json'
    save_json(config_path, config); save_json(recipe_path, recipe)
    rows = []
    # Alternate order across cells; both variants run sequentially on the same
    # worker/container with the same process-level Rayon setting and corpus.
    order = ['baseline-200mb', 'fast-200mb']
    if case['config']['index'] % 2: order.reverse()
    for variant in order:
        dest = folder / variant; marker = dest / 'complete.json'
        if marker.exists():
            saved = json.loads(marker.read_text())
            if saved['provenance'] != provenance or file_sha(dest/'tokenizer.json') != saved['manifest']['tokenizer_sha256']:
                raise ValueError('Cached benchmark provenance mismatch')
            rows.append(saved); continue
        dest.mkdir(parents=True, exist_ok=True)
        event = {'at_utc': now(), 'stage': 'training', 'case': case['key'], 'variant': variant}
        save_json(folder / 'status.json', event); outputs.commit(); print(json.dumps(event), flush=True)
        runtime = '/opt/base' if variant.startswith('baseline') else '/opt/fast'
        command = [sys.executable, tokenizer_speed.__file__, '--config', str(config_path),
                   '--recipe', str(recipe_path), '--destination', str(dest), '--heldout-root', '/train/input/eval']
        env = {**os.environ, 'PYTHONPATH': runtime, 'PYTHONUNBUFFERED': '1'}
        started = time.perf_counter()
        with (dest/'process.log').open('w') as stream:
            result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            tail = (dest/'process.log').read_text(errors='replace')[-4000:]
            save_json(dest/'failure.json', {'returncode': result.returncode, 'tail': tail, 'at_utc': now()}); outputs.commit()
            raise RuntimeError(f"{case['key']}/{variant} failed: {tail}")
        row = json.loads((dest/'benchmark.json').read_text())
        row.update({'case': case['key'], 'variant': variant, 'provenance': provenance,
                    'subprocess_seconds': time.perf_counter()-started, 'cpu_model': Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0]})
        save_json(marker, row); rows.append(row)
        save_json(folder/'status.json', {'at_utc': now(), 'stage': 'variant complete', 'case': case['key'], 'variant': variant})
        outputs.commit(); print(json.dumps({'stage': 'variant complete', 'case': case['key'], 'variant': variant,
                                           'ready_seconds': row['ready_seconds']}), flush=True)
    a, b = rows
    if (a['manifest']['source_stats'] != b['manifest']['source_stats']
            or a['manifest']['config'] != b['manifest']['config']
            or a['lossless_audit']['fixture_sha256'] != b['lossless_audit']['fixture_sha256']):
        raise ValueError('Matched benchmark used different text, settings or audit fixtures')
    return rows


@app.function(image=image, volumes={'/out': outputs}, timeout=21600)
def coordinate(plan, provenance, smoke=False):
    rows = []
    for result in benchmark_case.starmap([(case, provenance, smoke) for case in plan['cases']], order_outputs=False):
        rows.extend(result)
    return rows


@app.local_entrypoint()
def main(action: str = 'plan', resume_from: str = ''):
    plan = json.loads(PLAN.read_text()); STATE.mkdir(parents=True, exist_ok=True)
    provenance = {'plan_sha256': file_sha(PLAN), 'base_wheel_sha256': file_sha(BASE),
                  'fast_wheel_sha256': file_sha(FAST), 'trainer_sha256': file_sha(ROOT/'scripts/tokenizer_pool.py'),
                  'worker_sha256': file_sha(ROOT/'scripts/tokenizer_speed.py'), 'launcher_sha256': file_sha(__file__)}
    provenance['id'] = digest(provenance)[:16]
    if action == 'plan':
        print(json.dumps({'plan': plan, 'provenance': provenance}, indent=2)); return
    if action not in ['smoke', 'benchmark']: raise ValueError('Expected plan, smoke or benchmark')
    if os.environ.get('MODAL_PROFILE') != 'gildmember': raise ValueError('Explicit gildmember profile required')
    smoke = action == 'smoke'; pointer = STATE / f'{provenance["id"]}-{action}-call.json'
    if resume_from:
        pointer = Path(resume_from)
        saved = json.loads(pointer.read_text())
        if (saved['action'] != action or saved['provenance']['base_wheel_sha256'] != file_sha(BASE)
                or saved['provenance']['fast_wheel_sha256'] != file_sha(FAST)):
            raise ValueError('Resume pointer does not match requested wheels/action')
        provenance = saved['provenance']
    if pointer.exists(): call = modal.FunctionCall.from_id(json.loads(pointer.read_text())['call_id'])
    else:
        from modal_credits import billing_row
        balance = billing_row('gildmember', 'this month', Decimal(30))
        if balance['status'] != 'ok' or float(balance['estimated_remaining_usd']) < 6:
            raise RuntimeError('Verified $6 credit margin required for a new launch')
        with outputs.batch_upload(force=True) as batch:
            batch.put_file(PLAN, provenance['id']+'/plan.json')
            batch.put_file(BASE, provenance['id']+'/artifacts/base.whl')
            batch.put_file(FAST, provenance['id']+'/artifacts/fast.whl')
        call = coordinate.spawn(plan, provenance, smoke)
        save_json(pointer, {'call_id': call.object_id, 'provenance': provenance, 'action': action})
    job = provenance['id'] + ('-smoke' if smoke else '')
    while True:
        try:
            rows = call.get(timeout=10); break
        except (TimeoutError, modal.exception.TimeoutError): pass
        try: entries = list(outputs.iterdir(job, recursive=True))
        except (FileNotFoundError, modal.exception.NotFoundError): entries = []
        finished = [e for e in entries if e.path.endswith('/complete.json')]
        status(f'{action}: paired tokenizer generation', len(finished), run_id=job, call_id=call.object_id)
    save_json(STATE/f'{job}-results.json', rows)
    status(f'{action} complete', len(rows), run_id=job, call_id=call.object_id)
    print(json.dumps({'run_id': job, 'results': len(rows)}), flush=True)
