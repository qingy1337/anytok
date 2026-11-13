"""Bounded RL-only rerun: frozen random50 plus original worst on original broad text."""
from datetime import datetime, timezone
from decimal import Decimal
import gzip
import json
import os
from pathlib import Path

import modal
from modal_random_pool_50 import cpu_image, gpu_image, evaluate, volume
from pool_generalization import digest, file_sha, save_json

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'modal_results/corrected_pool_check'
PREFIX = 'corrected-pool-check-v1'
PROFILE = 'blackpantherx86'
app = modal.App('anytok-corrected-pool-check')
score = app.function(image=gpu_image, gpu='B300', volumes={'/sweep': volume},
    max_containers=1, timeout=600, scaledown_window=2)(evaluate.get_raw_f())


def status(stage, completed=0, **fields):
    row = dict(at_utc=datetime.now(timezone.utc).isoformat(), stage=stage, completed=completed,
               total=50, profile=PROFILE, **fields)
    save_json(STATE / 'status.json', row)
    with (STATE / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


@app.function(image=cpu_image.add_local_python_source('original_pool_check'),
    volumes={'/sweep': volume}, timeout=300, scaledown_window=2)
def prepare(original_plan):
    from original_pool_check import correct_panel
    from uniform_trajectory import CanonicalTokenizer, load_candidate
    volume.reload();root = Path('/sweep');folder = root / PREFIX
    if (folder / 'preparation.json').exists():
        old = json.loads((folder / 'preparation.json').read_text())
        if old['source_plan_sha256'] != digest(original_plan):
            raise ValueError('Cannot alter the frozen rerun')
        return old
    spec = json.loads((root / 'spec.json').read_text())
    panel = json.loads((root / 'panel.json').read_text())
    if digest(spec) != original_plan['source_spec_sha256'] or digest(panel) != original_plan['parent_panel_sha256']:
        raise ValueError('Original evaluation inputs changed')
    canonical_path = root / 'input/model/tokenizer.json'
    if file_sha(canonical_path) != spec['input_hashes']['model/tokenizer.json']:
        raise ValueError('Canonical tokenizer changed')
    sources = {}
    for domain in spec['domains']:
        path = root / f'input/eval/{domain}.jsonl.gz'
        if file_sha(path) != spec['input_hashes'][f'eval/{domain}.jsonl.gz']:
            raise ValueError('Original held-out source changed')
        with gzip.open(path, 'rt') as stream:
            sources[domain] = {r['id']: r['text'] for r in map(json.loads, stream)}
    tokenizers, forbidden = {}, {}
    for index in original_plan['indices']:
        path = root / f'input/pool/tokenizers/{index:04d}/tokenizer.json'
        if file_sha(path) != spec['tokenizers'][index]['sha256']:
            raise ValueError('Native tokenizer changed')
        native = load_candidate(path, 'pool');tokenizers[index] = native[0];forbidden[index] = native[4] | {0}
    rows = correct_panel(panel, sources, CanonicalTokenizer(canonical_path), tokenizers, forbidden)
    plan = {**original_plan, 'panel_sha256': digest(rows)}
    result = dict(plan=plan, source_plan_sha256=digest(original_plan), panel_sha256=digest(rows),
        metadata=[{k: v for k, v in r.items() if k != 'prefix'} for r in rows])
    save_json(folder / 'panel.json', rows);save_json(folder / 'plan.json', plan)
    save_json(folder / 'preparation.json', result);volume.commit()
    return result


@app.local_entrypoint()
def main():
    if os.environ.get('MODAL_PROFILE') != PROFILE:
        raise ValueError('Use ' + PROFILE)
    STATE.mkdir(parents=True, exist_ok=True)
    from modal_credits import billing_row
    try:
        credit = billing_row(PROFILE, 'this month', Decimal(30));save_json(STATE / 'credits-before.json', credit)
        if credit['status'] != 'ok' or Decimal(credit['estimated_remaining_usd']) < Decimal('2.2'):
            raise ValueError('Need $2.20 remaining for one bounded 600-second GPU call plus margin')
        status('correcting original broad-text cuts')
        original = json.loads((ROOT / 'plans/corrected-pool-check-v1.json').read_text())
        prepared = prepare.remote(original);save_json(STATE / 'preparation.json', prepared)
        plan = prepared['plan'];save_json(STATE / 'plan.json', plan)
        status('scoring corrected original broad text')
        call = score.spawn('rl20k', plan, run_prefix=PREFIX)
        save_json(STATE / 'call.json', dict(app_id=app.app_id, call_id=call.object_id))
        while True:
            try:
                result = call.get(timeout=10)
                with gzip.open(STATE / 'scores.json.gz', 'wt') as stream:
                    json.dump(result, stream)
                break
            except (TimeoutError, modal.exception.TimeoutError):
                try:
                    worker = json.loads(b''.join(volume.read_file(PREFIX + '/status/rl20k.json')))
                except FileNotFoundError:
                    worker = dict(completed=0, stage='loading model')
                status('scoring corrected original broad text', worker['completed'], worker=worker)
        status('complete', len(plan['indices']))
        save_json(STATE / 'credits-after.json', billing_row(PROFILE, 'this month', Decimal(30)))
    except Exception as exc:
        status('failed', error=str(exc));raise
