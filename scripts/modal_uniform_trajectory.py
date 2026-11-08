"""Profile reasonable pool tokenizers, then evaluate every saved uniform checkpoint."""
from datetime import datetime, timezone
from decimal import Decimal
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

import modal
from pool_generalization import digest, file_sha, load_plan, save_json

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'modal_results/uniform_trajectories'
NAME = 'anytok-uniform-trajectories'
app = modal.App(NAME)
training = modal.Volume.from_name('anytok-pool-20k', version=2)
outputs = modal.Volume.from_name(NAME, create_if_missing=True, version=2)
cpu_image = (modal.Image.debian_slim(python_version='3.12')
    .uv_pip_install('tokenizers==0.22.2', 'numpy==2.3.5')
    .env({'TOKENIZERS_PARALLELISM': 'false'})
    .add_local_python_source('uniform_trajectory', 'uniform_weakness', 'pool_generalization'))


def now(): return datetime.now(timezone.utc).isoformat()


def status(stage, completed=0, **details):
    value = {'at_utc': now(), 'stage': stage, 'completed': completed, 'total': 141,
             'profile': 'gildmember', **details}
    save_json(STATE / 'status.json', value)
    print(json.dumps(value), flush=True)


@app.function(image=cpu_image, volumes={'/train': training.with_mount_options(read_only=True), '/out': outputs},
              max_containers=100, single_use_containers=True, timeout=900)
def profile_pool(indices):
    from tokenizers import Tokenizer
    from uniform_trajectory import PROFILE_DOMAINS, profile_samples, tokenizer_utility
    canonical = Tokenizer.from_file('/train/input/model/tokenizer.json')
    samples = {}
    for domain in PROFILE_DOMAINS:
        with gzip.open(f'/train/input/eval/{domain}.jsonl.gz', 'rt', encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        samples[domain] = profile_samples(rows, domain)
    sample_digest = digest(samples)
    index = json.loads(Path('/train/input/pool/full-results.json').read_text())
    manifests = {r['config']['index']: r for r in index['tokenizers']}
    result = []
    for i in indices:
        manifest = manifests[i]
        path = Path(f'/train/input/pool/tokenizers/{i:04d}/tokenizer.json')
        if file_sha(path) != manifest['tokenizer_sha256']: raise ValueError('Pool tokenizer checksum mismatch')
        tokenizer = Tokenizer.from_file(str(path))
        if tokenizer.get_vocab_size() != manifest['vocab_size_actual']: raise ValueError('Wrong native vocabulary size')
        result.append({'index': i, 'sha256': manifest['tokenizer_sha256'],
                       'config': manifest['config'], 'sample_digest': sample_digest,
                       'utility': {d: tokenizer_utility(tokenizer, rows, canonical) for d, rows in samples.items()}})
        del tokenizer
    save_json(Path('/out/pool-profile') / f'{indices[0]:04d}.json', result)
    if indices[0] == 0: save_json('/out/profile-samples.json', samples)
    outputs.commit()
    return result


@app.function(image=cpu_image, volumes={'/out': outputs}, max_containers=100, timeout=1800)
def copy_aux(item):
    from grpclib.exceptions import StreamTerminatedError
    client = modal.Client.from_credentials(os.environ['SOURCE_MODAL_TOKEN_ID'], os.environ['SOURCE_MODAL_TOKEN_SECRET'])
    source = modal.Volume.from_name('anytok-pool-generalization', client=client)
    target = Path('/out/aux') / item['destination']
    if target.exists() and item.get('sha256') and file_sha(target) == item['sha256']:
        return {**item, 'reused': True, 'sha256': file_sha(target)}
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.' + uuid.uuid4().hex + '.partial')
    for attempt in range(4):
        try:
            with partial.open('wb') as stream:
                for block in source.read_file(item['source']): stream.write(block)
            actual = file_sha(partial)
            if partial.stat().st_size != item['bytes'] or (item.get('sha256') and actual != item['sha256']):
                raise ValueError('Auxiliary asset checksum mismatch')
            break
        except (OSError, StreamTerminatedError, modal.exception.ConnectionError):
            if attempt == 3: raise
            time.sleep(2 ** attempt)
    partial.replace(target); outputs.commit()
    return {**item, 'sha256': actual}


def stage_aux():
    from modal_tokenizer_pool_transfer import source_client
    client, secret = source_client('trenton')
    source = modal.Volume.from_name('anytok-pool-generalization', client=client)
    entries = {e.path: e for e in source.iterdir('weakness-v1', recursive=True)}
    previous = json.loads((ROOT / 'modal_results/uniform_weakness/report.json').read_text())
    items = []
    for candidate in previous['candidates']:
        path = f"weakness-v1/tokenizers/{candidate['id']}/tokenizer.json"
        items.append({'source': path, 'destination': f"tokenizers/{candidate['id']}/tokenizer.json",
                      'bytes': entries[path].size, 'sha256': candidate['sha256']})
    for name in ['panels.json', 'report.json', 'audit.json', 'plan.json']:
        path = 'weakness-v1/' + name
        items.append({'source': path, 'destination': name, 'bytes': entries[path].size})
    result = list(copy_aux.with_options(secrets=[secret]).map(items, order_outputs=False))
    save_json(STATE / 'aux-manifest.json', result)
    with outputs.batch_upload(force=True) as batch: batch.put_file(STATE / 'aux-manifest.json', 'aux-manifest.json')


@app.local_entrypoint()
def main(action: str = 'profile'):
    if os.environ.get('MODAL_PROFILE') != 'gildmember': raise ValueError('Use gildmember, which owns the complete checkpoint grid')
    from modal_credits import billing_row
    budget = billing_row('gildmember', 'this month', Decimal(30))
    if budget['status'] != 'ok' or float(budget['estimated_remaining_usd']) < 6:
        raise RuntimeError('Insufficient verified credit margin')
    STATE.mkdir(parents=True, exist_ok=True)
    if action == 'profile':
        status('profiling compression and lossless encoding for all 1000 pool tokenizers')
        with outputs.batch_upload(force=True) as batch:
            batch.put_file(ROOT / 'plans/uniform-trajectory-design.json', 'design.json')
            batch.put_file(ROOT / 'plans/uniform-trajectory-inventory.json', 'inventory.json')
        rows = []; batches = [list(range(i, i + 10)) for i in range(0, 1000, 10)]
        for result in profile_pool.map(batches, order_outputs=False):
            rows.extend(result)
            save_json(STATE / 'pool-profile.json', sorted(rows, key=lambda r: r['index']))
            status('profiling pool tokenizers', len(rows) // 10, tokenizers_profiled=len(rows), pool_size=1000)
        if sorted(r['index'] for r in rows) != list(range(1000)): raise ValueError('Missing pool profiles')
        if len({r['sample_digest'] for r in rows}) != 1: raise ValueError('Utility samples changed across workers')
        from uniform_trajectory import select_pool_candidates
        selection = select_pool_candidates(rows)
        save_json(STATE / 'pool-candidates.json', selection)
        with outputs.batch_upload(force=True) as batch:
            batch.put_file(STATE / 'pool-profile.json', 'pool-profile.json')
            batch.put_file(STATE / 'pool-candidates.json', 'pool-candidates.json')
        status('staging saved native tokenizer candidates', 100, pool_candidates=len(selection))
        stage_aux()
        status('pool profiling and candidate staging complete', 100, pool_candidates=len(selection))
    else: raise ValueError('Unknown action')
