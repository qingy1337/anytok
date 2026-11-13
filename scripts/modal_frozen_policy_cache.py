"""Full corrected 1000-tokenizer reward table; only policy learning runs locally."""
from datetime import datetime, timezone
from decimal import Decimal
import gzip
import json
import os
from pathlib import Path
import time
import uuid

import modal
from modal_random_pool_50 import evaluate
from pool_generalization import digest, file_sha, save_json

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'modal_results/frozen_policy_cache'
PREFIX = 'frozen-policy-search-v1'
PROFILE = os.environ.get('FROZEN_POLICY_PROFILE', os.environ.get('MODAL_PROFILE', ''))
if PROFILE in ['general', 'research-everettcc']:
    raise ValueError('Disabled profile')
VOLUME = 'anytok-rl-all-pool-20k' if PROFILE == 'blackpantherx86' else 'anytok-frozen-policy-search'
volume = modal.Volume.from_name(VOLUME, create_if_missing=True, version=2)
app = modal.App('anytok-frozen-policy-cache')
BASE = ROOT / 'runtime/anytok_pool_20k'
cpu_image = (modal.Image.debian_slim(python_version='3.12')
    .uv_pip_install('tokenizers==0.22.2', 'numpy==2.3.5')
    .env({'TOKENIZERS_PARALLELISM': 'false', 'FROZEN_POLICY_PROFILE': PROFILE})
    .add_local_python_source('modal_random_pool_50', 'original_pool_check', 'pool_generalization',
        'pool_regime', 'uniform_trajectory', 'uniform_weakness'))
gpu_image = (modal.Image.from_registry('ubuntu:24.04', add_python='3.12')
    .apt_install('libpython3.12')
    .uv_pip_install('torch==2.15.0', 'transformers==5.15.1', 'tokenizers==0.22.2',
        'numpy==2.3.5', 'safetensors==0.8.0', 'accelerate==1.14.0')
    .env({'TOKENIZERS_PARALLELISM': 'false', 'FROZEN_POLICY_PROFILE': PROFILE})
    .add_local_dir(BASE / 'hf_hypernet', '/opt/zett/hf_hypernet')
    .add_local_file(BASE / 'rust_utils/target/release/librust_utils.so',
        '/opt/zett/rust_utils/target/release/librust_utils.so')
    .add_local_python_source('modal_random_pool_50', 'natural_transfer', 'pool_generalization',
        'uniform_trajectory', 'uniform_weakness'))
score = app.function(image=gpu_image, gpu='B300', volumes={'/sweep': volume},
    max_containers=2, timeout=2400, scaledown_window=2)(evaluate.get_raw_f())


def status(stage, completed=0, **details):
    row = dict(at_utc=datetime.now(timezone.utc).isoformat(), stage=stage, completed=completed,
        total=1000, profile=PROFILE, **details)
    save_json(STATE / 'status.json', row)
    with (STATE / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


@app.function(image=cpu_image, volumes={'/sweep': volume}, timeout=300, scaledown_window=2)
def available():
    return dict(profile=PROFILE)


@app.function(image=cpu_image, volumes={'/sweep': volume}, timeout=300, scaledown_window=2)
def initial(plan):
    from original_pool_check import correct_panel
    from uniform_trajectory import CanonicalTokenizer
    volume.reload();root = Path('/sweep');spec = json.loads((root / 'spec.json').read_text())
    panel = json.loads((root / 'panel.json').read_text())
    if digest(spec) != plan['source_spec_sha256'] or digest(panel) != plan['parent_panel_sha256']:
        raise ValueError('Original full-pool inputs changed')
    sources = {}
    for domain in spec['domains']:
        path = root / f'input/eval/{domain}.jsonl.gz'
        if file_sha(path) != spec['input_hashes'][f'eval/{domain}.jsonl.gz']:
            raise ValueError('Original corpus changed')
        with gzip.open(path, 'rt') as stream:
            sources[domain] = {r['id']: r['text'] for r in map(json.loads, stream)}
    canonical = CanonicalTokenizer(root / 'input/model/tokenizer.json')
    if file_sha(root / 'input/model/tokenizer.json') != spec['input_hashes']['model/tokenizer.json']:
        raise ValueError('Canonical tokenizer changed')
    rows = correct_panel(panel, sources, canonical, {-1: canonical._tokenizer}, {-1: {0}})
    save_json(root / PREFIX / 'round-0.json', rows);volume.commit()
    return dict(path=PREFIX + '/round-0.json', passages=len(rows))


@app.function(image=cpu_image, volumes={'/sweep': volume}, max_containers=50, timeout=600, scaledown_window=2)
def constraints(indices, panel_path, collect):
    from pool_regime import required_start
    from uniform_trajectory import CanonicalTokenizer, load_candidate
    volume.reload();root = Path('/sweep');rows = json.loads((root / panel_path).read_text())
    spec = json.loads((root / 'spec.json').read_text());canonical = CanonicalTokenizer(root / 'input/model/tokenizer.json')
    starts = [0] * len(rows);lengths = {};files = []
    for index in indices:
        path = root / f'input/pool/tokenizers/{index:04d}/tokenizer.json'
        if file_sha(path) != spec['tokenizers'][index]['sha256']:
            raise ValueError('Native tokenizer changed')
        native = load_candidate(path, 'pool');tokenizer = native[0];forbidden = native[4] | {0}
        counts = []
        for j, row in enumerate(rows):
            start, count = required_start(row, canonical, tokenizer, forbidden)
            starts[j] = max(starts[j], start);counts.append(count)
        lengths[str(index)] = counts
        if collect:
            folder = root / f'features/{index:04d}';manifest = json.loads((folder / 'manifest.json').read_text())
            if manifest['tokenizer_sha256'] != spec['tokenizers'][index]['sha256']:
                raise ValueError('Feature tokenizer mapping changed')
            for target in [path, folder / 'manifest.json', folder / 'forms.npy', folder / 'priors.npy']:
                files.append(dict(path=str(target.relative_to(root)), bytes=target.stat().st_size, sha256=file_sha(target)))
    return dict(starts=starts, lengths=lengths, files=files, validated=len(indices))


@app.function(image=cpu_image, volumes={'/sweep': volume}, timeout=300, scaledown_window=2)
def reduce_panel(path, starts, lengths, final, plan, round_index):
    from pool_regime import shorten
    from uniform_trajectory import CanonicalTokenizer
    volume.reload();root = Path('/sweep');rows = json.loads((root / path).read_text())
    canonical = CanonicalTokenizer(root / 'input/model/tokenizer.json')
    rows = shorten(rows, starts, canonical)
    name = PREFIX + ('/panel.json' if final else f'/round-{round_index}.json')
    if final:
        for j, row in enumerate(rows):
            row['native_prefix_lengths'] = {str(i): lengths[str(i)][j] for i in range(1000)}
            row['maximum_positions'] = 33 + max(len(row['canonical_prefix_ids']), max(row['native_prefix_lengths'].values()))
            if row['maximum_positions'] > 128:
                raise ValueError('Position cap exceeded')
            if row['original_joint_encoding_matches'] and row.get('pool_trim_chars', 0):
                raise ValueError('An already-valid original prefix changed')
        if len(rows) != 576 or any(sum(r['split'] == s for r in rows) != 288 for s in ['screen', 'confirm']):
            raise ValueError('Full original text coverage required')
        plan = {**plan, 'panel_sha256': digest(rows)}
        save_json(root / PREFIX / 'plan.json', plan)
    save_json(root / name, rows);volume.commit()
    if not final:
        return dict(path=name)
    files = []
    for name_ in ['spec.json', 'input/model/config.json', 'input/model/model.safetensors', 'input/model/tokenizer.json',
        'checkpoint/lm.pt', 'checkpoint/hypernet/config.json', 'checkpoint/hypernet/model.safetensors',
        PREFIX + '/panel.json', PREFIX + '/plan.json']:
        target = root / name_;files.append(dict(path=name_, bytes=target.stat().st_size, sha256=file_sha(target)))
    result = dict(plan=plan, panel_sha256=digest(rows), metadata=[{k: v for k, v in r.items() if k != 'prefix'} for r in rows], files=files)
    save_json(root / PREFIX / 'preparation.json', result);volume.commit()
    return result


@app.function(image=cpu_image, volumes={'/sweep': volume}, max_containers=30, timeout=1200, scaledown_window=2)
def copy_files(items):
    volume.reload();root = Path('/sweep')
    client = modal.Client.from_credentials(os.environ['SOURCE_MODAL_TOKEN_ID'], os.environ['SOURCE_MODAL_TOKEN_SECRET'])
    source = modal.Volume.from_name('anytok-rl-all-pool-20k', version=2, client=client)
    for item in items:
        target = root / item['path'];target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and target.stat().st_size == item['bytes'] and file_sha(target) == item['sha256']:
            continue
        temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.partial')
        for attempt in range(4):
            try:
                with temporary.open('wb') as stream:
                    for block in source.read_file(item['path']):
                        stream.write(block)
                if temporary.stat().st_size != item['bytes'] or file_sha(temporary) != item['sha256']:
                    raise ValueError('Cloud copy checksum mismatch')
                temporary.replace(target);break
            except (OSError, modal.exception.ConnectionError):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
    volume.commit();return len(items)


@app.local_entrypoint()
def main(action: str = 'prepare'):
    STATE.mkdir(parents=True, exist_ok=True)
    from modal_credits import billing_row
    credit = billing_row(PROFILE, 'this month', Decimal(30));save_json(STATE / ('credits-' + PROFILE + '.json'), credit)
    if credit['status'] != 'ok':
        raise ValueError('Cannot verify profile credit')
    if action == 'probe':
        print(available.remote());return
    if action == 'prepare':
        if PROFILE != 'blackpantherx86':
            raise ValueError('Prepare on original source profile')
        plan = json.loads((ROOT / 'plans/frozen-policy-search-v1.json').read_text())
        status('correcting all original 576 passages for full-pool scoring')
        path = initial.remote(plan)['path'];files = []
        groups = [list(range(i, i + 20)) for i in range(0, 1000, 20)]
        for round_index in range(5):
            starts = [0] * 576;lengths = {};done = 0
            for result in constraints.map(groups, [path] * 50, [round_index == 0] * 50, order_outputs=False):
                starts = [max(a, b) for a, b in zip(starts, result['starts'])];lengths.update(result['lengths'])
                files.extend(result['files']);done += result['validated']
                status('validating all 1000 tokenizers on all 576 passages', validated=done, round=round_index + 1)
            final = not any(starts)
            result = reduce_panel.remote(path, starts, lengths, final, plan, round_index + 1)
            if final:
                result['files'] = files + result['files'];save_json(STATE / 'preparation.json', result)
                status('full corrected panel prepared');return
            path = result['path']
        raise ValueError('Full common-prefix validation did not converge')
    if action == 'stage':
        from modal_tokenizer_pool_transfer import source_client
        _, secret = source_client('blackpantherx86');prepared = json.loads((STATE / 'preparation.json').read_text())
        items = prepared['files'];large = [[i] for i in items if i['bytes'] > 100_000_000]
        small = [i for i in items if i['bytes'] <= 100_000_000]
        batches = large + [small[i:i + 25] for i in range(0, len(small), 25)];done = 0
        status('copying artifacts directly between cloud volumes', files_total=len(items))
        for count in copy_files.with_options(secrets=[secret]).map(batches, order_outputs=False):
            done += count;status('copying artifacts directly between cloud volumes', files_completed=done, files_total=len(items))
        status('ready for full corrected scoring');return
    if action in ['score', 'observe']:
        if action == 'score' and Decimal(credit['estimated_remaining_usd']) < Decimal('11'):
            raise ValueError('Need $11 for two bounded 2400-second B300 calls plus margin')
        plan = json.loads((STATE / 'preparation.json').read_text())['plan']
        if action == 'score':
            calls = {str(k): score.spawn('rl20k', plan, run_prefix=PREFIX, indices=list(range(k * 500, (k + 1) * 500)),
                shard=str(k), volume_name=VOLUME) for k in range(2)}
            saved_calls = dict(app_id=app.app_id, profile=PROFILE, calls={k: v.object_id for k, v in calls.items()})
            save_json(STATE / 'calls.json', saved_calls)
        else:
            saved_calls = json.loads((STATE / 'calls.json').read_text())
            if saved_calls['profile'] != PROFILE:
                raise ValueError('Observe on the original scoring profile')
            calls = {k: modal.FunctionCall.from_id(v) for k, v in saved_calls['calls'].items()
                if not (STATE / ('scores-' + k + '.json.gz')).exists()}
        workers = {};started = time.monotonic()
        status('scoring full corrected 1000-tokenizer reward table')
        while calls:
            for k, call in list(calls.items()):
                try:
                    result = call.get(timeout=10)
                    with gzip.open(STATE / ('scores-' + k + '.json.gz'), 'wt') as stream:
                        json.dump(result, stream)
                    del calls[k]
                except (TimeoutError, modal.exception.TimeoutError):
                    pass
            for k in range(2):
                try:
                    raw = b''.join(volume.read_file(PREFIX + f'/status/rl20k-{k}.json'))
                    workers[str(k)] = json.loads(raw.decode('utf-8'))
                except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError):
                    workers.setdefault(str(k), dict(completed=0, stage='loading / awaiting readable status'))
            done = sum(x['completed'] for x in workers.values());elapsed = time.monotonic() - started
            status('scoring full corrected 1000-tokenizer reward table', done, workers=workers,
                elapsed_seconds=elapsed, eta_seconds=elapsed / done * (1000 - done) if done else None)
        status('complete', 1000);save_json(STATE / 'credits-after.json', billing_row(PROFILE, 'this month', Decimal(30)))
        if action == 'observe':
            import subprocess, sys
            subprocess.run([sys.executable, '-m', 'modal', 'app', 'stop', saved_calls['app_id'], '--yes'], check=True)
        return
    raise ValueError('Unknown action')
