"""Compare the fixed24-tokenizer cohort against every member of the native pool."""
from datetime import datetime, timezone
from decimal import Decimal
import io
import json
import os
from pathlib import Path
import sys
import time
import uuid

import modal
from pool_generalization import digest, file_sha, save_json

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'modal_results/pool_regime_comparison'
PROFILE = os.environ.get('POOL_REGIME_PROFILE', os.environ.get('MODAL_PROFILE', ''))
if PROFILE in ['general', 'research-everettcc']:
    raise ValueError('Disabled profile')
PREFIX = 'pool-regime-v1'
SOURCE_PROFILE = 'blackpantherx86'
app = modal.App('anytok-pool-regime-comparison')
volume = modal.Volume.from_name('anytok-rl-all-pool-20k' if PROFILE == SOURCE_PROFILE
    else 'anytok-pool-regime-comparison', create_if_missing=PROFILE != SOURCE_PROFILE, version=2)
BASE = ROOT / 'runtime/anytok_pool_20k'
cpu_image = (modal.Image.debian_slim(python_version='3.12')
    .uv_pip_install('tokenizers==0.22.2', 'numpy==2.3.5')
    .env({'TOKENIZERS_PARALLELISM': 'false', 'POOL_REGIME_PROFILE': PROFILE})
    .add_local_python_source('pool_regime', 'pool_generalization', 'natural_transfer', 'uniform_trajectory', 'uniform_weakness'))


def now():
    return datetime.now(timezone.utc).isoformat()


def status(stage, completed=0, **fields):
    row = dict(at_utc=now(), stage=stage, completed=completed, total=3000, profile=PROFILE, **fields)
    save_json(STATE / 'status.json', row)
    with (STATE / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


@app.function(image=cpu_image, volumes={'/sweep': volume}, timeout=120, scaledown_window=2)
def available():
    return dict(profile=PROFILE, at_utc=now())


@app.function(image=cpu_image, volumes={'/sweep': volume}, timeout=300, scaledown_window=2)
def initial_panel():
    volume.reload();root = Path('/sweep')
    old = json.loads((root / 'natural-model-comparison-v1/plan.json').read_text())
    rows = json.loads((root / 'natural-transfer-aligned-v2/panel.json').read_text())
    if digest(rows) != old['panel_sha256']:
        raise ValueError('Corrected source panel changed')
    selected = set(old['context_selected_ids'])
    rows = [r for r in rows if r['id'] in selected and r['split'] == 'confirm']
    if len(rows) != 229:
        raise ValueError('Expected229 fixed confirmation passages')
    save_json(root / PREFIX / 'round-0.json', rows);volume.commit()
    return dict(panel_sha256=digest(rows), rows=len(rows), path=PREFIX + '/round-0.json')


@app.function(image=cpu_image, volumes={'/sweep': volume}, max_containers=50, timeout=600, scaledown_window=2)
def constraints(indices, panel_path, collect_files):
    from uniform_trajectory import CanonicalTokenizer, load_candidate
    from pool_regime import required_start
    volume.reload();root = Path('/sweep')
    spec = json.loads((root / 'spec.json').read_text())
    rows = json.loads((root / panel_path).read_text())
    canonical = CanonicalTokenizer(root / 'input/model/tokenizer.json')
    starts = [0] * len(rows);lengths = {};files = []
    for index in indices:
        path = root / f'input/pool/tokenizers/{index:04d}/tokenizer.json'
        if file_sha(path) != spec['tokenizers'][index]['sha256']:
            raise ValueError('Native tokenizer changed')
        tokenizer, _, _, _, forbidden = load_candidate(path, 'pool')
        length = []
        for j, row in enumerate(rows):
            start, count = required_start(row, canonical, tokenizer, forbidden | {0})
            starts[j] = max(starts[j], start);length.append(count)
        lengths[str(index)] = length
        if collect_files:
            folder = root / f'features/{index:04d}'
            manifest = json.loads((folder / 'manifest.json').read_text())
            if manifest['tokenizer_sha256'] != spec['tokenizers'][index]['sha256']:
                raise ValueError('Feature mapping changed')
            for name, sha in [('tokenizer', spec['tokenizers'][index]['sha256']),
                    ('forms', manifest['forms_sha256']), ('priors', manifest['priors_sha256']),
                    ('manifest', file_sha(folder / 'manifest.json'))]:
                source = path if name == 'tokenizer' else folder / (name + ('.json' if name == 'manifest' else '.npy'))
                files.append(dict(source=str(source.relative_to(root)), destination=str(source.relative_to(root)),
                                  bytes=source.stat().st_size, sha256=sha))
    return dict(indices=indices, starts=starts, lengths=lengths, files=files, panel_sha256=digest(rows))


@app.function(image=cpu_image, volumes={'/sweep': volume}, timeout=300, scaledown_window=2)
def finish_panel(panel_path, starts, lengths, final, local_plan):
    from pool_regime import shorten
    from uniform_trajectory import CanonicalTokenizer
    volume.reload();root = Path('/sweep')
    rows = json.loads((root / panel_path).read_text())
    canonical = CanonicalTokenizer(root / 'input/model/tokenizer.json')
    rows = shorten(rows, starts, canonical)
    if final:
        for j, row in enumerate(rows):
            row['native_prefix_lengths'] = {key: counts[j] for key, counts in lengths.items()}
            if len(row['native_prefix_lengths']) != 1000:
                raise ValueError('Missing a native prefix validation')
            row['maximum_positions'] = 33 + max(len(row['canonical_prefix_ids']), max(row['native_prefix_lengths'].values()))
        name = PREFIX + '/panel.json'
    else:
        name = PREFIX + '/round-' + str(local_plan['round']) + '.json'
    save_json(root / name, rows);volume.commit()
    if not final:
        return dict(path=name, panel_sha256=digest(rows))
    spec = json.loads((root / 'spec.json').read_text())
    parent = json.loads((root / 'natural-model-comparison-v1/plan.json').read_text())
    models = [dict(name='rl20k', source='checkpoint', manifest=spec['checkpoint_manifest'])]
    models += [dict(name=m['name'], source='natural-model-comparison-v1/models/' + m['name'], manifest=m['manifest']) for m in parent['models']]
    plan = {**local_plan, 'models': models, 'panel_sha256': digest(rows),
        'source_spec_sha256': digest(spec), 'position_cap': 128, 'suffix_tokens': 32,
        'tokenizers': spec['tokenizers'], 'context_thresholds': parent['context_thresholds']}
    save_json(root / PREFIX / 'plan.json', plan)
    items = []
    def add(source, destination, sha=None):
        path = root / source
        items.append(dict(source=source, destination=destination, bytes=path.stat().st_size, sha256=sha or file_sha(path)))
    for model in models:
        for filename in ['lm.pt', 'hypernet/config.json', 'hypernet/model.safetensors']:
            add(model['source'] + '/' + filename, 'models/' + model['name'] + '/' + filename,
                model['manifest']['files'][filename]['sha256'])
    for filename in ['model/config.json', 'model/model.safetensors', 'model/tokenizer.json']:
        add('input/' + filename, 'input/' + filename, spec['input_hashes'][filename])
    add('spec.json', 'spec.json');add(PREFIX + '/panel.json', 'panel.json');add(PREFIX + '/plan.json', 'plan.json')
    volume.commit()
    return dict(path=name, panel_sha256=digest(rows), plan=plan, files=items,
        metadata=[{k: v for k, v in r.items() if k != 'prefix'} for r in rows],
        trimmed_passages=sum(r.get('pool_trim_chars', 0) > 0 for r in rows),
        continuation_tokens=32 * len(rows), continuation_bytes=sum(r['continuation_bytes'] for r in rows))


def prepare():
    if PROFILE != SOURCE_PROFILE:
        raise ValueError('Prepare shared inputs on source profile')
    local_plan = json.loads((ROOT / 'plans/pool-regime-comparison-v1.json').read_text())
    status('validating identical stress passages across all1000 native tokenizers')
    initial = initial_panel.remote();path = initial['path'];files = []
    for round_index in range(5):
        starts = [0] * 229;lengths = {}
        groups = [list(range(i, min(i + 20, 1000))) for i in range(0, 1000, 20)]
        for result in constraints.map(groups, [path] * len(groups), [round_index == 0] * len(groups), order_outputs=False):
            starts = [max(a, b) for a, b in zip(starts, result['starts'])]
            lengths.update(result['lengths']);files.extend(result['files'])
        final = not any(starts)
        result = finish_panel.remote(path, starts, lengths, final, {**local_plan, 'round': round_index + 1})
        if final:
            result['files'] = files + result['files']
            save_json(STATE / 'preparation.json', result);save_json(STATE / 'plan.json', result['plan'])
            status('shared panel validated; ready to stage cloud inputs', trimmed_passages=result['trimmed_passages'])
            return
        path = result['path']
    raise ValueError('Shared-context constraints did not converge')


@app.local_entrypoint()
def main(action: str = 'prepare'):
    STATE.mkdir(parents=True, exist_ok=True)
    try:
        if action == 'prepare':
            prepare()
        elif action == 'probe':
            print(json.dumps(available.remote()), flush=True)
        else:
            raise ValueError('Unknown action')
    except Exception as exc:
        status('failed', error=str(exc));raise
