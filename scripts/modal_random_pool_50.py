"""Evaluate a fixed uniform random50 pool sample against the selected24 on identical passages."""
from datetime import datetime, timezone
from decimal import Decimal
import gzip
import json
import os
from pathlib import Path
import sys
import time
import uuid

import modal
from pool_generalization import digest, file_sha, save_json

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'modal_results/random_pool_50'
PROFILE = 'blackpantherx86'
PREFIX = 'random-pool-50-v1'
PARENT = 'natural-transfer-aligned-v2'
app = modal.App('anytok-random-pool-50')
volume = modal.Volume.from_name('anytok-rl-all-pool-20k', version=2)
BASE = ROOT / 'runtime/anytok_pool_20k'
cpu_image = (modal.Image.debian_slim(python_version='3.12')
    .uv_pip_install('tokenizers==0.22.2', 'numpy==2.3.5')
    .env({'TOKENIZERS_PARALLELISM': 'false'})
    .add_local_python_source('pool_generalization', 'pool_regime', 'uniform_trajectory', 'uniform_weakness'))
gpu_image = (modal.Image.from_registry('ubuntu:24.04', add_python='3.12')
    .apt_install('libpython3.12')
    .uv_pip_install('torch==2.15.0', 'transformers==5.15.1', 'tokenizers==0.22.2',
                    'numpy==2.3.5', 'safetensors==0.8.0', 'accelerate==1.14.0')
    .env({'TOKENIZERS_PARALLELISM': 'false'})
    .add_local_dir(BASE / 'hf_hypernet', '/opt/zett/hf_hypernet')
    .add_local_file(BASE / 'rust_utils/target/release/librust_utils.so',
                    '/opt/zett/rust_utils/target/release/librust_utils.so')
    .add_local_python_source('natural_transfer', 'pool_generalization', 'uniform_trajectory', 'uniform_weakness'))


def now():
    return datetime.now(timezone.utc).isoformat()


def status(stage, completed=0, **details):
    row = dict(at_utc=now(), stage=stage, completed=completed, total=216, profile=PROFILE, **details)
    save_json(STATE / 'status.json', row)
    with (STATE / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


@app.function(image=cpu_image, volumes={'/sweep': volume}, timeout=600, scaledown_window=2)
def prepare(original_plan):
    from pool_regime import required_start, shorten
    from uniform_trajectory import CanonicalTokenizer, load_candidate
    volume.reload();root = Path('/sweep');folder = root / PREFIX
    if (folder / 'preparation.json').exists():
        old = json.loads((folder / 'preparation.json').read_text())
        if old['source_plan_sha256'] != digest(original_plan):
            raise ValueError('Cannot change the frozen random sample')
        return old
    rows = json.loads((root / PARENT / 'panel.json').read_text())
    if digest(rows) != original_plan['parent_panel_sha256']:
        raise ValueError('Parent panel changed')
    selected = set(original_plan['context_selected_ids'])
    rows = [r for r in rows if r['split'] == 'confirm' and r['id'] in selected]
    if len(rows) != 229:
        raise ValueError('Expected229 original stress passages')
    spec = json.loads((root / 'spec.json').read_text())
    canonical = CanonicalTokenizer(root / 'input/model/tokenizer.json')
    for _ in range(5):
        starts = [0] * len(rows);lengths = {}
        for index in original_plan['indices']:
            path = root / f'input/pool/tokenizers/{index:04d}/tokenizer.json'
            if file_sha(path) != spec['tokenizers'][index]['sha256']:
                raise ValueError('Native tokenizer changed')
            tokenizer, _, _, _, forbidden = load_candidate(path, 'pool')
            lengths[str(index)] = []
            for j, row in enumerate(rows):
                start, count = required_start(row, canonical, tokenizer, forbidden | {0})
                starts[j] = max(starts[j], start);lengths[str(index)].append(count)
        if any(starts):
            rows = shorten(rows, starts, canonical)
        else:
            break
    else:
        raise ValueError('Common-prefix validation did not converge')
    for j, row in enumerate(rows):
        row['native_prefix_lengths'] = {k: v[j] for k, v in lengths.items()}
        row['maximum_positions'] = 33 + max(len(row['canonical_prefix_ids']), max(row['native_prefix_lengths'].values()))
    plan = {**original_plan, 'panel_sha256': digest(rows)}
    result = dict(plan=plan, source_plan_sha256=digest(original_plan), panel_sha256=digest(rows),
        metadata=[{k: v for k, v in row.items() if k != 'prefix'} for row in rows],
        trimmed_passages=sum(r.get('pool_trim_chars', 0) > 0 for r in rows),
        continuation_tokens=32 * len(rows), continuation_bytes=sum(r['continuation_bytes'] for r in rows))
    save_json(folder / 'panel.json', rows);save_json(folder / 'plan.json', plan)
    save_json(folder / 'preparation.json', result);volume.commit()
    return result


@app.function(image=gpu_image, gpu='B300', volumes={'/sweep': volume}, max_containers=2,
              timeout=600, scaledown_window=2)
def evaluate(model_name, plan, run_prefix=PREFIX):
    sys.path.insert(0, '/opt/zett')
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from hf_hypernet.modeling_hypernet import ZettHypernet
    from hf_hypernet.smollm_setup import separate_input_embeddings_and_freeze_output
    from hf_hypernet.tokenizer_inputs import enable_byte_fallbacks
    from uniform_trajectory import load_candidate
    from natural_transfer import score_rows, validate_scores
    torch.set_num_threads(1)
    volume.reload()
    root = Path('/sweep')
    spec = json.loads((root / 'spec.json').read_text())
    folder = root / run_prefix
    if json.loads((folder / 'plan.json').read_text()) != plan:
        raise ValueError('Comparison plan changed')
    rows = json.loads((folder / 'panel.json').read_text())
    if digest(rows) != plan['panel_sha256'] or not all(r['canonical_joint_encoding_matches'] for r in rows):
        raise ValueError('Corrected panel changed')
    source_model = next(r for r in plan['models'] if r['name'] == model_name)
    checkpoint = root / source_model['checkpoint_path']
    for name in ['lm.pt', 'hypernet/config.json', 'hypernet/model.safetensors']:
        expected = source_model['manifest']['files'][name]
        if (checkpoint / name).stat().st_size != expected['bytes'] or file_sha(checkpoint / name) != expected['sha256']:
            raise ValueError('Checkpoint changed')
    for name in ['model/tokenizer.json', 'model/model.safetensors', 'model/config.json']:
        if file_sha(root / 'input' / name) != spec['input_hashes'][name]:
            raise ValueError('Original model input changed')
    def emit(stage, completed):
        record = dict(at_utc=now(), stage=stage, completed=completed, total=len(plan['indices']),
                      model=model_name, gpu=torch.cuda.get_device_name())
        save_json(folder / 'status' / (model_name + '.json'), record)
        volume.commit()
        print(json.dumps(record), flush=True)
    canonical = AutoTokenizer.from_pretrained(root / 'input/model', local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(root / 'input/model', local_files_only=True,
        dtype=torch.float32, low_cpu_mem_usage=True).to('cuda')
    separate_input_embeddings_and_freeze_output(model)
    original = model.get_input_embeddings().weight.detach().clone()
    model.load_state_dict(torch.load(checkpoint / 'lm.pt', map_location='cpu', weights_only=True))
    model.eval()
    source = model.get_input_embeddings().weight.detach()
    if not torch.equal(source, original) or not torch.equal(model.get_output_embeddings().weight, original):
        raise ValueError('Canonical tables changed')
    del original
    hypernet = ZettHypernet.from_pretrained(checkpoint / 'hypernet', local_files_only=True).to('cuda').eval()
    fallbacks = enable_byte_fallbacks(hypernet, canonical)
    emit('scoring model-specific canonical controls', 0)
    base_path = folder / model_name / 'canonical.json'
    if base_path.exists():
        controls = json.loads(base_path.read_text())
        if controls['plan_sha256'] != digest(plan):
            raise ValueError('Canonical controls belong to another comparison')
    else:
        controls = dict(plan_sha256=digest(plan), canonical=score_rows(model, source, source, rows),
                        no_prefix=score_rows(model, source, source, rows, empty_prefix=True))
        save_json(base_path, controls);volume.commit()
    for values in [controls['canonical'], controls['no_prefix']]:
        validate_scores(rows, values)
    results = []
    for index in plan['indices']:
        target = folder / model_name / f'{index:04d}.json'
        if target.exists():
            result = json.loads(target.read_text())
            if result['plan_sha256'] != digest(plan):
                raise ValueError('Existing scores belong to another comparison')
            results.append(result);emit('scoring fixed natural-text panel', len(results));continue
        begin = time.monotonic()
        token_path = root / f'input/pool/tokenizers/{index:04d}/tokenizer.json'
        if file_sha(token_path) != spec['tokenizers'][index]['sha256']:
            raise ValueError('Native tokenizer changed')
        tokenizer, _, _, _, forbidden = load_candidate(token_path, 'pool')
        for row in rows:
            ids = tokenizer.encode(row['prefix'], add_special_tokens=False).ids
            if (len(ids) != row['native_prefix_lengths'][str(index)] or forbidden.intersection(ids) or 0 in ids
                    or tokenizer.decode(ids, skip_special_tokens=False) != row['prefix']):
                raise ValueError('Native prefix changed')
        features = root / f'features/{index:04d}'
        manifest = json.loads((features / 'manifest.json').read_text())
        if manifest['fallbacks'] != fallbacks or manifest['tokenizer_sha256'] != spec['tokenizers'][index]['sha256']:
            raise ValueError('Hypernetwork feature mapping changed')
        for name in ['forms', 'priors']:
            if file_sha(features / (name + '.npy')) != manifest[name + '_sha256']:
                raise ValueError('Feature arrays changed')
        forms = torch.from_numpy(np.load(features / 'forms.npy').astype(np.int64)).to('cuda')
        priors = torch.from_numpy(np.load(features / 'priors.npy')).to('cuda')
        if forms.shape[1] != hypernet.config.hn_surface_maxlen:
            raise ValueError('Surface feature lengths differ between models')
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            generated, output, _ = hypernet(forms, priors, source, None, deterministic=True)
        if output is not None or not torch.isfinite(generated).all():
            raise ValueError('Invalid full-vocabulary embeddings')
        transferred = score_rows(model, generated, source, rows, tokenizer)
        validate_scores(rows, transferred)
        result = dict(model=model_name, index=index, config=spec['tokenizers'][index]['config'],
                      transferred=transferred, plan_sha256=digest(plan), elapsed_seconds=time.monotonic() - begin)
        save_json(target, result);volume.commit();results.append(result)
        del forms, priors, generated
        torch.cuda.empty_cache()
        emit('scoring fixed natural-text panel', len(results))
    final = {**controls, 'model': model_name, 'panel_sha256': plan['panel_sha256'],
             'results': results, 'gpu': torch.cuda.get_device_name()}
    save_json(folder / model_name / 'results.json', final);volume.commit()
    emit('complete', len(results))
    return final


@app.local_entrypoint()
def main():
    if os.environ.get('MODAL_PROFILE') != PROFILE:
        raise ValueError('Use ' + PROFILE)
    STATE.mkdir(parents=True, exist_ok=True)
    try:
        original = json.loads((ROOT / 'plans/random-pool-50-v1.json').read_text())
        status('validating the fixed random50 and selected24 on shared passages')
        prepared = prepare.remote(original)
        save_json(STATE / 'preparation.json', prepared);save_json(STATE / 'plan.json', prepared['plan'])
        plan = prepared['plan']
        from modal_credits import billing_row
        credit = billing_row(PROFILE, 'this month', Decimal(30));save_json(STATE / 'credits-before.json', credit)
        if credit['status'] != 'ok' or Decimal(credit['estimated_remaining_usd']) < Decimal('4.1'):
            raise ValueError('Need$4.10 remaining to cover three bounded600-second calls with a margin')
        status('scoring random50 and selected24', trimmed_passages=prepared['trimmed_passages'])
        calls = {m['name']: evaluate.spawn(m['name'], plan) for m in plan['models']}
        save_json(STATE / 'calls.json', dict(app_id=app.app_id, calls={n: c.object_id for n, c in calls.items()}))
        while calls:
            for name, call in list(calls.items()):
                try:
                    result = call.get(timeout=10)
                    with gzip.open(STATE / (name + '.json.gz'), 'wt') as stream:
                        json.dump(result, stream)
                    del calls[name]
                except (TimeoutError, modal.exception.TimeoutError):
                    pass
            progress = {}
            for model in plan['models']:
                name = model['name']
                try:
                    progress[name] = json.loads(b''.join(volume.read_file(PREFIX + '/status/' + name + '.json')))
                except FileNotFoundError:
                    progress[name] = dict(completed=0, stage='waiting/loading')
            save_json(STATE / 'workers.json', progress)
            status('scoring random50 and selected24', sum(r['completed'] for r in progress.values()), workers=progress)
        status('complete', len(plan['indices']) * len(plan['models']))
        save_json(STATE / 'credits-after.json', billing_row(PROFILE, 'this month', Decimal(30)))
    except Exception as exc:
        status('failed', error=str(exc));raise


