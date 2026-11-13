"""Compare final ZeTT, uniform-pool and RL models on the fixed corrected text panel."""
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
STATE = ROOT / 'modal_results/natural_model_comparison'
PROFILE = 'blackpantherx86'
PREFIX = 'natural-model-comparison-v1'
PARENT = 'natural-transfer-aligned-v2'
app = modal.App('anytok-natural-model-comparison')
volume = modal.Volume.from_name('anytok-rl-all-pool-20k', version=2)
BASE = ROOT / 'runtime/anytok_pool_20k'
cpu_image = modal.Image.debian_slim(python_version='3.12').add_local_python_source('pool_generalization')
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
    row = dict(at_utc=now(), stage=stage, completed=completed, total=76, profile=PROFILE, **details)
    save_json(STATE / 'status.json', row)
    with (STATE / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


@app.function(image=cpu_image, volumes={'/sweep': volume}, max_containers=6, timeout=1800, scaledown_window=2)
def copy_weight(model, name):
    from grpclib.exceptions import StreamTerminatedError
    client = modal.Client.from_credentials(os.environ['SOURCE_MODAL_TOKEN_ID'], os.environ['SOURCE_MODAL_TOKEN_SECRET'])
    source = modal.Volume.from_name(model['source_volume'], version=2, client=client)
    target = Path('/sweep') / PREFIX / 'models' / model['name'] / name
    expected = model['manifest']['files'][name]
    if target.exists() and target.stat().st_size == expected['bytes'] and file_sha(target) == expected['sha256']:
        return dict(model=model['name'], name=name, reused=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.partial')
    for attempt in range(4):
        try:
            with temporary.open('wb') as stream:
                for block in source.read_file(model['source_checkpoint'] + '/' + name):
                    stream.write(block)
            if temporary.stat().st_size != expected['bytes'] or file_sha(temporary) != expected['sha256']:
                raise ValueError('Cloud checkpoint transfer failed checksum: ' + name)
            temporary.replace(target)
            volume.commit()
            return dict(model=model['name'], name=name, reused=False, bytes=expected['bytes'])
        except (OSError, StreamTerminatedError, modal.exception.ConnectionError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


@app.function(image=gpu_image, gpu='B300', volumes={'/sweep': volume}, max_containers=2,
              timeout=1200, scaledown_window=2)
def evaluate(model_name, plan):
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
    folder = root / PREFIX
    if json.loads((folder / 'plan.json').read_text()) != plan:
        raise ValueError('Comparison plan changed')
    rows = json.loads((root / PARENT / 'panel.json').read_text())
    if digest(rows) != plan['panel_sha256'] or not all(r['canonical_joint_encoding_matches'] for r in rows):
        raise ValueError('Corrected panel changed')
    source_model = next(r for r in plan['models'] if r['name'] == model_name)
    checkpoint = folder / 'models' / model_name
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
def main(action: str = 'run'):
    if os.environ.get('MODAL_PROFILE') != PROFILE:
        raise ValueError('Use ' + PROFILE)
    STATE.mkdir(parents=True, exist_ok=True)
    plan = json.loads((ROOT / 'plans/natural-model-comparison-v1.json').read_text())
    save_json(STATE / 'plan.json', plan)
    from modal_credits import billing_row
    balance = billing_row(PROFILE, 'this month', Decimal(30))
    save_json(STATE / 'credits-before.json', balance)
    if balance['status'] != 'ok' or Decimal(balance['estimated_remaining_usd']) < Decimal('4'):
        raise ValueError('At least$4 verified remaining credits required')
    try:
        status('transferring final inference weights cloud-to-cloud')
        with volume.batch_upload(force=True) as batch:
            batch.put_file(STATE / 'plan.json', PREFIX + '/plan.json')
        from modal_tokenizer_pool_transfer import source_client
        # Each source uses its own credentials; model bytes stay inside the cloud.
        calls = []
        for item in plan['models']:
            _, secret = source_client(item['source_profile'])
            for name in ['lm.pt', 'hypernet/config.json', 'hypernet/model.safetensors']:
                calls.append(copy_weight.with_options(secrets=[secret]).spawn(item, name))
        for call in calls:
            print(json.dumps(call.get()), flush=True)
        if action == 'prepare':
            status('prepared');return
        if action != 'run':
            raise ValueError('Unknown action')
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
            for item in plan['models']:
                name = item['name']
                try:
                    progress[name] = json.loads(b''.join(volume.read_file(PREFIX + '/status/' + name + '.json')))
                except FileNotFoundError:
                    progress[name] = dict(completed=0, stage='loading model')
            save_json(STATE / 'workers.json', progress)
            status('comparing trained models', sum(r['completed'] for r in progress.values()), workers=progress)
        status('complete', len(plan['indices']) * len(plan['models']))
        save_json(STATE / 'credits-after.json', billing_row(PROFILE, 'this month', Decimal(30)))
    except Exception as exc:
        status('failed', error=str(exc));raise
