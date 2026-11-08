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
SOURCE = ROOT / 'runtime/anytok_pool_20k'
gpu_image = (modal.Image.from_registry('ubuntu:24.04', add_python='3.12')
    .apt_install('libpython3.12')
    .uv_pip_install('torch==2.15.0', 'transformers==5.15.1', 'tokenizers==0.22.2',
                    'numpy==2.3.5', 'safetensors==0.8.0', 'accelerate==1.14.0')
    .env({'TOKENIZERS_PARALLELISM': 'false'})
    .add_local_dir(str(SOURCE / 'hf_hypernet'), '/opt/zett/hf_hypernet')
    .add_local_file(str(SOURCE / 'rust_utils/target/release/librust_utils.so'),
                    '/opt/zett/rust_utils/target/release/librust_utils.so')
    .add_local_python_source('uniform_trajectory', 'uniform_weakness', 'pool_ood_eval', 'pool_generalization'))


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


@app.function(image=cpu_image, volumes={'/train': training.with_mount_options(read_only=True), '/out': outputs}, timeout=300)
def benchmark_wordpiece():
    from tokenizers import Tokenizer
    from uniform_trajectory import profile_samples
    with gzip.open('/train/input/eval/web-zh.jsonl.gz', 'rt', encoding='utf-8') as stream:
        sample = profile_samples([json.loads(line) for line in stream], 'web-zh')[0]['text']
    measurements = []
    for index in [63, 314, 103]:
        tokenizer = Tokenizer.from_file(f'/train/input/pool/tokenizers/{index:04d}/tokenizer.json')
        for size in [256, 512, 1024]:
            text = sample[:size]; begin = time.perf_counter()
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            measurements.append({'index': index, 'characters': len(text), 'bytes': len(text.encode('utf-8')),
                                 'seconds': time.perf_counter() - begin, 'tokens': len(ids)})
    save_json('/out/wordpiece-runtime.json', measurements); outputs.commit(); return measurements


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


def native_candidates():
    previous = json.loads(Path('/out/aux/report.json').read_text())
    result = [{**c, 'path': f"/out/aux/tokenizers/{c['id']}/tokenizer.json"} for c in previous['candidates']]
    for row in json.loads(Path('/out/pool-candidates.json').read_text()):
        if row['config']['algorithm'] == 'wordpiece' and row['config']['pretokenizer'] == 'no-regex':
            continue  # Measured severe long-input encoding cost; not a primary practical case.
        result.append({'id': f"pool-{row['index']:04d}", 'kind': 'pool', 'sha256': row['sha256'],
                       'vocab_size': row['config']['vocab_size'], 'config': row['config'],
                       'path': f"/train/input/pool/tokenizers/{row['index']:04d}/tokenizer.json",
                       'selected_for': row['selected_for']})
    return result


def load_scoring_model(step):
    import torch
    from transformers import AutoModelForCausalLM
    from hf_hypernet.modeling_hypernet import ZettHypernet
    from hf_hypernet.smollm_setup import separate_input_embeddings_and_freeze_output
    cp = Path('/train/run') / f'step-{step:06d}'
    inventory = json.loads(Path('/out/inventory.json').read_text())
    expected = next(c['manifest'] for c in inventory['checkpoints'] if c['manifest']['step'] == step)
    actual = json.loads((cp / 'manifest.json').read_text())
    for name in ['lm.pt', 'hypernet/config.json', 'hypernet/model.safetensors']:
        if actual['files'][name] != expected['files'][name] or file_sha(cp / name) != expected['files'][name]['sha256']:
            raise ValueError(f'Checkpoint changed: {step}/{name}')
    model = AutoModelForCausalLM.from_pretrained('/train/input/model', local_files_only=True,
                dtype=torch.float32, low_cpu_mem_usage=True).to('cuda')
    separate_input_embeddings_and_freeze_output(model); model.eval()
    reference = model.get_input_embeddings().weight.detach().clone()
    model.load_state_dict(torch.load(cp / 'lm.pt', map_location='cpu', weights_only=True))
    table = model.get_input_embeddings().weight.detach()
    if not torch.equal(table, reference) or not torch.equal(model.get_output_embeddings().weight, reference):
        raise ValueError('Frozen canonical tables changed')
    del reference
    hypernet = ZettHypernet.from_pretrained(cp / 'hypernet', local_files_only=True).to('cuda').eval()
    return model, table, hypernet


@app.function(image=gpu_image, gpu='H200', volumes={'/train': training.with_mount_options(read_only=True), '/out': outputs},
              max_containers=4, timeout=3600)
def screen(candidate_ids):
    sys.path.insert(0, '/opt/zett')
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from hf_hypernet.tokenizer_inputs import build_surface_forms, enable_byte_fallbacks
    from uniform_trajectory import PROFILE_DOMAINS, fixed_document, load_candidate
    from uniform_weakness import compare_scores, compression
    from pool_ood_eval import examples_for, score_documents
    destination = Path('/out/screens') / digest(candidate_ids)[:12]
    destination.mkdir(parents=True, exist_ok=True)
    def emit(stage, **details):
        value = {'at_utc': now(), 'stage': stage, **details}
        save_json(destination / 'status.json', value); outputs.commit(); print(json.dumps(value), flush=True)
    emit('loading final20k model for screening')
    canonical = AutoTokenizer.from_pretrained('/train/input/model', local_files_only=True)
    model, table, hypernet = load_scoring_model(20000)
    fallbacks = enable_byte_fallbacks(hypernet, canonical)
    candidates = [c for c in native_candidates() if c['id'] in candidate_ids]; rows_by_domain = {}
    for domain in PROFILE_DOMAINS:
        with gzip.open(f'/train/input/eval/{domain}.jsonl.gz', 'rt', encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        rows_by_domain[domain] = sorted(rows, key=lambda r: hashlib.sha256(('trajectory-screen/' + domain + '/' + r['id']).encode()).digest())
    result = []; errors = []; canonical_cache = {}; valid_candidates = []
    for candidate in candidates:
        name = candidate['id']; emit('screening useful native tokenizers', tokenizer=name, completed=len(valid_candidates), total=len(candidates))
        try:
            if file_sha(candidate['path']) != candidate['sha256']: raise ValueError('Tokenizer checksum changed')
            tokenizer, tokens, priors, specials, forbidden = load_candidate(candidate['path'], candidate['kind'])
            forms11, _ = build_surface_forms(tokens, canonical, 11, fallbacks, specials)
            forms_np = forms11[:, :10].copy(); truncated_mask = forms11[:, 10] != len(canonical)
            folder = Path('/out/features') / name; folder.mkdir(parents=True, exist_ok=True)
            np.save(folder / 'forms.npy', forms_np); np.save(folder / 'truncated.npy', truncated_mask)
            np.save(folder / 'priors.npy', np.asarray(priors, dtype=np.float32))
            save_json(folder / 'manifest.json', {'tokenizer_sha256': candidate['sha256'], 'fallbacks': fallbacks,
                      'forms_sha256': file_sha(folder / 'forms.npy'), 'priors_sha256': file_sha(folder / 'priors.npy'),
                      'truncated_sha256': file_sha(folder / 'truncated.npy'), 'max_pieces': 10})
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                generated, output, _ = hypernet(torch.from_numpy(forms_np.astype(np.int64)).to('cuda'),
                    torch.tensor(priors, dtype=torch.float32, device='cuda'), table, None, deterministic=True)
            if output is not None or not torch.isfinite(generated).all(): raise ValueError('Invalid generated embeddings')
            for domain, raw_rows in rows_by_domain.items():
                panel = []; rejected = 0
                for raw in raw_rows:
                    try:
                        row = fixed_document(raw, canonical, tokenizer, forbidden)
                        text = row['compression_text']; ids = tokenizer.encode(text, add_special_tokens=False).ids
                        if tokenizer.decode(ids, skip_special_tokens=False) != text: raise ValueError('Utility excerpt changes under normalization')
                        panel.append(row)
                    except ValueError: rejected += 1
                    if len(panel) == 48: break
                if len(panel) != 48:
                    errors.append({'tokenizer': name, 'domain': domain, 'error': 'insufficient reversible screen documents'}); continue
                context = digest([(r['id'], r['prefix'], r['canonical_continuation_ids'], r['continuation_bytes']) for r in panel])
                if context not in canonical_cache:
                    canonical_cache[context] = score_documents(model, examples_for(panel), table, table)
                nats = score_documents(model, examples_for(panel, tokenizer), generated, table)
                score = compare_scores(nats, canonical_cache[context], [r['continuation_bytes'] for r in panel])
                result.append({'tokenizer': name, 'domain': domain, 'score': score,
                               'utility': compression(panel, tokenizer, canonical),
                               'screen_ids': [r['id'] for r in panel], 'context_sha256': context, 'rejected': rejected})
            valid_candidates.append(candidate)
            del generated, forms11; torch.cuda.empty_cache()
        except (ValueError, RuntimeError) as exc:
            errors.append({'tokenizer': name, 'error': str(exc)})
            if 'generated' in locals(): del generated
            torch.cuda.empty_cache()
        save_json(destination / 'screen.json', result); save_json(destination / 'errors.json', errors); outputs.commit()
    emit('screen shard complete', candidates=len(valid_candidates), errors=errors)
    return {'candidates': valid_candidates, 'screen': result, 'errors': errors}


@app.function(image=cpu_image, volumes={'/train': training.with_mount_options(read_only=True), '/out': outputs}, timeout=3600)
def screen_all():
    from uniform_trajectory import select_trajectory_cases
    candidates = native_candidates(); names = {c['id'] for c in candidates}
    cached = {}
    for path in Path('/out/screens').glob('*/screen.json'):
        for row in json.loads(path.read_text()):
            if row['tokenizer'] in names: cached[(row['tokenizer'], row['domain'])] = row
    complete = {name for name in names if len([r for (n, _), r in cached.items() if n == name]) == 9}
    result = [r for (name, _), r in cached.items() if name in complete]
    valid_candidates = [c for c in candidates if c['id'] in complete]
    errors = []
    missing = sorted(names - complete)
    chunks = [missing[i::4] for i in range(4) if missing[i::4]]
    for shard in screen.map(chunks, order_outputs=False):
        result.extend(shard['screen']); errors.extend(shard['errors']); valid_candidates.extend(shard['candidates'])
    cases = select_trajectory_cases(result, valid_candidates)
    used = {r['tokenizer'] for r in cases}
    # Backfill missing diversity tiers with further useful, weak distinct files.
    for row in sorted(result, key=lambda r: r['score']['gap_bpb'], reverse=True):
        if len({c['tokenizer'] for c in cases if c['cohort'] == 'fresh' and c['reason'] != 'ordinary English control'}) >= 8: break
        if row['tokenizer'] in used or row['score']['gap_bpb'] < .08 or row['utility']['token_reduction_vs_canonical_percent'] < 10: continue
        cases.insert(-1, {'tokenizer': row['tokenizer'], 'domain': row['domain'], 'cohort': 'fresh', 'reason': 'additional distinct useful weak tokenizer'})
        used.add(row['tokenizer'])
    report = {'candidates': valid_candidates, 'screen': result, 'errors': errors, 'cases': cases}
    save_json('/out/candidates.json', valid_candidates); save_json('/out/selected-cases.json', cases)
    save_json('/out/screen.json', result); save_json('/out/screen-errors.json', errors)
    save_json('/out/screen-report.json', report); outputs.commit()
    return report


@app.function(image=cpu_image, volumes={'/train': training.with_mount_options(read_only=True), '/out': outputs}, timeout=1800)
def prepare_cohorts():
    import numpy as np
    from uniform_trajectory import CanonicalTokenizer, fresh_cohort, load_candidate
    from uniform_weakness import compression
    from collections import defaultdict
    canonical = CanonicalTokenizer('/train/input/model/tokenizer.json')
    candidates = {c['id']: c for c in json.loads(Path('/out/candidates.json').read_text())}
    cases = json.loads(Path('/out/selected-cases.json').read_text())
    historical = json.loads(Path('/out/aux/panels.json').read_text())
    old_report = json.loads(Path('/out/aux/report.json').read_text())
    excluded = defaultdict(set)
    for domain, cohorts in historical.items():
        for split in ['screen', 'confirm']: excluded[domain].update(r['id'] for r in cohorts[split])
    for row in json.loads(Path('/out/screen.json').read_text()): excluded[row['domain']].update(row['screen_ids'])
    for domain, samples in json.loads(Path('/out/profile-samples.json').read_text()).items():
        excluded[domain].update(r['id'] for r in samples)
    corpora = {}; frozen = []
    for case in cases:
        candidate = candidates[case['tokenizer']]; domain = case['domain']
        tokenizer, _, _, _, forbidden = load_candidate(candidate['path'], candidate['kind'])
        if case['cohort'] == 'historical':
            panel = historical[domain]['confirm']
            previous = next(r for r in old_report['confirmation'] if r['tokenizer'] == case['tokenizer'] and r['domain'] == domain)
            if [r['id'] for r in panel] != previous['document_ids']: raise ValueError('Historical document IDs changed')
            details = {'examples': len(panel), 'source_documents': len(panel), 'later_passages': 0,
                       'canonical_prefix_budget': 95, 'position_cap': 128, 'suffix_tokens': 32}
        else:
            if domain not in corpora:
                with gzip.open(f'/train/input/eval/{domain}.jsonl.gz', 'rt', encoding='utf-8') as stream:
                    corpora[domain] = [json.loads(line) for line in stream if line.strip()]
            panel, details = fresh_cohort(corpora[domain], domain, canonical, tokenizer, forbidden, excluded[domain])
        key = case['tokenizer'] + '--' + domain
        save_json(Path('/out/cohorts') / (key + '.json'), panel)
        mask = np.load(Path('/out/features') / case['tokenizer'] / 'truncated.npy')
        prefix_ids = [token for r in panel for token in tokenizer.encode(r['prefix'], add_special_tokens=False).ids]
        utility = compression(panel, tokenizer, canonical)
        info = {**case, 'key': key, 'candidate': candidate, 'context_sha256': digest(panel),
                'details': details, 'utility': utility,
                'continuation_tokens': 32 * len(panel), 'continuation_bytes': sum(r['continuation_bytes'] for r in panel),
                'prefix_tokens': {'canonical': sum(len(r['canonical_prefix_ids']) for r in panel), 'target': len(prefix_ids)},
                'truncated_prefix_token_fraction': float(mask[prefix_ids].mean()),
                'truncated_vocabulary_forms': int(mask.sum())}
        frozen.append(info)
    inventory = json.loads(Path('/out/inventory.json').read_text())
    steps = [c['manifest']['step'] for c in inventory['checkpoints']]
    if steps != list(range(500, 20001, 500)): raise ValueError('Full 40-checkpoint inventory required')
    exposure = json.loads(Path('/out/exposures.json').read_text())
    for case in frozen:
        if case['candidate']['kind'] == 'pool':
            case['training_exposures_at_20k'] = exposure['counts'][str(case['candidate']['config']['index'])]
    design = json.loads(Path('/out/design.json').read_text())
    spec = {'design': design, 'cases': frozen, 'checkpoint_steps': steps,
            'inventory_sha256': file_sha('/out/inventory.json'), 'source': 'uniform pool 1000, 20k optimizer updates'}
    spec['spec_sha256'] = digest(spec)
    save_json('/out/trajectory-spec.json', spec); outputs.commit()
    return spec


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
    elif action == 'benchmark':
        result = benchmark_wordpiece.remote(); save_json(STATE / 'wordpiece-runtime.json', result)
        print(json.dumps(result), flush=True)
    elif action == 'screen':
        status('screening saved and useful pool tokenizers at20k on four H200s', 100)
        pointer = STATE / 'screen-call.json'
        if pointer.exists():
            call = modal.FunctionCall.from_id(json.loads(pointer.read_text())['call_id'])
        else:
            call = screen_all.spawn(); save_json(pointer, {'call_id': call.object_id})
        while True:
            try: report = call.get(timeout=10); break
            except (TimeoutError, modal.exception.TimeoutError): pass
        save_json(STATE / 'screen-report.json', report)
        status('20k screening complete; cases frozen for full trajectories', 101,
               selected_cases=len(report['cases']), scored_tokenizers=len(report['candidates']), errors=report['errors'])
        print(json.dumps({'cases': report['cases'], 'errors': report['errors']}), flush=True)
    elif action == 'prepare':
        # Count optimizer updates once; a failed/replayed segment repeats identical draws.
        records = {}
        for path in sorted((ROOT / 'modal_results/anytok_pool_20k').glob('*-train.log')):
            for line in path.read_text(errors='replace').splitlines():
                try: row = json.loads(line[line.index('{'):])
                except (ValueError, TypeError): continue
                if row.get('stage') != 'train' or 'tokenizer_config_index' not in row: continue
                step, index = row['step'], row['tokenizer_config_index']
                if step in records and records[step] != index: raise ValueError('Replayed tokenizer draws differ')
                records[step] = index
        if set(records) != set(range(1, 20001)): raise ValueError('Complete 20k exposure history required')
        from collections import Counter
        counts = Counter(records.values())
        save_json(STATE / 'exposures.json', {'updates': 20000, 'counts': {str(i): counts[i] for i in range(1000)}})
        with outputs.batch_upload(force=True) as batch: batch.put_file(STATE / 'exposures.json', 'exposures.json')
        status('freezing larger fixed cohorts and tokenizer features', 101)
        spec = prepare_cohorts.remote(); save_json(STATE / 'trajectory-spec.json', spec)
        status('fixed cohorts prepared for every checkpoint', 101, cases=len(spec['cases']))
    else: raise ValueError('Unknown action')
