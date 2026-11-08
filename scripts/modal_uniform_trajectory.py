"""Profile reasonable pool tokenizers, then evaluate every saved uniform checkpoint."""
from datetime import datetime, timezone
from decimal import Decimal
import gzip
import hashlib
import json
import math
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
    value = {'at_utc': now(), 'stage': stage, 'completed': completed, 'total': 142,
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
    from uniform_trajectory import CanonicalTokenizer, fresh_cohort, load_candidate, feature_aliases
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
        feature_dir = Path('/out/features') / case['tokenizer']
        aliases = feature_aliases(np.load(feature_dir / 'forms.npy'), np.load(feature_dir / 'priors.npy'), forbidden, prefix_ids)
        info = {**case, 'key': key, 'candidate': candidate, 'context_sha256': digest(panel),
                'details': details, 'utility': utility,
                'continuation_tokens': 32 * len(panel), 'continuation_bytes': sum(r['continuation_bytes'] for r in panel),
                'prefix_tokens': {'canonical': sum(len(r['canonical_prefix_ids']) for r in panel), 'target': len(prefix_ids)},
                'document_ids': [r['id'] for r in panel],
                'cluster_ids': [r.get('cluster_id', r['id']) for r in panel],
                'byte_counts': [r['continuation_bytes'] for r in panel],
                'truncated_prefix_token_fraction': float(mask[prefix_ids].mean()),
                'truncated_vocabulary_forms': int(mask.sum()), **aliases}
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


def grid_inputs(canonical, spec):
    """Validate raw held-out text and native contexts before loading any checkpoint."""
    import numpy as np
    from uniform_trajectory import load_candidate
    from pool_ood_eval import examples_for
    raw_sources = {}; prepared = {}
    for case in spec['cases']:
        key = case['key']; domain = case['domain']; candidate = case['candidate']
        panel = json.loads((Path('/out/cohorts') / (key + '.json')).read_text())
        if digest(panel) != case['context_sha256']: raise ValueError('Fixed cohort changed')
        if file_sha(candidate['path']) != candidate['sha256']: raise ValueError('Tokenizer file changed')
        tokenizer, _, _, _, forbidden = load_candidate(candidate['path'], candidate['kind'])
        if domain not in raw_sources:
            with gzip.open(f'/train/input/eval/{domain}.jsonl.gz', 'rt', encoding='utf-8') as stream:
                raw_sources[domain] = {r['id']: r['text'] for r in map(json.loads, stream)}
        for row in panel:
            doc_id = row.get('cluster_id', row['id']); start = row.get('source_window_start', 0)
            if raw_sources[domain][doc_id][start:start + 4096] != row['compression_text']:
                raise ValueError('Cohort text does not match persistent held-out source')
            suffix = canonical.decode(row['canonical_continuation_ids'], clean_up_tokenization_spaces=False)
            if (canonical(row['prefix'], add_special_tokens=False).input_ids != row['canonical_prefix_ids']
                    or len(row['canonical_continuation_ids']) != 32
                    or len(suffix.encode('utf-8')) != row['continuation_bytes']
                    or not row['compression_text'][:row['cut_chars']].endswith(row['prefix'])
                    or not row['compression_text'][row['cut_chars']:].startswith(suffix)):
                raise ValueError('Source prefix/suffix IDs or bytes changed')
            ids = tokenizer.encode(row['prefix'], add_special_tokens=False).ids
            if forbidden.intersection(ids): raise ValueError('Control token appears in ordinary prefix')
        if [r['id'] for r in panel] != case['document_ids'] or [r['continuation_bytes'] for r in panel] != case['byte_counts']:
            raise ValueError('Document/byte order changed')
        folder = Path('/out/features') / candidate['id']
        features = json.loads((folder / 'manifest.json').read_text())
        if features['tokenizer_sha256'] != candidate['sha256'] or features['max_pieces'] != 10:
            raise ValueError('Feature encoding changed')
        for name, field in [('forms.npy', 'forms_sha256'), ('priors.npy', 'priors_sha256'), ('truncated.npy', 'truncated_sha256')]:
            if file_sha(folder / name) != features[field]: raise ValueError('Feature checksum changed')
        prepared[key] = {'canonical_examples': examples_for(panel), 'target_examples': examples_for(panel, tokenizer),
                         'forms': np.load(folder / 'forms.npy'), 'priors': np.load(folder / 'priors.npy'),
                         'fallbacks': features['fallbacks'], 'context_sha256': case['context_sha256']}
    return prepared


@app.function(image=gpu_image, gpu='H200', volumes={'/train': training.with_mount_options(read_only=True), '/out': outputs},
              max_containers=4, timeout=1800)
def evaluate_steps(steps):
    sys.path.insert(0, '/opt/zett')
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from hf_hypernet.tokenizer_inputs import enable_byte_fallbacks
    from pool_ood_eval import score_documents
    from uniform_weakness import compare_scores
    spec = json.loads(Path('/out/trajectory-spec.json').read_text())
    if digest({k: v for k, v in spec.items() if k != 'spec_sha256'}) != spec['spec_sha256']:
        raise ValueError('Frozen trajectory spec changed')
    if file_sha('/out/inventory.json') != spec['inventory_sha256']: raise ValueError('Checkpoint inventory changed')
    canonical = AutoTokenizer.from_pretrained('/train/input/model', local_files_only=True)
    prepared = grid_inputs(canonical, spec)
    shard = digest(steps)[:12]; started = time.monotonic(); done = []
    def emit(stage, **details):
        value = {'at_utc': now(), 'stage': stage, 'shard': shard, **details}
        save_json(Path('/out/workers') / (shard + '.json'), value)
        outputs.commit(); print(json.dumps(value), flush=True)
    for step in steps:
        path = Path('/out/checkpoints') / f'step-{step:06d}.json'
        marker = Path('/out/checkpoints') / f'step-{step:06d}.complete.json'
        if marker.exists() and path.exists():
            previous = json.loads(path.read_text()); completed = json.loads(marker.read_text())
            if previous['spec_sha256'] != spec['spec_sha256'] or file_sha(path) != completed['sha256']:
                raise ValueError('Existing checkpoint result has different inputs')
            done.append(step); continue
        if time.monotonic() - started > 1500: break  # Return safely before the hard timeout.
        emit('loading checkpoint', step=step, completed_steps=done)
        model, table, hypernet = load_scoring_model(step)
        fallbacks = enable_byte_fallbacks(hypernet, canonical)
        rows = {}; canonical_cache = {}
        for case in spec['cases']:
            key = case['key']; data = prepared[key]
            if fallbacks != data['fallbacks']: raise ValueError('Byte fallback mapping changed across checkpoints')
            emit('evaluating checkpoint', step=step, case=key, completed_cases=len(rows), total_cases=len(spec['cases']))
            signature = digest(data['canonical_examples'])
            if signature not in canonical_cache:
                canonical_cache[signature] = score_documents(model, data['canonical_examples'], table, table)
            forms = torch.from_numpy(data['forms'].astype(np.int64)).to('cuda')
            priors = torch.from_numpy(data['priors']).to('cuda')
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                generated, output, _ = hypernet(forms, priors, table, None, deterministic=True)
            if output is not None or not torch.isfinite(generated).all(): raise ValueError('Invalid generated embeddings')
            nats = score_documents(model, data['target_examples'], generated, table)
            source = canonical_cache[signature]
            score = compare_scores(nats, source, case['byte_counts'], replicates=1000)
            rows[key] = {'canonical_nats': source, 'transferred_nats': nats, 'score': score,
                         'context_sha256': case['context_sha256']}
            del forms, priors, generated; torch.cuda.empty_cache()
        result = {'step': step, 'spec_sha256': spec['spec_sha256'], 'cases': rows,
                  'at_utc': now(), 'platform': 'Modal H200', 'profile': 'gildmember'}
        if set(rows) != {c['key'] for c in spec['cases']}: raise ValueError('Incomplete checkpoint evaluation')
        save_json(path, result)
        save_json(marker, {'step': step, 'spec_sha256': spec['spec_sha256'], 'sha256': file_sha(path),
                           'cases': len(rows), 'at_utc': now()})
        outputs.commit(); done.append(step)
        del model, table, hypernet; torch.cuda.empty_cache()
        emit('checkpoint complete', step=step, completed_steps=done)
    return {'completed': done, 'remaining': [s for s in steps if s not in done]}


@app.function(image=cpu_image, volumes={'/out': outputs}, timeout=7200)
def evaluate_all():
    spec = json.loads(Path('/out/trajectory-spec.json').read_text())
    remaining = list(spec['checkpoint_steps'])
    for attempt in range(3):
        chunks = [remaining[i::4] for i in range(4) if remaining[i::4]]
        results = list(evaluate_steps.map(chunks, order_outputs=False))
        remaining = [s for r in results for s in r['remaining']]
        if not remaining: break
    if remaining: raise RuntimeError(f'Unevaluated checkpoints after bounded worker segments: {remaining}')
    return {'completed_steps': spec['checkpoint_steps'], 'spec_sha256': spec['spec_sha256']}


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
    elif action == 'grid':
        if float(budget['estimated_remaining_usd']) < 10:
            raise RuntimeError('Full grid requires a verified $10 credit margin')
        spec = json.loads((STATE / 'trajectory-spec.json').read_text())
        pointer = STATE / 'grid-call.json'
        if pointer.exists():
            saved = json.loads(pointer.read_text())
            if saved['spec_sha256'] != spec['spec_sha256']: raise ValueError('Existing grid call has different inputs')
            call = modal.FunctionCall.from_id(saved['call_id'])
        else:
            call = evaluate_all.spawn()
            save_json(pointer, {'call_id': call.object_id, 'spec_sha256': spec['spec_sha256']})
        checkpoint_dir = STATE / 'checkpoints'; checkpoint_dir.mkdir(exist_ok=True)
        collected = set()
        for step in spec['checkpoint_steps']:
            if (checkpoint_dir / f'step-{step:06d}.complete.json').exists(): collected.add(step)
        finished = False
        while True:
            if not finished:
                try:
                    completion = call.get(timeout=5); finished = True
                    if completion['spec_sha256'] != spec['spec_sha256']: raise ValueError('Wrong completed grid spec')
                except (TimeoutError, modal.exception.TimeoutError): pass
            try: entries = list(outputs.iterdir('checkpoints'))
            except FileNotFoundError: entries = []
            for entry in entries:
                if not entry.path.endswith('.complete.json'): continue
                step = int(Path(entry.path).name.split('.')[0].removeprefix('step-'))
                if step in collected: continue
                marker = json.loads(b''.join(outputs.read_file(entry.path)))
                raw = b''.join(outputs.read_file(f'checkpoints/step-{step:06d}.json'))
                result = json.loads(raw)
                if (hashlib.sha256(raw).hexdigest() != marker['sha256']
                        or result['spec_sha256'] != spec['spec_sha256'] or result['step'] != step
                        or set(result['cases']) != {c['key'] for c in spec['cases']}):
                    raise ValueError('Corrupt or mismatched grid result')
                for case in spec['cases']:
                    row = result['cases'][case['key']]
                    if (row['context_sha256'] != case['context_sha256']
                            or len(row['canonical_nats']) != len(case['document_ids'])
                            or len(row['transferred_nats']) != len(case['document_ids'])
                            or row['score']['continuation_tokens'] != case['continuation_tokens']
                            or row['score']['continuation_bytes'] != case['continuation_bytes']
                            or not all(math.isfinite(x) and x >= 0 for x in row['canonical_nats'] + row['transferred_nats'])):
                        raise ValueError('Grid document cohort changed')
                    denominator = sum(case['byte_counts']) * math.log(2)
                    if (not math.isclose(sum(row['canonical_nats']) / denominator, row['score']['canonical_bpb'], abs_tol=1e-10)
                            or not math.isclose(sum(row['transferred_nats']) / denominator, row['score']['bpb'], abs_tol=1e-10)):
                        raise ValueError('Grid aggregate loss does not match per-document scores')
                target = checkpoint_dir / f'step-{step:06d}.json'
                temporary = target.with_suffix('.tmp'); temporary.write_bytes(raw); temporary.replace(target)
                save_json(checkpoint_dir / f'step-{step:06d}.complete.json', marker)
                collected.add(step)
            status('evaluating every saved checkpoint' if not finished else 'collecting verified grid results',
                   101 + len(collected), checkpoints_completed=len(collected), checkpoints_total=40,
                   cases=len(spec['cases']), completed_steps=sorted(collected))
            if finished and collected == set(spec['checkpoint_steps']): break
            time.sleep(5)
        status('evaluations complete; trajectory analysis pending', 141, checkpoints_completed=40, cases=len(spec['cases']))
    else: raise ValueError('Unknown action')
