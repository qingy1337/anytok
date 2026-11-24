"""Score the frozen 4,992-token held-out panel across all 1,000 native tokenizers."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

from natural_transfer import score_rows
from package_training_inputs import atomic_json, sha256
from pool_generalization import digest
from uniform_trajectory import PROFILE_DOMAINS, load_candidate


ROOT = Path(__file__).resolve().parent
INBOX = ROOT / 'inbox'
RUN = ROOT / 'run'
CODE = INBOX / 'code'
INPUT = Path(os.environ.get('ANYTOK_RIFT_INPUT', ROOT / 'input'))
POOL = INPUT / 'pool'
MODEL = INPUT / 'model'
PLAN = ROOT / 'plan.json'


def read(path):
    return json.loads(path.read_text())


def emit(worker, stage, completed=0, **fields):
    row = dict(at_utc=datetime.now(timezone.utc).isoformat(), stage=stage, completed=completed, **fields)
    atomic_json(RUN / 'status' / f'{worker}.json', row)
    if stage != 'scoring' or completed % 25 == 0:
        with (RUN / f'{worker}.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)


def setup(plan):
    import importlib.metadata

    if sha256(CODE / 'source-manifest.json') != plan['code_manifest_sha256']:
        raise ValueError('The original evaluator source snapshot changed')
    for file, expected in read(CODE / 'source-manifest.json').items():
        if sha256(CODE / file) != expected:
            raise ValueError(f'Training module changed: {file}')
    for name, version in plan['runtime'].items():
        if importlib.metadata.version(name).split('+')[0] != version:
            raise ValueError(f'Runtime mismatch for {name}')
    sys.path.insert(0, str(CODE))


def input_panel(plan):
    parent = read(INBOX / 'panel.json')
    if digest(parent) != plan['panel_sha256']:
        raise ValueError('The original evaluation passages changed')
    wanted = set(plan['documents'])
    rows = [row for row in parent if row['id'] in wanted]
    if (len(rows) != 156 or len(wanted) != len(rows) or len({r['cluster_id'] for r in rows}) != 156
            or any(r['split'] != 'screen' or not r['canonical_joint_encoding_matches']
                   or len(r['canonical_continuation_ids']) != 32 for r in rows)
            or sorted(sum(r['domain'] == d for r in rows) for d in PROFILE_DOMAINS) != [17] * 6 + [18] * 3):
        raise ValueError('Expected the exact 156 frozen passages across nine domains')
    if digest(continuation_proof(rows)) != plan['checks']['continuation_sha256']:
        raise ValueError('The frozen continuation tokens or bytes changed')
    return rows


def continuation_proof(rows):
    return [[r['id'], r['canonical_continuation_ids'], r['continuation_bytes']] for r in rows]


def pool_index(plan):
    path = POOL / 'full-results.json'
    if sha256(path) != plan['pool_index_sha256']:
        raise ValueError('Training pool index hash changed')
    candidates = sorted(read(path)['tokenizers'], key=lambda r: r['config']['index'])
    if [r['config']['index'] for r in candidates] != list(range(1000)):
        raise ValueError('Exactly 1,000 distinct pool tokenizers are required')
    return candidates


def native_inputs(task):
    candidate, rows = task
    index = candidate['config']['index']
    path = POOL / 'tokenizers' / f'{index:04d}' / 'tokenizer.json'
    if sha256(path) != candidate['tokenizer_sha256']:
        raise ValueError(f'Changed tokenizer {index}')
    tokenizer, _, _, _, forbidden = load_candidate(path, 'pool')
    prefixes = []
    issues = []
    for row in rows:
        ids = tokenizer.encode(row['prefix'], add_special_tokens=False).ids
        prefixes.append(ids)
        if not ids or len(ids) + 33 > 128:
            issues.append(dict(tokenizer=index, document=row['id'], cause='position budget', positions=len(ids) + 33))
        elif (forbidden | {0}).intersection(ids) or tokenizer.decode(ids, skip_special_tokens=False) != row['prefix']:
            issues.append(dict(tokenizer=index, document=row['id'], cause='invalid native encoding'))
    return index, prefixes, issues


def audit_panel():
    plan = read(PLAN)
    setup(plan)
    rows, candidates = input_panel(plan), pool_index(plan)
    encoded, issues = {}, []
    with ProcessPoolExecutor(max_workers=8) as executor:
        for count, (index, prefixes, failures) in enumerate(executor.map(native_inputs, ((c, rows) for c in candidates)), 1):
            encoded[str(index)] = prefixes
            issues.extend(failures)
            if count % 25 == 0:
                emit('panel', 'auditing original prefixes', audited_tokenizers=count, failures=len(issues))
        trimmed = []
        if issues:
            bad = {r['document'] for r in issues}
            changed = [i for i, r in enumerate(rows) if r['id'] in bad]
            replacements = trim_shared_prefixes(plan, [rows[i] for i in changed], candidates)
            for position, row in zip(changed, replacements):
                rows[position] = row
                trimmed.append(dict(document=row['id'], removed_characters=row['additional_trim_chars'],
                                    prefix_sha256=row['prefix_sha256'], maximum_positions=row['maximum_positions']))
            for index, prefixes, failures in executor.map(native_inputs, ((c, replacements) for c in candidates)):
                if failures:
                    raise ValueError('A minimally trimmed shared prefix failed its independent recheck')
                for position, prefix in zip(changed, prefixes):
                    encoded[str(index)][position] = prefix
    if digest(continuation_proof(rows)) != plan['checks']['continuation_sha256']:
        raise ValueError('Rebuilding the shared panel changed a scored continuation')
    audit = dict(stage='verified', scored_tokens=156 * 32,
                 scored_bytes=sum(r['continuation_bytes'] for r in rows), documents=len(rows),
                 original_prefix_failures=issues, trimmed_prefixes=trimmed,
                 panel_sha256=digest(rows), native_prefixes_sha256=digest(encoded),
                 plan_sha256=digest(plan))
    atomic_json(RUN / 'scoring-panel.json', rows)
    atomic_json(RUN / 'native-prefixes.json', encoded)
    atomic_json(RUN / 'panel-audit.json', audit)
    emit('panel', 'prefixes verified', audited_tokenizers=len(encoded), trimmed_prefixes=len(trimmed))


def trim_shared_prefixes(plan, rows, candidates):
    from transformers import AutoTokenizer
    from four_way_8k import common_panel

    emit('panel', 'checking minimally trimmed prefixes', audited_tokenizers=1000, affected_passages=len(rows))
    if sha256(MODEL / 'tokenizer.json') != plan['input_hashes']['model/tokenizer.json']:
        raise ValueError('Changed canonical tokenizer file')
    canonical = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    natives, forbidden = {}, {}
    for candidate in candidates:
        index = candidate['config']['index']
        file = POOL / 'tokenizers' / f'{index:04d}' / 'tokenizer.json'
        native, _, _, _, controls = load_candidate(file, 'pool')
        natives[str(index)] = native
        forbidden[str(index)] = controls | {0}
    return common_panel(rows, canonical, natives, forbidden)


class FrozenTokenizer:
    def __init__(self, rows, ids):
        self.prefixes = {r['prefix']: prefix for r, prefix in zip(rows, ids)}

    def encode(self, prefix, add_special_tokens=False):
        if add_special_tokens:
            raise ValueError('The old evaluator uses native prefixes without special tokens')
        return SimpleNamespace(ids=self.prefixes[prefix])


def load_models(plan, label):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from hf_hypernet.modeling_hypernet import ZettHypernet
    from hf_hypernet.smollm_setup import separate_input_embeddings_and_freeze_output
    from hf_hypernet.tokenizer_inputs import enable_byte_fallbacks

    torch.set_num_threads(1)
    required_gpu = os.environ.get('ANYTOK_POOL_GPU_NAME', 'L40S')
    if required_gpu not in torch.cuda.get_device_name():
        raise ValueError(f'Expected a {required_gpu} for this sweep')
    checkpoint = ROOT / 'checkpoints' / label
    manifest = read(checkpoint / 'ready.json')
    if manifest['step'] != int(label.rsplit('-', 1)[-1]):
        raise ValueError('Wrong checkpoint staged')
    expected = plan.get('checkpoint_hashes', {}).get(label)
    if expected and any(manifest['files'][name]['sha256'] != checksum for name, checksum in expected.items()):
        raise ValueError('Checkpoint differs from the earlier verified evaluation')
    canonical = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, dtype=torch.float32,
                                                low_cpu_mem_usage=True).to('cuda')
    separate_input_embeddings_and_freeze_output(model)
    original = model.get_input_embeddings().weight.detach().clone()
    weights = torch.load(checkpoint / 'lm.pt', map_location='cpu', weights_only=True, mmap=True)
    model.load_state_dict(weights)
    model.eval().requires_grad_(False)
    source = model.get_input_embeddings().weight.detach()
    if not torch.equal(source, original) or not torch.equal(model.get_output_embeddings().weight, original):
        raise ValueError('Canonical embedding tables changed')
    hn = ZettHypernet.from_pretrained(checkpoint / 'hypernet', local_files_only=True).to('cuda').eval().requires_grad_(False)
    fallbacks = enable_byte_fallbacks(hn, canonical)
    return model, source, hn, canonical, fallbacks


def features(canonical, fallbacks, candidate):
    import torch
    from hf_hypernet.tokenizer_inputs import build_surface_forms

    index = candidate['config']['index']
    path = POOL / 'tokenizers' / f'{index:04d}' / 'tokenizer.json'
    native, tokens, priors, specials, _ = load_candidate(path, 'pool')
    if len(tokens) != candidate['vocab_size_actual']:
        raise ValueError(f'Native vocabulary size changed for {index}')
    forms, _ = build_surface_forms(tokens, canonical, 10, fallbacks, specials)
    return native, torch.from_numpy(forms.astype(np.int64)).to('cuda'), torch.tensor(priors, dtype=torch.float32, device='cuda')


def prepare_one_feature(task):
    from transformers import AutoTokenizer
    from hf_hypernet.tokenizer_inputs import build_surface_forms, missing_byte_symbols

    candidate, fingerprint = task
    index = candidate['config']['index']
    output = RUN / 'features' / f'{index:04d}.npz'
    receipt = output.with_suffix('.json')
    expected = dict(tokenizer_sha256=candidate['tokenizer_sha256'], model_inputs=fingerprint)
    if receipt.is_file() and read(receipt) == expected | {'sha256': sha256(output)}:
        return index
    path = POOL / 'tokenizers' / f'{index:04d}' / 'tokenizer.json'
    if sha256(path) != candidate['tokenizer_sha256']:
        raise ValueError(f'Changed tokenizer {index}')
    canonical = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    fallbacks = {symbol: len(canonical) + 1 + i for i, symbol in enumerate(missing_byte_symbols(canonical))}
    _, tokens, priors, specials, _ = load_candidate(path, 'pool')
    forms, _ = build_surface_forms(tokens, canonical, 10, fallbacks, specials)
    if len(forms) != candidate['vocab_size_actual']:
        raise ValueError(f'Unexpected feature count for {index}')
    tmp = output.with_suffix('.partial')
    with tmp.open('wb') as stream:
        np.savez(stream, forms=forms, priors=np.asarray(priors, dtype=np.float32))
    tmp.replace(output)
    atomic_json(receipt, dict(expected, sha256=sha256(output)))
    return index


def prepare_features():
    plan = read(PLAN)
    setup(plan)
    fingerprint = digest([plan['input_hashes']['model/tokenizer.json'], plan['code_manifest_sha256']])
    if sha256(MODEL / 'tokenizer.json') != plan['input_hashes']['model/tokenizer.json']:
        raise ValueError('Changed canonical tokenizer file')
    (RUN / 'features').mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=8) as executor:
        for count, _ in enumerate(executor.map(prepare_one_feature, ((c, fingerprint) for c in pool_index(plan))), 1):
            if count % 25 == 0:
                emit('features', 'preparing reusable tokenizer inputs', features_prepared=count)
    atomic_json(RUN / 'features-ready.json', dict(features=1000, inputs=fingerprint))
    emit('features', 'tokenizer inputs ready', features_prepared=1000)


def load_features(candidate):
    import torch

    index = candidate['config']['index']
    file = RUN / 'features' / f'{index:04d}.npz'
    receipt = read(file.with_suffix('.json'))
    if receipt['tokenizer_sha256'] != candidate['tokenizer_sha256'] or receipt['sha256'] != sha256(file):
        raise ValueError(f'Unverified features for {index}')
    with np.load(file) as data:
        forms = torch.from_numpy(data['forms'].astype(np.int64)).to('cuda')
        priors = torch.from_numpy(data['priors']).to('cuda')
    return forms, priors


def generated_embeddings(hn, forms, priors, source):
    import torch

    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        predicted, out, _ = hn(forms, priors, source, None, deterministic=True)
    if out is not None or predicted.shape != (len(forms), source.shape[1]) or not torch.isfinite(predicted).all():
        raise ValueError('Invalid full-vocabulary embeddings')
    return predicted


def hardware_check(label='predictor-008000'):
    import torch
    from rift_eval_memory import apply_rift_memory_fix

    plan = read(PLAN)
    setup(plan)
    rows, candidates = input_panel(plan), pool_index(plan)
    for name, expected in plan['input_hashes'].items():
        if not name.startswith('model/'):
            continue
        file = INPUT / name
        if sha256(file) != expected:
            raise ValueError(f'Changed canonical model artifact {name}')
    emit('hardware', 'checking native and lower-memory forwards')
    model, source, hn, canonical, fallbacks = load_models(plan, label)
    small = candidates[plan['smoke_tokenizer']]
    large = max(candidates, key=lambda c: c['vocab_size_actual'])
    native, forms, priors = features(canonical, fallbacks, small)
    pristine = generated_embeddings(hn, forms, priors, source)
    original_scores = score_rows(model, pristine, source, rows, native)
    apply_rift_memory_fix(hn)
    optimized = generated_embeddings(hn, forms, priors, source)
    if not torch.equal(pristine, optimized) or original_scores != score_rows(model, optimized, source, rows, native):
        raise ValueError('Lower-memory scoring changed outputs on the same GPU')
    del pristine, optimized, forms, priors
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    _, forms, priors = features(canonical, fallbacks, large)
    generated = generated_embeddings(hn, forms, priors, source)
    torch.cuda.synchronize()
    result = dict(vocab_size=len(generated), smaller_vocab_bitwise_equal=True, full_vocab_forward_seconds=time.monotonic() - start,
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30, gpu=torch.cuda.get_device_name())
    atomic_json(RUN / 'hardware-check.json', result)
    emit('hardware', 'largest native vocabulary fits', **result)


def frozen_panel(plan):
    audit = read(RUN / 'panel-audit.json')
    if (audit['stage'] != 'verified' or audit['plan_sha256'] != digest(plan)
            or (audit['documents'], audit['scored_tokens'], audit['scored_bytes']) != (156, 4992, 14429)):
        raise ValueError('The exact held-out evaluation inputs must pass the all-pool audit')
    rows = read(RUN / 'scoring-panel.json')
    prefixes = read(RUN / 'native-prefixes.json')
    if (digest(rows) != audit['panel_sha256'] or digest(prefixes) != audit['native_prefixes_sha256']
            or digest(continuation_proof(rows)) != plan['checks']['continuation_sha256']):
        raise ValueError('The shared scoring panel changed after audit')
    return audit, rows, prefixes


def verify_launch(checkpoint_list):
    plan = read(PLAN)
    setup(plan)
    audit, _, _ = frozen_panel(plan)
    pool_index(plan)
    if not (RUN / 'features-ready.json').is_file():
        raise ValueError('Prepare all tokenizer features on the login node first')
    labels = checkpoint_list.read_text().splitlines()
    if (not labels or len(set(labels)) != len(labels) or any(label not in plan['checkpoints'] for label in labels)):
        raise ValueError('Expected one unique planned checkpoint label per line')
    for label in labels:
        folder = ROOT / 'checkpoints' / label
        staged = read(folder / 'ready.json')
        original = read(ROOT / 'resume-reference' / 'checkpoints' / label / 'ready.json')
        if staged['step'] != original['step']:
            raise ValueError(f'Wrong checkpoint staged for {label}')
        for name in ('lm.pt', 'hypernet/config.json', 'hypernet/model.safetensors'):
            path, expected = folder / name, original['files'][name]
            if (staged['files'][name] != expected or path.stat().st_size != expected['bytes']
                    or sha256(path) != expected['sha256']):
                raise ValueError(f'Changed checkpoint file for {label}/{name}')
    print(json.dumps(dict(protocol='heldout-pool-4992', plan_sha256=audit['plan_sha256'],
                          checkpoints=len(labels), tokenizers=1000, pairs=1000 * len(labels),
                          passages=audit['documents'], scored_tokens=audit['scored_tokens'],
                          scored_bytes=audit['scored_bytes'])), flush=True)


def score_checkpoint(worker, label, score_dir=None):
    import torch
    from rift_eval_memory import apply_rift_memory_fix

    plan = read(PLAN)
    setup(plan)
    audit, rows, prefixes = frozen_panel(plan)
    smoke = read(RUN / 'hardware-check.json')
    if (not smoke['smaller_vocab_bitwise_equal']
            or os.environ.get('ANYTOK_POOL_GPU_NAME', 'L40S') not in smoke['gpu']):
        raise ValueError('Run the memory check on the scoring GPU type first')
    candidates = pool_index(plan)
    output = (score_dir or RUN / 'scores') / label
    emit(worker, 'loading checkpoint', checkpoint=label)
    model, source, hn, _, _ = load_models(plan, label)
    apply_rift_memory_fix(hn)
    key = dict(plan_sha256=digest(plan), checkpoint=label)
    baseline = validate_matrix(score_rows(model, source, source, rows))
    canonical_bpb = balanced_bpb(baseline, rows)
    canonical_path = output / 'canonical.json'
    if canonical_path.is_file() and os.environ.get('ANYTOK_POOL_VERIFY_RESUME') == '1':
        saved = read(canonical_path)
        if any(saved.get(k) != v for k, v in key.items()) or not np.array_equal(saved['nats'], baseline):
            raise ValueError(f'Canonical per-passage losses changed on the resume GPU for {label}')
    else:
        atomic_json(canonical_path, dict(key, bpb=canonical_bpb, nats=baseline.tolist()))
    rankings = []
    for index, candidate in enumerate(candidates):
        path = output / f'{index:04d}.json'
        if path.is_file():
            result = read(path)
            if any(result.get(k) != v for k, v in key.items()) or result['tokenizer_sha256'] != candidate['tokenizer_sha256']:
                raise ValueError('Existing scores have different inputs')
        else:
            forms, priors = load_features(candidate)
            generated = generated_embeddings(hn, forms, priors, source)
            tokenizer = FrozenTokenizer(rows, prefixes[str(index)])
            values = validate_matrix(score_rows(model, generated, source, rows, tokenizer))
            result = dict(key, index=index, tokenizer_sha256=candidate['tokenizer_sha256'],
                          nats=values.tolist(), bpb=balanced_bpb(values, rows),
                          gap_bpb=balanced_bpb(values - baseline, rows))
            atomic_json(path, result)
            del forms, priors, generated
            torch.cuda.empty_cache()
        config = candidate['config']
        rankings.append(dict(checkpoint=label, tokenizer=index, algorithm=config['algorithm'],
                             pretokenizer=config['pretokenizer'], vocab=candidate['vocab_size_actual'],
                             canonical_bpb=canonical_bpb, transferred_bpb=result['bpb'], gap_bpb=result['gap_bpb']))
        emit(worker, 'scoring', index + 1, checkpoint=label, peak_gib=torch.cuda.max_memory_allocated() / 2**30)
    ordered = sorted(rankings, key=lambda r: (-r['gap_bpb'], r['tokenizer']))
    temp = output / 'ranking.csv.partial'
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['rank', *ordered[0]])
        writer.writeheader()
        writer.writerows(dict(rank=i, **row) for i, row in enumerate(ordered, 1))
    temp.replace(output / 'ranking.csv')
    atomic_json(output / 'complete.json', dict(key, completed=1000, documents=156, scored_tokens=4992,
                                              scored_bytes=audit['scored_bytes']))
    emit(worker, 'checkpoint complete', 1000, checkpoint=label)


def validate_matrix(values):
    values = np.asarray(values)
    if values.shape != (156, 32) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Incomplete continuation scores')
    return values.sum(axis=1)


def balanced_bpb(values, rows):
    result = []
    for domain in PROFILE_DOMAINS:
        indices = [i for i, row in enumerate(rows) if row['domain'] == domain]
        sizes = sum(rows[i]['continuation_bytes'] for i in indices)
        result.append(values[indices].sum() / (sizes * math.log(2)))
    return float(np.mean(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['audit', 'features', 'smoke', 'score', 'verify-launch'])
    parser.add_argument('worker', nargs='?')
    parser.add_argument('checkpoint', nargs='?')
    parser.add_argument('--score-dir', type=Path)
    parser.add_argument('--smoke-checkpoint', default='predictor-008000')
    parser.add_argument('--checkpoint-list', type=Path)
    args = parser.parse_args()
    if args.action == 'verify-launch' and args.checkpoint_list is None:
        parser.error('--checkpoint-list is required for verify-launch')
    try:
        if args.action == 'audit':
            audit_panel()
        elif args.action == 'features':
            prepare_features()
        elif args.action == 'smoke':
            hardware_check(args.smoke_checkpoint)
        elif args.action == 'verify-launch':
            verify_launch(args.checkpoint_list)
        else:
            score_checkpoint(args.worker, args.checkpoint, args.score_dir)
    except Exception as error:
        worker = args.worker or {'audit': 'panel', 'features': 'features', 'smoke': 'hardware',
                                 'score': 'scorer', 'verify-launch': 'preflight'}[args.action]
        emit(worker, 'failed', error=str(error), checkpoint=args.checkpoint)
        raise


if __name__ == '__main__':
    main()
