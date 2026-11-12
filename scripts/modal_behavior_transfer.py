"""Search for repeatable English information-retention failures on the RL20k model."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import modal
from pool_generalization import digest, file_sha, save_json
from behavior_transfer import choose_tokenizers

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'modal_results/behavior_transfer'
PROFILE = 'blackpantherx86'
NAME = 'anytok-rl-behavior-transfer'
PREFIX = 'behavior-transfer-v1'
app = modal.App(NAME)
volume = modal.Volume.from_name('anytok-rl-all-pool-20k', version=2)
BASE = ROOT / 'runtime/anytok_pool_20k'
cpu_image = (modal.Image.debian_slim(python_version='3.12')
    .uv_pip_install('tokenizers==0.22.2', 'numpy==2.3.5')
    .env({'TOKENIZERS_PARALLELISM': 'false'})
    .add_local_dir(BASE / 'hf_hypernet', '/opt/zett/hf_hypernet')
    .add_local_python_source('behavior_transfer', 'pool_generalization', 'uniform_trajectory', 'uniform_weakness'))
gpu_image = (modal.Image.from_registry('ubuntu:24.04', add_python='3.12')
    .apt_install('libpython3.12')
    .uv_pip_install('torch==2.15.0', 'transformers==5.15.1', 'tokenizers==0.22.2',
                    'numpy==2.3.5', 'safetensors==0.8.0', 'accelerate==1.14.0')
    .env({'TOKENIZERS_PARALLELISM': 'false'})
    .add_local_dir(BASE / 'hf_hypernet', '/opt/zett/hf_hypernet')
    .add_local_file(BASE / 'rust_utils/target/release/librust_utils.so',
                    '/opt/zett/rust_utils/target/release/librust_utils.so')
    .add_local_python_source('behavior_transfer', 'pool_generalization', 'uniform_trajectory', 'uniform_weakness'))


def now():
    return datetime.now(timezone.utc).isoformat()


@app.function(image=cpu_image, volumes={'/sweep': volume}, timeout=600, scaledown_window=2)
def prepare(plan):
    from behavior_transfer import synthetic_rows, validate_examples
    from uniform_trajectory import CanonicalTokenizer, load_candidate
    volume.reload()
    root = Path('/sweep')
    spec = json.loads((root / 'spec.json').read_text())
    canonical = CanonicalTokenizer(root / 'input/model/tokenizer.json')
    if file_sha(root / 'input/model/tokenizer.json') != spec['input_hashes']['model/tokenizer.json']:
        raise ValueError('Canonical tokenizer changed')
    tokenizers = {}
    for index in plan['indices']:
        path = root / f'input/pool/tokenizers/{index:04d}/tokenizer.json'
        if file_sha(path) != spec['tokenizers'][index]['sha256']:
            raise ValueError('Native tokenizer changed')
        tokenizers[index] = load_candidate(path, 'pool')[0]
    old = json.loads((root / 'panel.json').read_text())
    rows, rejected = [], []
    for split in ['screen', 'confirm']:
        generated = synthetic_rows(canonical, split, plan['per_family'])
        accepted, failures = validate_examples(generated, canonical, tokenizers)
        rows.extend(accepted)
        rejected.extend(failures)
        if digest(old[split]) != spec['panel_hashes'][split]:
            raise ValueError('Original text controls changed')
        for original in old[split]:
            if original['domain'] not in ['web-en', 'code', 'math-en', 'medical-en', 'legal-en']:
                continue
            suffix = canonical.decode(original['canonical_continuation_ids'], clean_up_tokenization_spaces=False)
            if '![](' in original['prefix'] + suffix:
                continue
            # Existing held-out continuations provide a natural-text reference.
            rows.append({**original, 'id': split + '/' + original['id'], 'split': split,
                'family': 'natural_text', 'size': 'natural', 'distance': 'natural',
                'answer_ids': original['canonical_continuation_ids'],
                'answer_bytes': original['continuation_bytes'], 'answer': suffix})
    rows, extra = validate_examples(rows, canonical, tokenizers)
    if extra:
        raise ValueError('An already accepted case changed during validation')
    counts = {}
    for row in rows:
        key = '/'.join([row['split'], row['family'], row['size'], row['distance']])
        counts[key] = counts.get(key, 0) + 1
    if any(v < 8 for k, v in counts.items() if '/natural_text/' not in k):
        raise ValueError('Too few shared valid cases in a behavior cell')
    folder = root / PREFIX
    save_json(folder / 'plan.json', plan)
    save_json(folder / 'panel.json', rows)
    save_json(folder / 'preparation.json', {'rows': len(rows), 'counts': counts,
        'panel_sha256': digest(rows), 'rejected': rejected})
    volume.commit()
    return {'rows': len(rows), 'counts': counts, 'panel_sha256': digest(rows), 'rejected': rejected,
        'metadata': [{k: r[k] for k in ['id', 'split', 'family', 'size', 'distance',
                     'answer_ids', 'answer_bytes', 'continuation_bytes']} for r in rows]}


@app.function(image=cpu_image, volumes={'/sweep': volume}, timeout=120, scaledown_window=2)
def inspect_case():
    sys.path.insert(0, '/opt/zett')
    import numpy as np
    from uniform_trajectory import CanonicalTokenizer, load_candidate
    from hf_hypernet.tokenizer_inputs import build_surface_forms
    volume.reload()
    root = Path('/sweep')
    rows = json.loads((root / PREFIX / 'panel.json').read_text())
    prepared = json.loads((root / PREFIX / 'preparation.json').read_text())
    if digest(rows) != prepared['panel_sha256']:
        raise ValueError('Frozen behavior panel changed')
    row = next(r for r in rows if r['id'] == 'confirm/numbers/000')
    plan = json.loads((root / PREFIX / 'plan.json').read_text())
    canonical = CanonicalTokenizer(root / 'input/model/tokenizer.json')
    audit = []
    for index in plan['indices']:
        tokenizer, tokens, _, specials, _ = load_candidate(root / f'input/pool/tokenizers/{index:04d}/tokenizer.json', 'pool')
        ids = tokenizer.encode(row['prefix'], add_special_tokens=False).ids
        if tokenizer.decode(ids, skip_special_tokens=False) != row['prefix']:
            raise ValueError('Example prefix changed')
        folder = root / f'features/{index:04d}'
        manifest = json.loads((folder / 'manifest.json').read_text())
        forms, _ = build_surface_forms([tokens[i] for i in ids], canonical, 11, manifest['fallbacks'], specials)
        cached = np.load(folder / 'forms.npy')
        if not np.array_equal(cached[ids], forms[:, :10]):
            raise ValueError('Stored prefix features differ from independent construction')
        audit.append({'index': index, 'prefix_tokens': len(ids),
                      'truncated_prefix_tokens': int((forms[:, 10] != len(canonical)).sum())})
    result = {'example': row, 'feature_audit': audit, 'panel_sha256': prepared['panel_sha256']}
    save_json(root / PREFIX / 'example-audit.json', result);volume.commit()
    return result


def score_rows(model, table, source, rows, tokenizer=None):
    import torch
    import torch.nn.functional as F
    from hf_hypernet.joint_training import make_lm_microbatch
    result = []
    with torch.inference_mode():
        for start in range(0, len(rows), 16):
            micro = rows[start:start + 16]
            examples = [(r['canonical_prefix_ids'] if tokenizer is None else
                tokenizer.encode(r['prefix'], add_special_tokens=False).ids,
                r['canonical_continuation_ids'], r['continuation_bytes']) for r in micro]
            embeds, mask, labels, positions = make_lm_microbatch(examples, table, source, 'cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = model(inputs_embeds=embeds, attention_mask=mask, position_ids=positions,
                               use_cache=False, labels=labels)
            gold = labels[:, 1:]
            logits = output.logits[:, :-1].float()
            nats = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), gold.reshape(-1),
                                   ignore_index=-100, reduction='none').reshape(gold.shape)
            valid = gold != -100
            if valid.sum(1).tolist() != [32] * len(micro):
                raise ValueError('Scored token count changed')
            if abs(float(nats[valid].mean()) - float(output.loss)) > 1e-5:
                raise ValueError('Built-in causal LM loss disagrees with token losses')
            predicted = logits.argmax(-1)
            for j in range(len(micro)):
                result.append({'nats': nats[j][valid[j]].cpu().tolist(),
                    'correct': (predicted[j][valid[j]] == gold[j][valid[j]]).cpu().tolist()})
    return result


def greedy_answer(model, table, source, row, tokenizer, canonical):
    import torch
    prefix = tokenizer.encode(row['prefix'], add_special_tokens=False).ids
    values = torch.cat([source[0:1], table[torch.tensor(prefix, device='cuda')]]).to(torch.bfloat16)[None]
    output_ids = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        output = model(inputs_embeds=values, use_cache=True)
        for _ in row['answer_ids']:
            index = int(output.logits[0, -1].argmax())
            output_ids.append(index)
            output = model(inputs_embeds=source[index:index + 1].to(torch.bfloat16)[None],
                           past_key_values=output.past_key_values, use_cache=True)
    return {'ids': output_ids, 'text': canonical.decode(output_ids, clean_up_tokenization_spaces=False)}


def generate_rows(model, table, source, rows, canonical, tokenizer=None):
    import torch
    from behavior_transfer import answer_present
    results = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for start in range(0, len(rows), 16):
            micro = rows[start:start + 16]
            sequences = []
            for row in micro:
                ids = row['canonical_prefix_ids'] if tokenizer is None else tokenizer.encode(row['prefix'], add_special_tokens=False).ids
                if 1 + len(ids) + 32 > 128:
                    raise ValueError('Greedy generation would exceed128 positions')
                sequences.append(torch.cat([source[0:1], table[torch.tensor(ids, device='cuda')]]).to(torch.bfloat16))
            width = max(len(s) for s in sequences)
            inputs = torch.zeros((len(micro), width, source.shape[1]), device='cuda', dtype=torch.bfloat16)
            mask = torch.zeros((len(micro), width), device='cuda', dtype=torch.long)
            for j, sequence in enumerate(sequences):
                inputs[j, -len(sequence):] = sequence
                mask[j, -len(sequence):] = 1
            positions = (mask.cumsum(1) - 1).clamp(min=0)
            output = model(inputs_embeds=inputs, attention_mask=mask, position_ids=positions, use_cache=True)
            generated = []
            for step in range(32):
                ids = output.logits[:, -1].argmax(-1)
                generated.append(ids)
                if step != 31:
                    positions = mask.sum(1)[:, None]
                    mask = torch.cat([mask, torch.ones((len(micro), 1), device='cuda', dtype=torch.long)], dim=1)
                    output = model(inputs_embeds=source[ids].to(torch.bfloat16)[:, None], attention_mask=mask,
                        position_ids=positions, past_key_values=output.past_key_values, use_cache=True)
            for row, ids in zip(micro, torch.stack(generated, dim=1).cpu().tolist()):
                text = canonical.decode(ids, clean_up_tokenization_spaces=False)
                results.append({'id': row['id'], 'family': row['family'], 'split': row['split'],
                    'size': row['size'], 'distance': row['distance'], 'variant': row['variant'],
                    'answer': row['answer'], 'text': text, 'value_correct': answer_present(text, row)})
    return results


@app.function(image=gpu_image, gpu='B300', volumes={'/sweep': volume},
              max_containers=2, scaledown_window=2, timeout=1200)
def evaluate(indices, plan, panel_sha256, generation=False):
    sys.path.insert(0, '/opt/zett')
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from hf_hypernet.modeling_hypernet import ZettHypernet
    from hf_hypernet.smollm_setup import separate_input_embeddings_and_freeze_output
    from hf_hypernet.tokenizer_inputs import enable_byte_fallbacks
    from uniform_trajectory import load_candidate
    from behavior_transfer import summarize
    torch.set_num_threads(1)
    volume.reload()
    root = Path('/sweep')
    spec = json.loads((root / 'spec.json').read_text())
    folder = root / PREFIX
    rows = json.loads((folder / 'panel.json').read_text())
    if digest(rows) != panel_sha256 or json.loads((folder / 'plan.json').read_text()) != plan:
        raise ValueError('Frozen behavior panel changed')
    if generation:
        rows = [r for r in rows if r['family'] in ['numbers', 'names']]
    for name in ['lm.pt', 'hypernet/config.json', 'hypernet/model.safetensors']:
        if file_sha(root / 'checkpoint' / name) != spec['checkpoint_manifest']['files'][name]['sha256']:
            raise ValueError('RL20k checkpoint changed')
    canonical = AutoTokenizer.from_pretrained(root / 'input/model', local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(root / 'input/model', local_files_only=True,
        dtype=torch.float32, low_cpu_mem_usage=True).to('cuda')
    separate_input_embeddings_and_freeze_output(model)
    original_table = model.get_input_embeddings().weight.detach().clone()
    model.load_state_dict(torch.load(root / 'checkpoint/lm.pt', map_location='cpu', weights_only=True))
    model.eval()
    source = model.get_input_embeddings().weight.detach()
    if not torch.equal(source, original_table) or not torch.equal(model.get_output_embeddings().weight, source):
        raise ValueError('Canonical tables changed')
    del original_table
    hypernet = ZettHypernet.from_pretrained(root / 'checkpoint/hypernet', local_files_only=True).to('cuda').eval()
    fallbacks = enable_byte_fallbacks(hypernet, canonical)
    baseline = generate_rows(model, source, source, rows, canonical) if generation else score_rows(model, source, source, rows)
    results = []
    for index in indices:
        target_path = folder / ('generation' if generation else 'results') / f'{index:04d}.json'
        if target_path.exists():
            old = json.loads(target_path.read_text())
            if old['panel_sha256'] != panel_sha256:
                raise ValueError('Existing behavior result changed')
            results.append(old)
            continue
        token_path = root / f'input/pool/tokenizers/{index:04d}/tokenizer.json'
        if file_sha(token_path) != spec['tokenizers'][index]['sha256']:
            raise ValueError('Native tokenizer changed')
        tokenizer = load_candidate(token_path, 'pool')[0]
        features = root / f'features/{index:04d}'
        manifest = json.loads((features / 'manifest.json').read_text())
        if fallbacks != manifest['fallbacks']:
            raise ValueError('Fallback mapping changed')
        for name in ['forms', 'priors']:
            if file_sha(features / (name + '.npy')) != manifest[name + '_sha256']:
                raise ValueError('Native vocabulary features changed')
        forms = torch.from_numpy(np.load(features / 'forms.npy').astype(np.int64)).to('cuda')
        priors = torch.from_numpy(np.load(features / 'priors.npy')).to('cuda')
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            generated, output, _ = hypernet(forms, priors, source, None, deterministic=True)
        if output is not None or not torch.isfinite(generated).all():
            raise ValueError('Invalid full-vocabulary embedding output')
        if generation:
            scored = generate_rows(model, generated, source, rows, canonical, tokenizer)
            result = {'index': index, 'config': spec['tokenizers'][index]['config'],
                'panel_sha256': panel_sha256, 'canonical': baseline, 'transferred': scored,
                'generation_tokens': 32, 'gpu': torch.cuda.get_device_name()}
            save_json(target_path, result);volume.commit();results.append(result)
            print(json.dumps({'at_utc': now(), 'stage': 'checked greedy values', 'index': index,
                              'shard_completed': len(results)}), flush=True)
            del forms, priors, generated
            torch.cuda.empty_cache()
            continue
        scored = score_rows(model, generated, source, rows, tokenizer)
        failures = [j for j, row in enumerate(rows) if row['family'] != 'natural_text'
            and all(baseline[j]['correct'][:len(row['answer_ids'])])
            and not all(scored[j]['correct'][:len(row['answer_ids'])])]
        failures.sort(key=lambda j: -(sum(scored[j]['nats'][:len(rows[j]['answer_ids'])]) -
                                      sum(baseline[j]['nats'][:len(rows[j]['answer_ids'])])))
        examples = []
        for j in failures[:2]:
            examples.append({**rows[j], 'canonical': baseline[j], 'transferred': scored[j],
                'greedy_transferred_answer': greedy_answer(model, generated, source, rows[j], tokenizer, canonical)})
        result = {'index': index, 'config': spec['tokenizers'][index]['config'],
            'panel_sha256': panel_sha256, 'plan_sha256': digest(plan), 'gpu': torch.cuda.get_device_name(),
            'canonical': baseline, 'transferred': scored, 'examples': examples,
            'summaries': {'canonical': summarize(rows, baseline), 'transferred': summarize(rows, scored)}}
        save_json(target_path, result)
        volume.commit()
        results.append(result)
        print(json.dumps({'at_utc': now(), 'stage': 'scored behavior probe', 'index': index,
                          'completed': len(results), 'total': len(indices)}), flush=True)
        del forms, priors, generated
        torch.cuda.empty_cache()
    return results


@app.local_entrypoint()
def main(action: str = 'run'):
    if os.environ.get('MODAL_PROFILE') != PROFILE:
        raise ValueError('Use ' + PROFILE)
    STATE.mkdir(parents=True, exist_ok=True)
    if action == 'inspect':
        result = inspect_case.remote()
        save_json(STATE / 'example-audit.json', result)
        print(json.dumps(result), flush=True)
        return
    from decimal import Decimal
    from modal_credits import billing_row
    balance = billing_row(PROFILE, 'this month', Decimal(30))
    save_json(STATE / 'credits-before.json', balance)
    if balance['status'] != 'ok' or Decimal(balance['estimated_remaining_usd']) < Decimal('6'):
        raise ValueError('Insufficient verified credits for bounded two-worker probe')
    def status(stage, completed, total):
        row = {'at_utc': now(), 'stage': stage, 'completed': completed, 'total': total, 'profile': PROFILE}
        save_json(STATE / ('generation-status.json' if action == 'generate' else 'status.json'), row)
        print(json.dumps(row), flush=True)
    spec = json.loads((ROOT / 'reports/rl-pool-sweep-20k/spec.json').read_text())
    if action == 'generate':
        plan = json.loads((STATE / 'plan.json').read_text())
        prepared = json.loads((STATE / 'preparation.json').read_text())
        indices = plan['indices'];results = []
        status('checking actual greedy answers with flexible formatting', 0, len(indices))
        for batch in evaluate.map([indices[::2], indices[1::2]], [plan] * 2,
                [prepared['panel_sha256']] * 2, [True] * 2, order_outputs=False):
            results.extend(batch)
            save_json(STATE / 'generation.json', {'plan': plan, 'results': results})
            status('checking actual greedy answers with flexible formatting', len(results), len(indices))
        status('complete', len(results), len(indices))
        return
    if action != 'run':
        raise ValueError('Unknown action')
    plan = {'version': 1, 'indices': choose_tokenizers(spec), 'per_family': 64,
        'representative_indices': choose_tokenizers(spec, include_known=False),
        'families': ['numbers', 'identifiers', 'names', 'quoted_text', 'semantic_retrieval'],
        'primary': 'Answer-only BPB and exact greedy answer accuracy; separately report next32-token BPB.',
        'position_cap': 128, 'suffix_tokens': 32, 'seeds': [20261114, 20261115],
        'confirmation': 'Fresh payloads and different prompt wordings, same task families and length/distance cells.',
        'failure_criterion': 'Canonical exact accuracy>=80% and transferred accuracy lower by>=20 percentage points, with positive paired answer-BPB gap in both splits. Report support across representative tokenizer settings separately from the two previously weak tokenizers.'}
    save_json(STATE / 'plan.json', plan)
    status('preparing frozen behavior tasks', 0, len(plan['indices']))
    prepared = prepare.remote(plan)
    save_json(STATE / 'preparation.json', prepared)
    status('scoring information retention tasks', 0, len(plan['indices']))
    indices = plan['indices']
    groups = [indices[::2], indices[1::2]]
    results = []
    for batch in evaluate.map(groups, [plan] * 2, [prepared['panel_sha256']] * 2, order_outputs=False):
        results.extend(batch)
        save_json(STATE / 'results.json', {'plan': plan, 'preparation': prepared, 'results': results})
        status('scoring information retention tasks', len(results), len(indices))
    status('complete', len(results), len(indices))
