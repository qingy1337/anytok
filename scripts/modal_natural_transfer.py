"""Frozen natural-text transfer map on the existing RL20k checkpoint."""
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import sys
import time

import modal
from pool_generalization import digest, file_sha, save_json

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'modal_results/natural_transfer'
PROFILE = 'blackpantherx86'
PREFIX = 'natural-transfer-v1'
DOMAINS = ['web-en', 'code', 'math-en', 'medical-en', 'legal-en']
app = modal.App('anytok-natural-transfer-map')
volume = modal.Volume.from_name('anytok-rl-all-pool-20k', version=2)
BASE = ROOT / 'runtime/anytok_pool_20k'
cpu_image = (modal.Image.debian_slim(python_version='3.12')
    .uv_pip_install('tokenizers==0.22.2', 'numpy==2.3.5')
    .env({'TOKENIZERS_PARALLELISM': 'false'})
    .add_local_python_source('natural_transfer', 'pool_generalization',
                             'uniform_trajectory', 'uniform_weakness'))
gpu_image = (modal.Image.from_registry('ubuntu:24.04', add_python='3.12')
    .apt_install('libpython3.12')
    .uv_pip_install('torch==2.15.0', 'transformers==5.15.1', 'tokenizers==0.22.2',
                    'numpy==2.3.5', 'safetensors==0.8.0', 'accelerate==1.14.0')
    .env({'TOKENIZERS_PARALLELISM': 'false'})
    .add_local_dir(BASE / 'hf_hypernet', '/opt/zett/hf_hypernet')
    .add_local_file(BASE / 'rust_utils/target/release/librust_utils.so',
                    '/opt/zett/rust_utils/target/release/librust_utils.so')
    .add_local_python_source('natural_transfer', 'pool_generalization',
                             'uniform_trajectory', 'uniform_weakness'))


def now():
    return datetime.now(timezone.utc).isoformat()


def status(stage, completed=0, total=0, **fields):
    row = dict(at_utc=now(), stage=stage, completed=completed, total=total,
               profile=PROFILE, **fields)
    save_json(STATE / 'status.json', row)
    with (STATE / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


@app.function(image=cpu_image, volumes={'/sweep': volume}, timeout=300, scaledown_window=2)
def inventory():
    volume.reload()
    root = Path('/sweep')
    spec = json.loads((root / 'spec.json').read_text())
    old = json.loads((root / 'panel.json').read_text())
    result = {}
    for domain in DOMAINS:
        path = root / f'input/eval/{domain}.jsonl.gz'
        if file_sha(path) != spec['input_hashes'][f'eval/{domain}.jsonl.gz']:
            raise ValueError('Source corpus changed')
        with gzip.open(path, 'rt') as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        excluded = {r['cluster_id'] for split in ['screen', 'confirm'] for r in old[split]
                    if r['domain'] == domain}
        result[domain] = {'documents': len(rows), 'fresh_documents': sum(r['id'] not in excluded for r in rows),
                          'text_bytes': sum(len(r['text'].encode()) for r in rows),
                          'min_chars': min(len(r['text']) for r in rows),
                          'max_chars': max(len(r['text']) for r in rows)}
    return result


@app.function(image=cpu_image, volumes={'/sweep': volume}, timeout=900, scaledown_window=2)
def prepare(plan):
    from natural_transfer import build_panel
    from uniform_trajectory import CanonicalTokenizer, load_candidate
    volume.reload()
    root = Path('/sweep')
    folder = root / PREFIX
    if (folder / 'preparation.json').exists():
        if json.loads((folder / 'plan.json').read_text()) != plan:
            raise ValueError('Cannot overwrite a frozen evaluation plan')
        old = json.loads((folder / 'preparation.json').read_text())
        if digest(json.loads((folder / 'panel.json').read_text())) != old['panel_sha256']:
            raise ValueError('Frozen panel changed')
        return old
    spec = json.loads((root / 'spec.json').read_text())
    canonical = CanonicalTokenizer(root / 'input/model/tokenizer.json')
    if file_sha(root / 'input/model/tokenizer.json') != spec['input_hashes']['model/tokenizer.json']:
        raise ValueError('Canonical tokenizer changed')
    tokenizers, forbidden = {}, {}
    for index in plan['indices']:
        path = root / f'input/pool/tokenizers/{index:04d}/tokenizer.json'
        if file_sha(path) != spec['tokenizers'][index]['sha256']:
            raise ValueError('Native tokenizer changed')
        native = load_candidate(path, 'pool')
        tokenizers[index], forbidden[index] = native[0], native[4] | {0}
    old = json.loads((root / 'panel.json').read_text())
    sources, excluded = {}, {}
    for domain in DOMAINS:
        path = root / f'input/eval/{domain}.jsonl.gz'
        if file_sha(path) != spec['input_hashes'][f'eval/{domain}.jsonl.gz']:
            raise ValueError('Heldout corpus changed')
        with gzip.open(path, 'rt') as stream:
            sources[domain] = [json.loads(line) for line in stream if line.strip()]
        excluded[domain] = {r['cluster_id'] for split in ['screen', 'confirm'] for r in old[split]
                            if r['domain'] == domain}
    rows, report = build_panel(sources, canonical, tokenizers, forbidden, excluded,
                                plan['documents_per_domain_per_split'])
    metadata = [{k: value for k, value in row.items() if k != 'prefix'} for row in rows]
    report.update(panel_sha256=digest(rows), rows=len(rows), metadata=metadata,
                  plan_sha256=digest(plan), source_spec_sha256=digest(spec))
    for split in ('screen', 'confirm'):
        selected = [r for r in rows if r['split'] == split]
        report[split] = dict(passages=len(selected), documents=len({r['cluster_id'] for r in selected}),
            continuation_tokens=len(selected) * 32,
            continuation_bytes=sum(r['continuation_bytes'] for r in selected),
            jointly_trimmed_prefixes=sum(r['prefix_trim_chars'] > 0 for r in selected),
            tags={tag: sum(r['tags'][tag] for r in selected) for tag in selected[0]['tags']})
    save_json(folder / 'plan.json', plan)
    save_json(folder / 'panel.json', rows)
    save_json(folder / 'preparation.json', report)
    volume.commit()
    return report


def score_rows(model, table, source, rows, tokenizer=None, empty_prefix=False):
    import torch
    import torch.nn.functional as F
    from hf_hypernet.joint_training import make_lm_microbatch
    scored = []
    with torch.inference_mode():
        for start in range(0, len(rows), 16):
            micro = rows[start:start + 16]
            examples = []
            for row in micro:
                ids = ([] if empty_prefix else row['canonical_prefix_ids'] if tokenizer is None else
                       tokenizer.encode(row['prefix'], add_special_tokens=False).ids)
                if 1 + len(ids) + len(row['canonical_continuation_ids']) > 128:
                    raise ValueError('Sequence exceeds128 positions')
                examples.append((ids, row['canonical_continuation_ids'], row['continuation_bytes']))
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
                raise ValueError('HF causal loss disagrees with per-token losses')
            scored.extend(nats[i][valid[i]].cpu().tolist() for i in range(len(micro)))
    return scored


@app.cls(image=gpu_image, gpu='B300', volumes={'/sweep': volume},
         max_containers=2, scaledown_window=2, timeout=900)
class Scorer:
    @modal.enter()
    def load(self):
        sys.path.insert(0, '/opt/zett')
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from hf_hypernet.modeling_hypernet import ZettHypernet
        from hf_hypernet.smollm_setup import separate_input_embeddings_and_freeze_output
        from hf_hypernet.tokenizer_inputs import enable_byte_fallbacks
        torch.set_num_threads(1)
        volume.reload()
        self.root = Path('/sweep')
        self.spec = json.loads((self.root / 'spec.json').read_text())
        for name in ['lm.pt', 'hypernet/config.json', 'hypernet/model.safetensors']:
            if file_sha(self.root / 'checkpoint' / name) != self.spec['checkpoint_manifest']['files'][name]['sha256']:
                raise ValueError('RL20k checkpoint changed')
        for name in ['model/tokenizer.json', 'model/model.safetensors', 'model/config.json']:
            if file_sha(self.root / 'input' / name) != self.spec['input_hashes'][name]:
                raise ValueError('Original model inputs changed')
        self.canonical = AutoTokenizer.from_pretrained(self.root / 'input/model', local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.root / 'input/model', local_files_only=True,
            dtype=torch.float32, low_cpu_mem_usage=True).to('cuda')
        separate_input_embeddings_and_freeze_output(self.model)
        original = self.model.get_input_embeddings().weight.detach().clone()
        self.model.load_state_dict(torch.load(self.root / 'checkpoint/lm.pt', map_location='cpu', weights_only=True))
        self.model.eval()
        self.source = self.model.get_input_embeddings().weight.detach()
        if not torch.equal(self.source, original) or not torch.equal(self.model.get_output_embeddings().weight, original):
            raise ValueError('Canonical tables changed')
        del original
        self.hypernet = ZettHypernet.from_pretrained(self.root / 'checkpoint/hypernet', local_files_only=True).to('cuda').eval()
        self.fallbacks = enable_byte_fallbacks(self.hypernet, self.canonical)
        self.cache = {}

    @modal.method()
    def score(self, index, plan, panel_sha256):
        import numpy as np
        import torch
        from uniform_trajectory import load_candidate
        from natural_transfer import validate_scores
        begin = time.monotonic()
        volume.reload()
        folder = self.root / PREFIX
        target = folder / 'results' / f'{index:04d}.json'
        if target.exists():
            old = json.loads(target.read_text())
            if old['plan_sha256'] != digest(plan) or old['panel_sha256'] != panel_sha256:
                raise ValueError('Existing scores belong to another evaluation')
            return old
        rows = json.loads((folder / 'panel.json').read_text())
        if digest(rows) != panel_sha256 or json.loads((folder / 'plan.json').read_text()) != plan:
            raise ValueError('Frozen natural panel changed')
        if panel_sha256 not in self.cache:
            self.cache[panel_sha256] = (score_rows(self.model, self.source, self.source, rows),
                score_rows(self.model, self.source, self.source, rows, empty_prefix=True))
        baseline, no_prefix = self.cache[panel_sha256]
        token_path = self.root / f'input/pool/tokenizers/{index:04d}/tokenizer.json'
        if file_sha(token_path) != self.spec['tokenizers'][index]['sha256']:
            raise ValueError('Native tokenizer changed')
        tokenizer, _, _, _, forbidden = load_candidate(token_path, 'pool')
        for row in rows:
            ids = tokenizer.encode(row['prefix'], add_special_tokens=False).ids
            if (len(ids) != row['native_prefix_lengths'][str(index)] or forbidden.intersection(ids)
                    or 0 in ids or tokenizer.decode(ids, skip_special_tokens=False) != row['prefix']):
                raise ValueError('Native prefix encoding changed')
        features = self.root / f'features/{index:04d}'
        manifest = json.loads((features / 'manifest.json').read_text())
        if self.fallbacks != manifest['fallbacks'] or manifest['tokenizer_sha256'] != self.spec['tokenizers'][index]['sha256']:
            raise ValueError('Feature mapping changed')
        for name in ['forms', 'priors']:
            if file_sha(features / (name + '.npy')) != manifest[name + '_sha256']:
                raise ValueError('Native features changed')
        forms = torch.from_numpy(np.load(features / 'forms.npy').astype(np.int64)).to('cuda')
        priors = torch.from_numpy(np.load(features / 'priors.npy')).to('cuda')
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            generated, output, _ = self.hypernet(forms, priors, self.source, None, deterministic=True)
        if output is not None or not torch.isfinite(generated).all():
            raise ValueError('Invalid full-vocabulary embedding output')
        transferred = score_rows(self.model, generated, self.source, rows, tokenizer)
        for values in [baseline, no_prefix, transferred]:
            validate_scores(rows, values)
        result = dict(index=index, config=self.spec['tokenizers'][index]['config'],
            tokenizer_sha256=self.spec['tokenizers'][index]['sha256'],
            panel_sha256=panel_sha256, plan_sha256=digest(plan), canonical=baseline,
            no_prefix=no_prefix, transferred=transferred, elapsed_seconds=time.monotonic() - begin,
            gpu=torch.cuda.get_device_name())
        save_json(target, result)
        volume.commit()
        del forms, priors, generated
        torch.cuda.empty_cache()
        print(json.dumps(dict(at_utc=now(), stage='scored natural text', index=index,
                              elapsed_seconds=result['elapsed_seconds'])), flush=True)
        return result


@app.function(image=cpu_image, volumes={'/sweep': volume}, timeout=300, scaledown_window=2)
def inspect_examples(requests, panel_sha256):
    from uniform_trajectory import CanonicalTokenizer
    volume.reload()
    root = Path('/sweep')
    panel = json.loads((root / PREFIX / 'panel.json').read_text())
    if digest(panel) != panel_sha256:
        raise ValueError('Frozen panel changed')
    selected = {r['id']: r for r in panel}
    canonical = CanonicalTokenizer(root / 'input/model/tokenizer.json')
    result = []
    for request in requests:
        row = selected[request['id']]
        result.append({**request, 'domain': row['domain'], 'split': row['split'],
            'source_document_id': row['source_document_id'],
            'absolute_cut_chars': row['absolute_cut_chars'], 'prefix': row['prefix'],
            'continuation': canonical.decode(row['canonical_continuation_ids'], clean_up_tokenization_spaces=False),
            'continuation_tokens': len(row['canonical_continuation_ids']),
            'continuation_bytes': row['continuation_bytes']})
    save_json(root / PREFIX / 'examples.json', result)
    volume.commit()
    return result


def drive(action):
    if os.environ.get('MODAL_PROFILE') != PROFILE:
        raise ValueError('Use ' + PROFILE)
    STATE.mkdir(parents=True, exist_ok=True)
    if action == 'inventory':
        status('checking held-out corpus inventory')
        result = inventory.remote()
        save_json(STATE / 'inventory.json', result)
        print(json.dumps(result, indent=2), flush=True)
        status('inventory complete')
    elif action == 'examples':
        report = json.loads((ROOT / 'reports/natural-transfer-20k/analysis.json').read_text())
        prepared = json.loads((STATE / 'preparation.json').read_text())
        save_json(STATE / 'examples.json', inspect_examples.remote(report['example_requests'], prepared['panel_sha256']))
    elif action in ['prepare', 'run']:
        plan = json.loads((ROOT / 'plans/natural-transfer-v1.json').read_text())
        save_json(STATE / 'plan.json', plan)
        status('preparing disjoint natural passages', total=len(plan['indices']))
        prepared = prepare.remote(plan)
        save_json(STATE / 'preparation.json', prepared)
        print(json.dumps({split: prepared[split] for split in ['screen', 'confirm']}), flush=True)
        if action == 'prepare':
            print(json.dumps({k: v for k, v in prepared.items() if k not in ['metadata', 'rejected']}), flush=True)
            status('prepared', total=len(plan['indices']))
            return
        from decimal import Decimal
        from modal_credits import billing_row
        credit = billing_row(PROFILE, 'this month', Decimal(30))
        save_json(STATE / 'credits-before.json', credit)
        if credit['status'] != 'ok' or Decimal(credit['estimated_remaining_usd']) < Decimal('4'):
            raise ValueError('Need at least$4 verified credits before the bounded evaluation')
        results = []
        scorer = Scorer()
        # Two initial settings provide an actual timing measurement before the remainder.
        pilot = [plan['representative_indices'][0], plan['representative_indices'][-1]]
        for group in [pilot, [i for i in plan['indices'] if i not in pilot]]:
            if results:
                measured = max(r['elapsed_seconds'] for r in results)
                estimate = (2 * 60 + measured * len(group)) * .001972 * 1.5
                save_json(STATE / 'cost-estimate.json', dict(remaining_upper_estimate_usd=estimate,
                    pilot_slowest_seconds=measured, rate_per_gpu_second=.001972))
                if estimate > float(credit['estimated_remaining_usd']) - 1:
                    raise ValueError('Measured conservative estimate exceeds remaining credits with$1 margin')
            status('scoring natural-text transfer gaps', len(results), len(plan['indices']))
            for result in scorer.score.map(group, [plan] * len(group), [prepared['panel_sha256']] * len(group), order_outputs=False):
                index = result['index']
                with gzip.open(STATE / f'{index:04d}.json.gz', 'wt') as stream:
                    json.dump(result, stream)
                results.append({k: result[k] for k in ['index', 'elapsed_seconds', 'gpu']})
                save_json(STATE / 'completed.json', results)
                status('scoring natural-text transfer gaps', len(results), len(plan['indices']),
                       latest_index=index)
        status('complete', len(results), len(plan['indices']))
        save_json(STATE / 'credits-after.json', billing_row(PROFILE, 'this month', Decimal(30)))
    else:
        raise ValueError('Unknown action')


@app.local_entrypoint()
def main(action: str = 'inventory'):
    try:
        drive(action)
    except Exception as exc:
        status('failed', error=str(exc))
        raise
