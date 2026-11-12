"""Recover selected high-loss passages from the frozen cloud evaluation panel."""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

import modal
from pool_generalization import digest, file_sha, save_json

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'modal_results/rl_pool_passages'
app = modal.App('anytok-rl-pool-passages')
PROFILE = 'blackpantherx86'
volume = modal.Volume.from_name('anytok-rl-all-pool-20k', version=2)
cpu_image = (modal.Image.debian_slim(python_version='3.12')
    .uv_pip_install('tokenizers==0.22.2', 'numpy==2.3.5')
    .env({'TOKENIZERS_PARALLELISM': 'false'})
    .add_local_python_source('pool_generalization', 'uniform_trajectory', 'uniform_weakness'))


@app.function(image=cpu_image, volumes={'/sweep': volume}, timeout=300,
              scaledown_window=2)
def inspect():
    from uniform_trajectory import CanonicalTokenizer, load_candidate
    volume.reload()
    root = Path('/sweep')
    read = lambda path: json.loads(path.read_text())
    spec = read(root / 'spec.json')
    panel = read(root / 'panel.json')
    if file_sha(root / 'input/model/tokenizer.json') != spec['input_hashes']['model/tokenizer.json']:
        raise ValueError('Canonical tokenizer changed')
    for index in [635, 3]:
        path = root / f'input/pool/tokenizers/{index:04d}/tokenizer.json'
        if file_sha(path) != spec['tokenizers'][index]['sha256']:
            raise ValueError('Native tokenizer changed')
    canonical = CanonicalTokenizer(root / 'input/model/tokenizer.json')
    tokenizers = {i: load_candidate(root / f'input/pool/tokenizers/{i:04d}/tokenizer.json', 'pool')[0]
                  for i in [635, 3]}
    results = {}
    examples = []
    for split, filename in [('screen', 'screen.json'), ('confirm', 'confirmation.json')]:
        if digest(panel[split]) != spec['panel_hashes'][split]:
            raise ValueError('Frozen raw-text panel changed')
        scored = read(root / filename)
        if scored['evaluation_sha256'] != digest(spec):
            raise ValueError('Frozen score specification changed')
        chosen = [r for r in scored['rows'] if r['index'] in tokenizers]
        rows = []
        for position, passage in enumerate(panel[split]):
            suffix = canonical.decode(passage['canonical_continuation_ids'], clean_up_tokenization_spaces=False)
            if (len(passage['canonical_continuation_ids']) != 32
                    or len(suffix.encode()) != passage['continuation_bytes']
                    or canonical(passage['prefix']).input_ids != passage['canonical_prefix_ids']
                    or passage['id'] != spec['panels'][split]['document_ids'][position]):
                raise ValueError('Saved passage/counts differ')
            domain = passage['domain']
            scores = {}
            for candidate in chosen:
                nats = candidate['transferred_nats'][position]
                base = candidate['canonical_nats'][position]
                denominator = passage['continuation_bytes'] * math.log(2)
                scores[str(candidate['index'])] = {
                    'canonical_bpb': base / denominator, 'transferred_bpb': nats / denominator,
                    'gap_bpb': (nats - base) / denominator,
                    'contribution_to_mean_gap_bpb': (nats - base) /
                        (candidate['domains'][domain]['continuation_bytes'] * math.log(2) * 9)}
            rows.append({'position': position, 'id': passage['id'], 'domain': domain,
                         'continuation_bytes': passage['continuation_bytes'], 'scores': scores})
        ranked = sorted(rows, key=lambda r: -r['scores']['635']['contribution_to_mean_gap_bpb'])
        # Largest aggregate contributors plus leading Chinese/Japanese/math
        # examples, using already saved losses rather than another text search.
        positions = {r['position'] for r in ranked[:10]}
        for domain in ['web-zh', 'web-ja', 'math-en']:
            positions.add(next(r['position'] for r in ranked if r['domain'] == domain))
        for position in sorted(positions):
            passage = panel[split][position]
            suffix = canonical.decode(passage['canonical_continuation_ids'], clean_up_tokenization_spaces=False)
            native = {}
            for index, tokenizer in tokenizers.items():
                ids = tokenizer.encode(passage['prefix'], add_special_tokens=False).ids
                if tokenizer.decode(ids, skip_special_tokens=False) != passage['prefix']:
                    raise ValueError('Native prefix does not round trip')
                native[str(index)] = {'ids': ids, 'tokens': [tokenizer.id_to_token(i) for i in ids]}
            examples.append({**rows[position], 'split': split, 'prefix': passage['prefix'],
                'continuation': suffix, 'canonical_continuation_ids': passage['canonical_continuation_ids'],
                'canonical_prefix_tokens': len(passage['canonical_prefix_ids']),
                'native_prefix': native})
        results[split] = {'rows': rows, 'domain_scores': {str(r['index']): r['domains'] for r in chosen},
                         'mean_gaps': {str(r['index']): r['mean_domain_gap_bpb'] for r in chosen}}
    result = {'evaluation_sha256': digest(spec), 'panel_hashes': spec['panel_hashes'],
              'examples': examples, 'scores': results,
              'selection': 'Top10 per-split contributors to tokenizer0635 mean-domain gap, plus top Chinese/Japanese/math contributors. Two tokenizer losses are paired on identical passages.'}
    save_json(root / 'passage-inspection/result.json', result)
    volume.commit()
    return result


@app.local_entrypoint()
def main():
    if os.environ.get('MODAL_PROFILE') != PROFILE:
        raise ValueError('Use ' + PROFILE)
    STATE.mkdir(parents=True, exist_ok=True)
    def status(stage, completed):
        value = {'at_utc': datetime.now(timezone.utc).isoformat(), 'stage': stage,
                 'completed': completed, 'total': 576, 'profile': PROFILE}
        save_json(STATE / 'status.json', value)
        print(json.dumps(value), flush=True)
    status('recovering exact scored passages', 0)
    result = inspect.remote()
    save_json(STATE / 'result.json', result)
    status('complete', 576)
    for split in ['screen', 'confirm']:
        rows = sorted([r for r in result['examples'] if r['split'] == split],
                      key=lambda r: -r['scores']['635']['contribution_to_mean_gap_bpb'])
        for row in rows[:3]:
            print(json.dumps({k: row[k] for k in ['split', 'id', 'domain', 'prefix', 'continuation', 'scores']}, ensure_ascii=False), flush=True)
