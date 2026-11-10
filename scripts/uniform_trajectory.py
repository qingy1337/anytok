"""Fixed-cohort tokenizer trajectories and paired late-training analysis."""
import hashlib
import json
from pathlib import Path

from pool_generalization import digest, file_sha, save_json

PROFILE_DOMAINS = ('web-en', 'code', 'web-zh', 'web-ja', 'web-ar', 'web-hi',
                   'math-en', 'medical-en', 'legal-en')
SEED = 20261108


def profile_samples(rows, domain, count=16):
    ordered = sorted(rows, key=lambda r: hashlib.sha256(('trajectory-profile/' + domain + '/' + r['id']).encode()).digest())
    selected = []
    for row in ordered:
        raw = row['text'][:4096]
        if len(raw) < 128: continue
        start = max(0, len(raw) // 2 - 1024)
        selected.append({'id': row['id'], 'text': raw[start:start + 2048]})
        if len(selected) == count: break
    if len(selected) != count: raise ValueError('Insufficient utility-profile documents')
    return selected


def tokenizer_utility(tokenizer, samples, canonical):
    source = target = byte_count = unknown = 0
    unk = tokenizer.token_to_id('<|unk|>')
    for row in samples:
        text = row['text']; ids = tokenizer.encode(text, add_special_tokens=False).ids
        if tokenizer.decode(ids, skip_special_tokens=False) != text:
            return {'valid': False, 'reason': 'not lossless'}
        if unk is not None: unknown += ids.count(unk)
        target += len(ids); source += len(canonical.encode(text, add_special_tokens=False).ids)
        byte_count += len(text.encode('utf-8'))
    return {'valid': unknown == 0, 'unknown_tokens': unknown,
            'target_tokens': target, 'canonical_tokens': source, 'text_bytes': byte_count,
            'token_reduction_percent': 100 * (1 - target / source)}


def select_pool_candidates(rows):
    """Choose by useful compression, never by model loss or a chosen noise value."""
    selected = {}
    for algorithm in ('bpe', 'wordpiece', 'unigram'):
        for pretokenizer in ('regex', 'line256', 'no-regex'):
            group = [r for r in rows if (r['config']['algorithm'], r['config']['pretokenizer']) == (algorithm, pretokenizer)]
            for domain in ('web-zh', 'web-ja', 'medical-en', 'code'):
                valid = [r for r in group if r['utility'][domain]['valid']
                         and r['utility'][domain]['token_reduction_percent'] >= 10]
                if not valid: continue
                best = max(valid, key=lambda r: (r['utility'][domain]['token_reduction_percent'], -r['index']))
                item = selected.setdefault(best['index'], {**best, 'selected_for': []})
                item['selected_for'].append(domain)
    return [selected[i] for i in sorted(selected)]


def load_candidate(path, kind):
    """Keep native IDs/priors; use the same EOS mapping as pool training."""
    from uniform_weakness import load_tokenizer
    tokenizer, tokens, priors, declared, forbidden = load_tokenizer(path)
    if kind == 'pool' or kind == 'saved-native':
        specials = {'<|endoftext|>': 0}
    else:
        # Published FIM/control strings may be unmarked in added_tokens.
        # They are excluded from ordinary-text prefixes, never scored as words.
        raw = json.loads(Path(path).read_text())
        specials = {r['content']: 0 for r in raw.get('added_tokens', [])}
        forbidden |= {r['id'] for r in raw.get('added_tokens', [])}
        for index, token in enumerate(tokens):
            if token.startswith('<｜') and token.endswith('｜>'):
                specials[token] = 0; forbidden.add(index)
    return tokenizer, tokens, priors, specials, forbidden


def fixed_document(row, canonical, tokenizer, forbidden):
    """Ordinary text, 64-token canonical prefix budget, 32 fixed suffix tokens."""
    from uniform_weakness import prepare_document
    from pool_generalization import shared_prefix
    prepared = prepare_document(row, canonical, [tokenizer], [forbidden])
    if 0 in prepared['canonical_continuation_ids']: raise ValueError('Canonical control token in suffix')
    raw = prepared['prefix']
    low, high = 0, len(raw) - 1
    while low < high:
        middle = (low + high) // 2
        if len(canonical(raw[middle:], add_special_tokens=False).input_ids) > 64: low = middle + 1
        else: high = middle
    prefix, source_ids, maximum = shared_prefix(raw[low:], prepared['canonical_continuation_ids'],
                                               canonical, [tokenizer], 128)
    if len(source_ids) < 16 or len(source_ids) > 64 or 0 in source_ids:
        raise ValueError('Insufficient ordinary source context')
    prepared.update(prefix=prefix, canonical_prefix_ids=source_ids,
                    maximum_positions=1 + maximum + 32,
                    cluster_id=row.get('cluster_id', row['id']),
                    source_window_start=row.get('source_window_start', 0))
    return prepared


def select_trajectory_cases(screen_rows, candidates):
    """Freeze diverse cases before looking at their checkpoint trajectories."""
    metadata = {c['id']: c for c in candidates}
    seeds = [('ood48-chinese35', 'web-zh'), ('ood48-japanese35', 'web-ja'), ('base64', 'medical-en')]
    selected = [{'tokenizer': name, 'domain': domain, 'cohort': 'historical', 'reason': 'previously reported reference'}
                for name, domain in seeds]
    used = {name for name, _ in seeds}
    weak = [r for r in screen_rows if r.get('score', {}).get('gap_bpb', -1) >= .08
            and r.get('utility', {}).get('token_reduction_vs_canonical_percent', -100) >= 10]
    def family(candidate):
        if candidate['kind'] == 'pool': return candidate['config']['algorithm']
        return candidate.get('settings', {}).get('algorithm', 'bpe').replace('byte-level ', '')
    tiers = [('pool', 'wordpiece', True), ('pool', 'unigram', True), ('pool', 'bpe', True),
             ('published-native', None, True), ('saved-native', 'wordpiece', False),
             ('saved-native', 'unigram', False), ('published-native', None, False),
             ('saved-native', 'bpe', True)]
    for kind, algorithm, in_range in tiers:
        options = [r for r in weak if r['tokenizer'] not in used
                   and metadata[r['tokenizer']]['kind'] == kind
                   and (algorithm is None or family(metadata[r['tokenizer']]) == algorithm)
                   and (32000 <= metadata[r['tokenizer']]['vocab_size'] <= 128000) == in_range]
        if not options: continue
        represented = {c['domain'] for c in selected if c['cohort'] == 'fresh'}
        best = max(options, key=lambda r: (r['domain'] not in represented, r['score']['gap_bpb'],
                                          r['utility']['token_reduction_vs_canonical_percent']))
        used.add(best['tokenizer'])
        selected.append({'tokenizer': best['tokenizer'], 'domain': best['domain'], 'cohort': 'fresh',
                         'reason': f'useful compression and screen gap>=.08; {kind}/{family(metadata[best["tokenizer"]])}/vocab_in_range={in_range}'})
    # Keep a low-gap ordinary-English control, without screening it into failure.
    selected.append({'tokenizer': 'ood48-base', 'domain': 'web-en', 'cohort': 'fresh', 'reason': 'ordinary English control'})
    return selected


class CanonicalTokenizer:
    """The pinned fast-tokenizer backend, without loading transformers on CPU."""
    def __init__(self, path):
        from tokenizers import Tokenizer
        self._tokenizer = Tokenizer.from_file(str(path))

    def __len__(self): return self._tokenizer.get_vocab_size()

    def __call__(self, text, add_special_tokens=False):
        from types import SimpleNamespace
        return SimpleNamespace(input_ids=self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids)

    def decode(self, ids, **kwargs):
        return self._tokenizer.decode(ids, skip_special_tokens=kwargs.get('skip_special_tokens', False))


def fresh_cohort(rows, domain, canonical, tokenizer, forbidden, excluded, count=256):
    """New documents when available; later nonoverlapping passages otherwise."""
    ordered = sorted(rows, key=lambda r: hashlib.sha256(('trajectory-confirm/' + domain + '/' + r['id']).encode()).digest())
    selected = []; rejected = 0; windows_by_doc = {}
    def append(raw, start):
        nonlocal rejected
        text = raw['text'][start:start + 4096]
        if len(text) < 256: return
        item = {'id': raw['id'] if start == 0 else raw['id'] + f'::chars-{start}',
                'cluster_id': raw['id'], 'source_window_start': start, 'text': text}
        try:
            prepared = fixed_document(item, canonical, tokenizer, forbidden)
            ids = tokenizer.encode(prepared['compression_text'], add_special_tokens=False).ids
            if tokenizer.decode(ids, skip_special_tokens=False) != prepared['compression_text']:
                raise ValueError('Utility excerpt changes under normalization')
            selected.append(prepared)
            windows_by_doc[raw['id']] = windows_by_doc.get(raw['id'], 0) + 1
        except ValueError: rejected += 1
    for raw in ordered:
        if raw['id'] not in excluded: append(raw, 0)
        if len(selected) == count: break
    for start in [4096, 8192, 12288, 16384]:
        if len(selected) == count: break
        for raw in ordered:
            if windows_by_doc.get(raw['id'], 0) < 4: append(raw, start)
            if len(selected) == count: break
    if len(selected) != count: raise ValueError(f'{domain}: only {len(selected)} usable fixed passages')
    if len({r['id'] for r in selected}) != count: raise ValueError('Duplicate passage IDs')
    return selected, {'examples': count, 'source_documents': len(windows_by_doc), 'rejected': rejected,
                      'later_passages': sum(r['source_window_start'] > 0 for r in selected),
                      'overlapping_screen_document_ids': len(set(windows_by_doc) & set(excluded)),
                      'canonical_prefix_budget': 64, 'position_cap': 128, 'suffix_tokens': 32}


def feature_aliases(forms, priors, forbidden, prefix_ids, return_mask=False):
    """Count indistinguishable HN inputs, including the native float32 prior."""
    import numpy as np
    forms = np.asarray(forms, dtype=np.int32)
    priors = np.asarray(priors, dtype=np.float32)
    keys = np.concatenate([forms.view(np.uint8).reshape(len(forms), -1),
                           priors.view(np.uint8).reshape(len(priors), -1)], axis=1)
    ordinary = np.array([i for i in range(len(forms)) if i not in forbidden])
    _, inverse, counts = np.unique(keys[ordinary], axis=0, return_inverse=True, return_counts=True)
    aliased = np.zeros(len(forms), dtype=bool); aliased[ordinary] = counts[inverse] > 1
    result = {'aliased_vocabulary_tokens': int(aliased.sum()),
              'aliased_prefix_token_fraction': float(aliased[prefix_ids].mean())}
    if return_mask: result['mask'] = aliased
    return result


def paired_slope(steps, nats_by_step, byte_counts, clusters=None, replicates=5000):
    """OLS slope in BPB/1k updates, with paired document-cluster bootstrap."""
    import numpy as np
    x = np.asarray(steps, dtype=float) / 1000
    nats = np.asarray(nats_by_step, dtype=float)
    counts = np.asarray(byte_counts, dtype=float)
    if (len(x) < 3 or nats.shape != (len(x), len(counts))
            or not np.isfinite(nats).all() or not (counts > 0).all()
            or not (np.diff(x) > 0).all()): raise ValueError('Invalid paired trajectory')
    centered = x - x.mean(); coefficients = centered / (centered @ centered)
    per_example_slope = coefficients @ nats
    denominator = counts.sum() * np.log(2)
    slope = float(per_example_slope.sum() / denominator)
    labels = list(range(len(counts))) if clusters is None else list(clusters)
    if len(labels) != len(counts): raise ValueError('Cluster labels differ from documents')
    unique = list(dict.fromkeys(labels)); indices = {label: i for i, label in enumerate(unique)}
    cluster_loss = np.zeros(len(unique)); cluster_bytes = np.zeros(len(unique))
    for i, label in enumerate(labels):
        cluster_loss[indices[label]] += per_example_slope[i]
        cluster_bytes[indices[label]] += counts[i]
    rng = np.random.default_rng(SEED)
    draws = rng.integers(len(unique), size=(replicates, len(unique)))
    bootstrap = cluster_loss[draws].sum(1) / (cluster_bytes[draws].sum(1) * np.log(2))
    interval = np.quantile(bootstrap, [.025, .975]).tolist()
    margins = {str(m): bool(interval[0] > -m and interval[1] < m) for m in [.001, .002, .005]}
    return {'slope_bpb_per_1k': slope, 'ci95': interval,
            'clusters': len(unique), 'examples': len(counts),
            'equivalent_to_flat_within_margin': margins,
            'rules_out_improvement_faster_than_002': interval[0] > -.002,
            'clear_improvement': interval[1] < 0,
            'clear_worsening': interval[0] > 0}


def grid_inputs(canonical, spec):
    """Validate raw held-out text and native contexts before loading any checkpoint."""
    import numpy as np
    import gzip
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

