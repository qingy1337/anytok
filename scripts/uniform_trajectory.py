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
    return tokenizer, tokens, priors, specials, forbidden


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
