"""Useful-tokenizer search: paired suffix loss, compression, independent confirmation."""
import hashlib
import json
import math
from pathlib import Path

from pool_generalization import shared_prefix

DOMAINS = ('web-en', 'code', 'web-zh', 'web-ja', 'web-ar', 'web-hi',
           'math-en', 'medical-en', 'legal-en')


def load_tokenizer(path):
    """Use native IDs and learned priors; support the existing byte-level interface."""
    import numpy as np
    from tokenizers import Tokenizer
    raw = json.loads(Path(path).read_text())
    tokenizer = Tokenizer.from_file(str(path)); vocab = tokenizer.get_vocab()
    if set(vocab.values()) != set(range(len(vocab))):
        raise ValueError('Noncontiguous native IDs')
    if raw.get('decoder', {}).get('type') != 'ByteLevel':
        raise ValueError('Requires a byte-level surface-form adapter; not a model failure')
    ordered = [None] * len(vocab)
    for token, index in vocab.items(): ordered[index] = token
    model = raw['model']
    priors = [0.] * len(vocab)
    if model['type'] == 'Unigram':
        for index, (_, score) in enumerate(model['vocab']): priors[index] = float(score)
    if not np.isfinite(priors).all(): raise ValueError('Non-finite tokenizer priors')
    specials = {r['content']: 0 for r in raw.get('added_tokens', []) if r.get('special')}
    special_ids = {vocab[s] for s in specials}
    # These rows are never used in an ordinary text prefix. Mapping to canonical
    # EOS avoids inventing surface forms for special strings such as DeepSeek's.
    return tokenizer, ordered, priors, specials, special_ids


def prepare_document(row, canonical, targets, special_ids):
    text = row['text'][:4096]
    midpoint = len(text) // 2
    for offset in range(64):
        cut = midpoint + offset
        if cut >= len(text): break
        raw_suffix = text[cut:]
        continuation = canonical(raw_suffix, add_special_tokens=False).input_ids[:32]
        if len(continuation) != 32: continue
        suffix = canonical.decode(continuation, clean_up_tokenization_spaces=False)
        if (not raw_suffix.startswith(suffix)
                or canonical(suffix, add_special_tokens=False).input_ids != continuation): continue
        raw_prefix = text[max(0, cut - 512):cut]
        try:
            prefix, canonical_ids, maximum = shared_prefix(raw_prefix, continuation, canonical, targets, 128)
        except ValueError:
            continue
        encodings = [t.encode(prefix, add_special_tokens=False).ids for t in targets]
        if any(set(ids).intersection(forbidden) for ids, forbidden in zip(encodings, special_ids)):
            continue
        return {'id': row['id'], 'prefix': prefix, 'canonical_prefix_ids': canonical_ids,
                'canonical_continuation_ids': continuation, 'continuation_bytes': len(suffix.encode('utf-8')),
                'maximum_positions': 1 + maximum + 32, 'cut_chars': cut,
                'compression_text': text}
    raise ValueError('No reversible 32-token suffix and shared capped prefix')


def make_panels(rows, domain, canonical, targets, special_ids, screen=48, confirm=96):
    """Choose only by fixed ID hash/encoding validity, before any model losses."""
    if len({r['id'] for r in rows}) != len(rows): raise ValueError('Duplicate source IDs')
    ordered = sorted(rows, key=lambda r: hashlib.sha256((domain + '/' + r['id']).encode()).digest())
    selected = []; rejected = []
    for row in ordered:
        try: selected.append(prepare_document(row, canonical, targets, special_ids))
        except ValueError as exc: rejected.append({'id': row['id'], 'reason': str(exc)})
        if len(selected) == screen + confirm: break
    if len(selected) < screen + 64:
        raise ValueError(f'{domain}: insufficient independent confirmation documents: {len(selected)}')
    return {'screen': selected[:screen], 'confirm': selected[screen:], 'rejected': rejected}


def compare_scores(nats, canonical_nats, byte_counts, replicates=5000):
    import numpy as np
    nats = np.asarray(nats, dtype=float); canonical = np.asarray(canonical_nats, dtype=float)
    byte_counts = np.asarray(byte_counts, dtype=float)
    if (nats.ndim != 1 or nats.shape != canonical.shape or nats.shape != byte_counts.shape
            or not np.isfinite(nats).all() or not np.isfinite(canonical).all()
            or not (byte_counts > 0).all()): raise ValueError('Invalid paired scores')
    denominator = byte_counts.sum() * math.log(2)
    delta = nats - canonical
    rng = np.random.default_rng(20261108)
    draws = rng.integers(len(nats), size=(replicates, len(nats)))
    boot = delta[draws].sum(1) / (byte_counts[draws].sum(1) * math.log(2))
    return {'bpb': float(nats.sum() / denominator),
            'canonical_bpb': float(canonical.sum() / denominator),
            'gap_bpb': float(delta.sum() / denominator),
            'gap_ci95': np.quantile(boot, [.025, .975]).tolist(),
            'documents': len(nats), 'continuation_tokens': 32 * len(nats),
            'continuation_bytes': int(byte_counts.sum())}


def compression(panel, tokenizer, canonical):
    source_tokens = target_tokens = total_bytes = 0
    for row in panel:
        text = row['compression_text']
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if tokenizer.decode(ids, skip_special_tokens=False) != text:
            raise ValueError('Compression sample fails lossless round-trip')
        target_tokens += len(ids)
        source_tokens += len(canonical(text, add_special_tokens=False).input_ids)
        total_bytes += len(text.encode('utf-8'))
    return {'canonical_tokens': source_tokens, 'target_tokens': target_tokens, 'text_bytes': total_bytes,
            'token_reduction_vs_canonical_percent': 100 * (1 - target_tokens / source_tokens)}


def classify(score, utility):
    """A shortlist entry must meet both the requested gap and a concrete use case."""
    weak = score['gap_bpb'] >= .1
    useful = utility['token_reduction_vs_canonical_percent'] >= 10
    return {'weak_at_requested_threshold': weak,
            'compression_benefit_at_least_10_percent': useful,
            'confirmed_weak_and_useful': weak and useful,
            'ci_entirely_above_threshold': score['gap_ci95'][0] >= .1}


def compare_checkpoints(previous, current):
    """Separate the paired change in transfer gap from the LM's own improvement."""
    import numpy as np
    if (previous['document_ids'] != current['document_ids']
            or previous['continuation_bytes'] != current['continuation_bytes']):
        raise ValueError('Checkpoint comparison changed document IDs/order or scored bytes')
    counts = np.asarray(current['continuation_bytes'], dtype=float)
    old = compare_scores(previous['nats'], previous['canonical_nats'], counts)
    new = compare_scores(current['nats'], current['canonical_nats'], counts)
    delta = ((np.asarray(current['nats']) - current['canonical_nats'])
             - (np.asarray(previous['nats']) - previous['canonical_nats']))
    denominator = counts.sum() * math.log(2)
    rng = np.random.default_rng(20261108)
    draws = rng.integers(len(counts), size=(5000, len(counts)))
    boot = delta[draws].sum(1) / (counts[draws].sum(1) * math.log(2))
    interval = np.quantile(boot, [.025, .975]).tolist()
    return {'previous': old, 'current': new,
            'canonical_bpb_change': new['canonical_bpb'] - old['canonical_bpb'],
            'transferred_bpb_change': new['bpb'] - old['bpb'],
            'gap_bpb_change': float(delta.sum() / denominator),
            'gap_change_ci95': interval,
            'gap_change_evidence': ('improved' if interval[1] < 0 else
                                    'worsened' if interval[0] > 0 else 'inconclusive')}
