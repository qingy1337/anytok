"""Shared-context rules and exact finite-pool cohort comparisons."""
import math
from pool_generalization import digest


def required_start(row, canonical, tokenizer, forbidden):
    """Find the smallest valid common-prefix trim, without changing targets."""
    suffix = canonical.decode(row['canonical_continuation_ids'], clean_up_tokenization_spaces=False)
    for start in range(len(row['prefix'])):
        prefix = row['prefix'][start:]
        native = tokenizer.encode(prefix, add_special_tokens=False).ids
        if not native or len(native) > 95 or forbidden.intersection(native):
            continue
        source = canonical(prefix, add_special_tokens=False).input_ids
        if not 16 <= len(source) <= 64 or 0 in source:
            continue
        if (tokenizer.decode(native, skip_special_tokens=False) == prefix
                and canonical.decode(source, clean_up_tokenization_spaces=False) == prefix
                and canonical(prefix + suffix, add_special_tokens=False).input_ids
                    == source + row['canonical_continuation_ids']):
            return start, len(native)
    raise ValueError('No shared lossless prefix with16 canonical tokens: ' + row['id'])


def shorten(rows, starts, canonical):
    if len(rows) != len(starts):
        raise ValueError('Prefix constraint count changed')
    result = []
    for row, start in zip(rows, starts):
        if not isinstance(start, int) or not 0 <= start < len(row['prefix']):
            raise ValueError('Invalid prefix constraint')
        prefix = row['prefix'][start:]
        ids = canonical(prefix, add_special_tokens=False).input_ids
        result.append({**row, 'prefix': prefix, 'canonical_prefix_ids': ids,
            'parent_context_sha256': row.get('parent_context_sha256', row['context_sha256']),
            'pool_trim_chars': row.get('pool_trim_chars', 0) + start,
            'context_sha256': digest([prefix, row['canonical_continuation_ids']])})
    return result


def cohort_values(gaps, indices, hard):
    """Equal tokenizer weights; retain hard members in the full-pool reference."""
    import numpy as np
    values = np.asarray(gaps, dtype=float)
    if values.shape != (len(indices),) or not np.isfinite(values).all():
        raise ValueError('Invalid tokenizer gaps')
    hard_set = set(hard)
    if not hard_set <= set(indices):
        raise ValueError('Missing hard-cohort member')
    mask = np.array([i in hard_set for i in indices])
    hard_mean, pool_mean = float(values[mask].mean()), float(values.mean())
    return dict(pool_tokenizers=len(indices), hard_tokenizers=int(mask.sum()),
        pool_mean_gap=pool_mean, pool_median_gap=float(np.median(values)),
        hard_mean_gap=hard_mean, hard_median_gap=float(np.median(values[mask])),
        hard_minus_pool=hard_mean - pool_mean,
        hard_mean_percentile=float((values < hard_mean).mean()),
        hard_individual_percentiles={str(i): float((values < v).mean()) for i, v in zip(indices, values) if i in hard_set})
