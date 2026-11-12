"""Common-context evaluation and ranking of the entire existing tokenizer pool."""
import math

from pool_generalization import digest
from uniform_trajectory import PROFILE_DOMAINS, fixed_document


def make_panel(sources, canonical, count=32):
    """Select by document hash, before looking at tokenizer or model losses."""
    import hashlib
    panel = {'screen': [], 'confirm': []}
    rejected = {}
    for domain in PROFILE_DOMAINS:
        rows = sources[domain]
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate source document IDs')
        ordered = sorted(rows, key=lambda r: hashlib.sha256(
            ('rl-all-pool-v1/' + domain + '/' + r['id']).encode()).digest())
        selected = []
        rejected[domain] = 0
        for raw in ordered:
            try:
                row = fixed_document(raw, canonical, canonical._tokenizer, {0})
            except ValueError:
                rejected[domain] += 1
                continue
            selected.append({k: row[k] for k in (
                'id', 'prefix', 'canonical_prefix_ids', 'canonical_continuation_ids',
                'continuation_bytes', 'cluster_id', 'cut_chars')})
            selected[-1]['domain'] = domain
            if len(selected) == 2 * count:
                break
        if len(selected) != 2 * count:
            raise ValueError(f'{domain}: insufficient independent documents')
        panel['screen'].extend(selected[:count])
        panel['confirm'].extend(selected[count:])
    panel['rejected'] = rejected
    return panel


def flat_panel(panel):
    return panel['screen'] + panel['confirm']


def fit_start(row, canonical, tokenizer, forbidden):
    """Find a valid tail without changing the cut or the scored continuation."""
    suffix = row['canonical_continuation_ids']
    if len(suffix) != 32:
        raise ValueError('Expected 32 canonical continuation tokens')
    for start in range(len(row['prefix'])):
        prefix = row['prefix'][start:]
        target = tokenizer.encode(prefix, add_special_tokens=False).ids
        source = canonical(prefix, add_special_tokens=False).input_ids
        if (source and target and not forbidden.intersection(target)
                and 0 not in source and max(len(source), len(target)) <= 95
                and tokenizer.decode(target, skip_special_tokens=False) == prefix
                and canonical.decode(source, clean_up_tokenization_spaces=False) == prefix):
            return start
    raise ValueError(f"No shared nonempty prefix for {row['domain']}/{row['id']}")


def trim_panel(panel, starts, canonical):
    import copy
    result = copy.deepcopy(panel)
    if len(starts) != len(flat_panel(result)):
        raise ValueError('Prefix constraint length mismatch')
    for row, start in zip(flat_panel(result), starts):
        if not isinstance(start, int) or not 0 <= start < len(row['prefix']):
            raise ValueError('Invalid common prefix constraint')
        row['prefix'] = row['prefix'][start:]
        row['canonical_prefix_ids'] = canonical(row['prefix'], add_special_tokens=False).input_ids
    return result


def summarize_candidate(index, config, panel, nats, canonical_nats):
    """Primary score gives each of the nine fixed domains equal weight."""
    if len(nats) != len(panel) or len(canonical_nats) != len(panel):
        raise ValueError('Missing per-document scores')
    if not all(math.isfinite(x) and x >= 0 for x in nats + canonical_nats):
        raise ValueError('Invalid per-document losses')
    domains = {}
    for domain in PROFILE_DOMAINS:
        selected = [i for i, r in enumerate(panel) if r['domain'] == domain]
        if not selected:
            raise ValueError('Missing domain')
        byte_count = sum(panel[i]['continuation_bytes'] for i in selected)
        if byte_count <= 0:
            raise ValueError('Invalid continuation bytes')
        denominator = byte_count * math.log(2)
        bpb = sum(nats[i] for i in selected) / denominator
        baseline = sum(canonical_nats[i] for i in selected) / denominator
        domains[domain] = {'bpb': bpb, 'canonical_bpb': baseline,
                           'gap_bpb': bpb - baseline, 'continuation_bytes': byte_count,
                           'documents': len(selected), 'continuation_tokens': 32 * len(selected)}
    mean = lambda key: sum(r[key] for r in domains.values()) / len(domains)
    denominator = sum(r['continuation_bytes'] for r in panel) * math.log(2)
    return {'index': index, 'config': config, 'domains': domains,
            'mean_domain_bpb': mean('bpb'), 'mean_domain_canonical_bpb': mean('canonical_bpb'),
            'mean_domain_gap_bpb': mean('gap_bpb'),
            'pooled_bpb': sum(nats) / denominator,
            'pooled_canonical_bpb': sum(canonical_nats) / denominator,
            'pooled_gap_bpb': (sum(nats) - sum(canonical_nats)) / denominator,
            'panel_sha256': digest(panel), 'transferred_nats': nats,
            'canonical_nats': canonical_nats}


def rank_results(rows, expected_indices):
    indices = [r['index'] for r in rows]
    if len(set(indices)) != len(indices) or set(indices) != set(expected_indices):
        raise ValueError('Incomplete or duplicate tokenizer results')
    if len({r['panel_sha256'] for r in rows}) != 1:
        raise ValueError('Tokenizers were scored on different text')
    if any(r['canonical_nats'] != rows[0]['canonical_nats'] for r in rows):
        raise ValueError('Canonical baseline differs across tokenizers')
    return sorted(rows, key=lambda r: (-r['mean_domain_bpb'], r['index']))
