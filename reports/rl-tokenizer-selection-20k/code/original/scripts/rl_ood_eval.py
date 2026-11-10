"""Paired RL-versus-uniform comparison on the existing frozen trajectory panel."""
import math
import numpy as np
from pool_generalization import digest
from analyze_uniform_trajectory import curve_bootstrap


def validate_result(spec, result):
    if digest({k:v for k,v in spec.items() if k!='spec_sha256'}) != spec['spec_sha256']:
        raise ValueError('Frozen specification changed')
    if result['step'] != 20000 or result['spec_sha256'] != spec['spec_sha256']:
        raise ValueError('Wrong checkpoint or frozen panel')
    if set(result['cases']) != {c['key'] for c in spec['cases']}:
        raise ValueError('Missing or extra cases')
    for case in spec['cases']:
        row = result['cases'][case['key']]
        if row['context_sha256'] != case['context_sha256']:
            raise ValueError('Scored context changed')
        count = len(case['document_ids'])
        if 'input_audit' in row:
            audit=row['input_audit']
            if (audit['document_ids']!=case['document_ids'] or audit['byte_counts']!=case['byte_counts']
                    or audit['tokenizer_sha256']!=case['candidate']['sha256']
                    or audit['suffix_tokens']!=32 or audit['position_cap']!=128):
                raise ValueError('Document IDs, tokenizer or continuations changed')
        if (row['score']['continuation_tokens'] != 32 * count
                or row['score']['continuation_bytes'] != sum(case['byte_counts'])):
            raise ValueError('Scored continuation counts differ')
        for key, score in [('canonical_nats', 'canonical_bpb'), ('transferred_nats', 'bpb')]:
            array = np.asarray(row[key], dtype=float)
            if array.shape != (count,) or not np.isfinite(array).all() or (array < 0).any():
                raise ValueError('Invalid document losses')
            if not math.isclose(array.sum() / (sum(case['byte_counts']) * math.log(2)),
                                row['score'][score], rel_tol=0, abs_tol=1e-10):
                raise ValueError('Aggregate differs from document scores')


def compare(spec, uniform, rl):
    validate_result(spec, uniform)
    validate_result(spec, rl)
    rows = []
    for case in spec['cases']:
        a, b = [r['cases'][case['key']] for r in [uniform, rl]]
        curves = curve_bootstrap([a['canonical_nats'], b['canonical_nats']],
                                 [a['transferred_nats'], b['transferred_nats']],
                                 case['byte_counts'], case['cluster_ids'])
        scores = {}
        for field, (point, boot) in curves.items():
            scores[field] = {'uniform': float(point[0]), 'rl': float(point[1]),
                             'delta_rl_minus_uniform': float(point[1] - point[0]),
                             'delta_ci95': np.quantile(boot[1] - boot[0], [.025, .975]).tolist()}
        weak = a['score']['gap_bpb'] >= .1 and case['utility']['token_reduction_vs_canonical_percent'] >= 10
        group = ('in_pool_weak' if case['candidate']['kind'] == 'pool' else 'ood_weak') if weak else 'control'
        rows.append({'key': case['key'], 'group': group, 'candidate': case['candidate'],
                     'domain': case['domain'], 'context_sha256': case['context_sha256'],
                     'documents': len(case['document_ids']), 'clusters': len(set(case['cluster_ids'])),
                     'continuation_tokens': len(case['document_ids']) * 32,
                     'continuation_bytes': sum(case['byte_counts']), 'scores': scores})
    return {'step': 20000, 'spec_sha256': spec['spec_sha256'], 'cases': rows,
            'uniform_result_digest': digest(uniform), 'rl_result_digest': digest(rl),
            'uncertainty': '5000 paired source-document-cluster bootstrap draws. One training seed per method; selected cases, per-case intervals, no population or multiple-comparison claim.',
            'position_cap': 128, 'suffix_tokens': 32}
