"""Analyze every saved checkpoint using paired document-cluster uncertainty."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from pool_generalization import digest, file_sha, save_json
from uniform_trajectory import SEED

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'modal_results/uniform_trajectories'


def curve_bootstrap(canonical, transferred, byte_counts, clusters, replicates=5000):
    counts = np.asarray(byte_counts, dtype=float)
    canonical = np.asarray(canonical, dtype=float); transferred = np.asarray(transferred, dtype=float)
    if (canonical.shape != transferred.shape or canonical.shape[1] != len(counts)
            or not np.isfinite(canonical).all() or not np.isfinite(transferred).all()):
        raise ValueError('Invalid per-document trajectory scores')
    unique = list(dict.fromkeys(clusters)); lookup = {c: i for i, c in enumerate(unique)}
    source_cluster = np.zeros((len(canonical), len(unique))); target_cluster = source_cluster.copy()
    bytes_cluster = np.zeros(len(unique))
    for i, label in enumerate(clusters):
        j = lookup[label]
        source_cluster[:, j] += canonical[:, i]; target_cluster[:, j] += transferred[:, i]
        bytes_cluster[j] += counts[i]
    rng = np.random.default_rng(SEED)
    draws = rng.integers(len(unique), size=(replicates, len(unique)))
    weights = np.zeros((replicates, len(unique)))
    np.add.at(weights, (np.arange(replicates)[:, None], draws), 1)
    denominator = weights @ bytes_cluster * np.log(2)
    source_boot = (source_cluster @ weights.T) / denominator
    target_boot = (target_cluster @ weights.T) / denominator
    total = counts.sum() * np.log(2)
    source = canonical.sum(1) / total; target = transferred.sum(1) / total
    return {'canonical': (source, source_boot), 'transferred': (target, target_boot),
            'gap': (target - source, target_boot - source_boot)}


def slope_summary(steps, point, boot, window):
    x = np.asarray(steps, dtype=float) / 1000
    mask = (np.asarray(steps) >= window[0]) & (np.asarray(steps) <= window[1])
    centered = x[mask] - x[mask].mean(); coefficients = centered / (centered @ centered)
    estimate = float(coefficients @ point[mask]); samples = coefficients @ boot[mask]
    interval = np.quantile(samples, [.025, .975]).tolist()
    margins = {str(m): bool(interval[0] > -m and interval[1] < m) for m in [.001, .002, .005]}
    return {'window': window, 'checkpoints': int(mask.sum()), 'slope_bpb_per_1k': estimate,
            'ci95': interval, 'equivalent_to_flat_within_margin': margins,
            'clear_improvement': interval[1] < 0, 'clear_worsening': interval[0] > 0}, samples


def classify(final_gap, utility, slope):
    if final_gap < .1 or utility < 10: return 'does not meet weakness/utility threshold'
    if slope['equivalent_to_flat_within_margin']['0.002']: return 'practical transfer-gap plateau'
    if slope['ci95'][1] < -.002: return 'continuing material transfer improvement'
    if slope['ci95'][0] > 0: return 'transfer gap worsening'
    if slope['ci95'][1] < 0: return 'continuing improvement; plateau not established'
    return 'late trend inconclusive'


def analyze(spec, results):
    steps = spec['checkpoint_steps']
    if sorted(results) != steps: raise ValueError('All saved checkpoints are required')
    rows = []
    for case in spec['cases']:
        key = case['key']; source = []; target = []
        for step in steps:
            result = results[step]
            if result['spec_sha256'] != spec['spec_sha256']: raise ValueError('Mixed trajectory specifications')
            row = result['cases'][key]
            if row['context_sha256'] != case['context_sha256']: raise ValueError('Changed context cohort')
            source.append(row['canonical_nats']); target.append(row['transferred_nats'])
        curves = curve_bootstrap(source, target, case['byte_counts'], case['cluster_ids'])
        points = []
        for i, step in enumerate(steps):
            point = {'step': step}
            for name, (values, boot) in curves.items():
                point[name + '_bpb'] = float(values[i])
                point[name + '_ci95'] = np.quantile(boot[i], [.025, .975]).tolist()
            points.append(point)
        windows = {}; samples = {}
        for name, (values, boot) in curves.items():
            windows[name] = {}; samples[name] = {}
            for label, limits in [('early', [10000, 15000]), ('late', [15000, 20000]),
                                  ('last_2500', [17500, 20000]), ('last_2000', [18000, 20000])]:
                windows[name][label], samples[name][label] = slope_summary(steps, values, boot, limits)
        change = samples['gap']['late'] - samples['gap']['early']
        deceleration = {'late_minus_early_slope': windows['gap']['late']['slope_bpb_per_1k'] - windows['gap']['early']['slope_bpb_per_1k'],
                        'ci95': np.quantile(change, [.025, .975]).tolist()}
        primary = classify(points[-1]['gap_bpb'], case['utility']['token_reduction_vs_canonical_percent'], windows['gap']['late'])
        rows.append({'case': case, 'points': points, 'slopes': windows,
                     'primary_classification': primary, 'gap_deceleration': deceleration,
                     'final_gap_lower_bound_above_01': points[-1]['gap_ci95'][0] >= .1})
    return {'at_utc': datetime.now(timezone.utc).isoformat(), 'spec_sha256': spec['spec_sha256'],
            'checkpoints': steps, 'checkpoint_count': len(steps), 'cases': rows,
            'primary_window': [15000, 20000], 'primary_flat_margin_bpb_per_1k': .002,
            'secondary_windows': [[17500, 20000], [18000, 20000]],
            'uncertainty': '5000 paired document-cluster bootstrap draws shared across all checkpoints and score types. Raw GPU quick CIs are not used. Intervals describe held-out text variation for one training seed.',
            'limits': 'A practical plateau is an equivalence result within the stated finite training window, not a proof of an irreversible wall, future convergence or RL effectiveness.'}
