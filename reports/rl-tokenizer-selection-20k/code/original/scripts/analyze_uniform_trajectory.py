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


def load_verified(state=STATE):
    spec = json.loads((state / 'trajectory-spec.json').read_text())
    if digest({k: v for k, v in spec.items() if k != 'spec_sha256'}) != spec['spec_sha256']:
        raise ValueError('Specification checksum mismatch')
    results = {}
    for step in spec['checkpoint_steps']:
        path = state / 'checkpoints' / f'step-{step:06d}.json'
        marker = json.loads(path.with_suffix('.complete.json').read_text())
        row = json.loads(path.read_text())
        if (file_sha(path) != marker['sha256'] or row['step'] != step or marker['step'] != step
                or marker['spec_sha256'] != spec['spec_sha256']
                or set(row['cases']) != {c['key'] for c in spec['cases']}):
            raise ValueError(f'Checkpoint completion audit failed at {step}')
        for case in spec['cases']:
            r = row['cases'][case['key']]; n = len(case['document_ids'])
            if (len(r['canonical_nats']) != n or len(r['transferred_nats']) != n
                    or len(case['cluster_ids']) != n or len(case['byte_counts']) != n
                    or r['score']['continuation_tokens'] != 32 * n
                    or r['score']['continuation_bytes'] != sum(case['byte_counts'])):
                raise ValueError('Changed scored continuations')
            for array, field in [('canonical_nats', 'canonical_bpb'), ('transferred_nats', 'bpb')]:
                values = np.asarray(r[array])
                if (not np.isfinite(values).all() or (values < 0).any()
                        or not np.isclose(values.sum() / (sum(case['byte_counts']) * np.log(2)),
                                          r['score'][field], rtol=0, atol=1e-10)):
                    raise ValueError('Invalid or inconsistent checkpoint losses')
        results[step] = row
    return spec, results


def clean_sensitivity(spec, results, diagnostic):
    if diagnostic['spec_sha256'] != spec['spec_sha256']: raise ValueError('Wrong feature diagnostic')
    rows = {}
    for case in spec['cases']:
        d = diagnostic['cases'][case['key']]
        if d['context_sha256'] != case['context_sha256'] or d['document_ids'] != case['document_ids']:
            raise ValueError('Feature diagnostic cohort mismatch')
        keep = d['clean_indices']
        expected = [i for i, flags in enumerate(zip(d['has_truncated_prefix'], d['has_aliased_prefix']))
                    if not any(flags)]
        if keep != expected or len(keep) < 64: raise ValueError('Invalid or too small clean cohort')
        source = [np.asarray(results[s]['cases'][case['key']]['canonical_nats'])[keep] for s in spec['checkpoint_steps']]
        target = [np.asarray(results[s]['cases'][case['key']]['transferred_nats'])[keep] for s in spec['checkpoint_steps']]
        curves = curve_bootstrap(source, target, np.asarray(case['byte_counts'])[keep],
                                 [case['cluster_ids'][i] for i in keep])
        row = {'examples': len(keep), 'source_documents': d['clean_clusters'], 'slopes': {}}
        for name, (point, boot) in curves.items():
            row[name + '_bpb_20k'] = float(point[-1])
            row[name + '_ci95_20k'] = np.quantile(boot[-1], [.025, .975]).tolist()
            row['slopes'][name] = slope_summary(spec['checkpoint_steps'], point, boot, [15000, 20000])[0]
        rows[case['key']] = row
    return rows


def historical_audit(spec, results):
    folder = ROOT / 'modal_results/uniform_weakness'; audited = []
    for step in [12000, 18000, 20000]:
        path = folder / ('report.json' if step == 12000 else f'reruns/step-{step:06d}/report.json')
        previous = json.loads(path.read_text())
        for case in spec['cases'][:3]:
            old = next(r for r in previous['confirmation' if step == 12000 else 'rows']
                       if r['tokenizer'] == case['tokenizer'] and r['domain'] == case['domain'])
            scores = old if step == 12000 else old['current_scores']
            if scores['document_ids'] != case['document_ids'] or scores['continuation_bytes'] != case['byte_counts']:
                raise ValueError('Historical document or byte alignment changed')
            if step != 12000 and (old['context_sha256'] != case['context_sha256']
                                  or old['tokenizer_sha256'] != case['candidate']['sha256']):
                raise ValueError('Historical native tokenizer or context changed')
            row = results[step]['cases'][case['key']]
            denominator = sum(case['byte_counts']) * np.log(2)
            errors = {field: float(np.max(np.abs(np.asarray(row[field]) - scores[old_field])))
                      for field, old_field in [('canonical_nats', 'canonical_nats'), ('transferred_nats', 'nats')]}
            aggregate_errors = {k: abs(sum(row[k]) - sum(scores[v])) / denominator
                                for k, v in [('canonical_nats', 'canonical_nats'), ('transferred_nats', 'nats')]}
            if max(aggregate_errors.values()) > 1e-4: raise ValueError('Historical BPB failed reproduction')
            audited.append({'step': step, 'case': case['key'], 'previous_report_sha256': file_sha(path),
                            'max_per_passage_nats_difference': errors, 'aggregate_bpb_difference': aggregate_errors})
    return audited


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def findings(report):
    lines = ['# Uniform tokenizer transfer across all 40 checkpoints', '',
             'All 40 saved checkpoints (500–20,000, every 500 steps) were evaluated on 12 fixed tokenizer/domain cases: 480 evaluations. The three historical panels reproduce their separate 12k/18k/20k evaluations. Every checkpoint uses identical document IDs, prefix text, native prefix token IDs, canonical continuation IDs and continuation bytes within each case. Each continuation has 32 canonical tokens; BOS + prefix + continuation stays within 128 positions. Output and continuation input embeddings are the frozen canonical tables.', '',
             'Seven cases have a final transfer gap ≥0.1 BPB and ≥10% token savings. Four are additional cases on fresh scored passages, including two actual training-pool members. None establishes a practical transfer-gap plateau under the prespecified criterion: the entire 95% slope interval must fit inside ±0.002 BPB per 1k steps over 15k–20k (±0.01 BPB across that window). All seven weak cases still show improving absolute transferred BPB over that window.', '',
             '| Tokenizer / domain | 20k canonical BPB | 20k transferred BPB | Gap [95% CI] | Input tokens saved | Gap slope / 1k [95% CI], 15k–20k |',
             '|---|---:|---:|---|---:|---|']
    for row in report['cases']:
        c = row['case']; p = row['points'][-1]; s = row['slopes']['gap']['late']
        ci = p['gap_ci95']; lo, hi = s['ci95']
        lines.append(f"| {c['key']} | {p['canonical_bpb']:.4f} | {p['transferred_bpb']:.4f} | {p['gap_bpb']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] | {c['utility']['token_reduction_vs_canonical_percent']:.1f}% | {s['slope_bpb_per_1k']:+.5f} [{lo:+.5f}, {hi:+.5f}] |")
    lines += ['', 'The in-pool weak cases are Unigram 123k/regex (pool 822, seen 16 updates) on Japanese and BPE 115k/line256 (pool 616, seen 26 updates) on Chinese. Their final gaps are about +0.140 and +0.152 BPB. Pool 616 has only 4.1836% merge noise, so this example does not require extreme noise. Both retain useful lossless compression. The other new weak cases are Unigram 150k/Japanese and Qwen2.5-Coder 151,665/Chinese; both vocabularies are outside the training pool’s 32k–128k range, so they are not evidence of an in-range sampling failure.', '',
              'The two roughly +0.46 BPB medical screening gaps (pool 416 WordPiece and pool 635 BPE) did not replicate on the later scored passages: final gaps are only about +0.032 and +0.024. The 150k WordPiece medical case also fails to confirm a weakness. These negative results are retained. The original Base64k medical panel is a different early-passage cohort and remains weak.', '',
              'Japanese35, Base64k medical, pool822 and 150k Unigram have slope intervals inside the looser ±0.005 BPB/1k band. This supports slow improvement at that tolerance, not the stricter plateau claim. Shorter 17.5k–20k and 18k–20k fits are exploratory and can be noisy; they do not replace the primary window. Base64k medical even has a positive 17.5k–20k gap slope, illustrating sensitivity to the chosen endpoint.', '',
              'The original three cases use 96 documents each and canonical prefix budgets up to 95 tokens. New cases use 256 passages and canonical prefix budgets up to 64. Chinese/Japanese/English fresh cohorts use distinct unseen documents. Medical fresh cohorts use later, nonoverlapping passages from 102 source articles; article IDs overlap screening, so they are not an independent-article confirmation. Article-cluster bootstrap preserves this dependence. Different cases are not averaged into an overall metric.', '',
              'Uncertainty uses 5,000 paired bootstrap draws of original source-document clusters, shared across all checkpoints and canonical/transferred scores. Raw worker quick confidence intervals are discarded. These intervals describe text variation for a single training seed; checkpoint fits may oscillate, intervals are per case, and no population or multiple-comparison claim is made. Endpoint-selected cases can have selection bias. Exact pool configurations, tokenizer hashes, domain mixtures, exposure counts and utility measurements are saved in trajectory-spec.json.', '',
              'Removing every passage whose prefix contains a truncated or aliased hypernetwork feature leaves the main residual gaps largely unchanged. Pool 616 and Qwen have no such prefix tokens in this panel at all. This is a sensitivity analysis, not a new primary cohort; it does not remove all possible model architecture limitations or establish that sampling caused the residual gap.', '',
              '| Case | Clean passages / articles | Clean 20k gap | Clean late gap slope [95% CI] |',
              '|---|---:|---:|---|']
    for row in report['cases']:
        key = row['case']['key']; clean = report['clean_sensitivity'][key]; s = clean['slopes']['gap']; lo, hi = s['ci95']
        lines.append(f"| {key} | {clean['examples']} / {clean['source_documents']} | {clean['gap_bpb_20k']:+.4f} | {s['slope_bpb_per_1k']:+.5f} [{lo:+.5f}, {hi:+.5f}] |")
    lines += ['', 'The data establish useful residual transfer weaknesses, including members of the sampled pool. They do not establish that uniform sampling has reached a strict wall by 20k. Most weak cases continue improving, and a comparison against a different sampling policy is needed to attribute gains to RL rather than more training.', '',
              'Three WordPiece/no-regex candidates were excluded before full-trajectory selection after a measured encoding-latency problem; all remaining 48 candidates completed the 9-domain screen without errors. Excluded runtime failures are not counted as high-loss successes.', '',
              'Figures: [absolute BPB](http://100.84.104.59:1337/plots/uniform_trajectory_bpb.png), [full transfer gaps](http://100.84.104.59:1337/plots/uniform_trajectory_gap.png), [late-window zoom](http://100.84.104.59:1337/plots/uniform_trajectory_late.png), [slope summary](http://100.84.104.59:1337/plots/uniform_trajectory_plateau.png). Each also has a PDF at the same path with a .pdf suffix.', '',
              'Reproduce: `OPENBLAS_NUM_THREADS=2 python scripts/analyze_uniform_trajectory.py`, then `python plotting_scripts/plot_uniform_trajectory.py`. Scores and cohorts persist in Modal profile gildmember, Volume anytok-uniform-trajectories; original weights remain in anytok-pool-20k. Only score metadata was collected locally. Publish derived reports/figures with `MODAL_PROFILE=gildmember modal run scripts/modal_uniform_trajectory.py --action publish`.', '']
    return '\n'.join(lines)


def main():
    spec, results = load_verified()
    report = analyze(spec, results)
    diagnostic = json.loads((STATE / 'feature-diagnostics.json').read_text())
    report['clean_sensitivity'] = clean_sensitivity(spec, results, diagnostic)
    report['historical_reproduction'] = historical_audit(spec, results)
    report['checkpoint_result_sha256'] = {str(s): file_sha(STATE / 'checkpoints' / f'step-{s:06d}.json') for s in results}
    save_json(STATE / 'analysis.json', report)
    rows = []; slopes = []
    for row in report['cases']:
        for p in row['points']:
            rows.append({'case': row['case']['key'], **{k: v for k, v in p.items() if not isinstance(v, list)},
                         **{k + suffix: v[i] for k, v in p.items() if isinstance(v, list)
                            for i, suffix in enumerate(['_lower', '_upper'])}})
        for metric, windows in row['slopes'].items():
            for window, s in windows.items():
                slopes.append({'case': row['case']['key'], 'metric': metric, 'window': window,
                               'slope_bpb_per_1k': s['slope_bpb_per_1k'], 'lower': s['ci95'][0], 'upper': s['ci95'][1],
                               'equivalent_within_0002': s['equivalent_to_flat_within_margin']['0.002']})
    write_csv(STATE / 'curves.csv', rows); write_csv(STATE / 'slopes.csv', slopes)
    (ROOT / 'plans/uniform-trajectory-findings.md').write_text(findings(report))
    print(json.dumps({'checkpoints': len(results), 'cases': len(report['cases']), 'evaluations': len(rows),
                      'strict_weak_plateaus': sum(r['primary_classification'] == 'practical transfer-gap plateau' for r in report['cases']),
                      'historical_audits': len(report['historical_reproduction'])}))


if __name__ == '__main__': main()
