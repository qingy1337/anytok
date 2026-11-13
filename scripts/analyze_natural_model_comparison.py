"""Paired trained-model comparison on the frozen natural-text stress setting."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np

from analyze_natural_transfer import bootstrap_balanced
from pool_generalization import digest, file_sha, save_json

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['zett20k', 'uniform20k', 'rl20k']
LABELS = {'zett20k': 'ZeTT 20k', 'uniform20k': 'Uniform pool 20k', 'rl20k': 'RL 20k'}
COLORS = {'zett20k': '#8657BC', 'uniform20k': '#2F8FD0', 'rl20k': '#36967B'}


def pack(state):
    plan = json.loads((state / 'plan.json').read_text())
    parent = ROOT / 'reports/natural-transfer-20k/results.json.gz'
    if file_sha(parent) != plan['rl_results_sha256']:
        raise ValueError('Original RL results changed')
    with gzip.open(parent, 'rt') as stream:
        rl = json.load(stream)
    if rl['preparation']['panel_sha256'] != plan['panel_sha256']:
        raise ValueError('Parent panel changed')
    models = {}
    for name in MODELS:
        if name == 'rl20k':
            raw = rl
        else:
            with gzip.open(state / (name + '.json.gz'), 'rt') as stream:
                raw = json.load(stream)
            if raw['plan_sha256'] != digest(plan) or raw['panel_sha256'] != plan['panel_sha256']:
                raise ValueError('Unpaired model result')
        ordered = sorted(raw['results'], key=lambda r: r['index'])
        if [r['index'] for r in ordered] != plan['indices']:
            raise ValueError('Missing tokenizer results')
        baseline = np.asarray(raw['canonical'], dtype=float)
        targets = np.asarray([r['transferred'] for r in ordered], dtype=float)
        n = len(rl['preparation']['metadata'])
        if baseline.shape != (n, 32) or targets.shape != (len(ordered), n, 32):
            raise ValueError('Missing continuation tokens')
        if (not np.isfinite(baseline).all() or not np.isfinite(targets).all()
                or (baseline < 0).any() or (targets < 0).any()):
            raise ValueError('Invalid prediction losses')
        models[name] = dict(canonical_nats=baseline.sum(1).tolist(), transferred_nats=targets.sum(2).tolist())
    pool = json.loads((ROOT / 'reports/rl-pool-sweep-20k/spec.json').read_text())['tokenizers']
    weak_pool = [r for r in pool if r['config']['pretokenizer'] in ['line256', 'no-regex']]
    pool_counts = dict(total=len(pool), weak_pretokenization=len(weak_pool),
        large_vocab_weak=sum(r['vocab_size'] >= plan['large_vocab_minimum'] for r in weak_pool))
    return dict(plan=plan, metadata=rl['preparation']['metadata'], models=models, pool_regime_counts=pool_counts,
        tokenizers=[dict(index=r['index'], config=r['config']) for r in sorted(rl['results'], key=lambda r: r['index'])],
        score_definition='Each number is the float64 sum of32 saved per-token natural-log losses. Full per-token scores remain in persistent cloud storage.')


def analyze(data):
    plan, rows = data['plan'], data['metadata']
    if digest(plan['context_selected_ids']) != plan['context_selected_ids_sha256']:
        raise ValueError('Frozen stress-subset membership changed')
    if [r['index'] for r in data['tokenizers']] != plan['indices']:
        raise ValueError('Tokenizer order changed')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate passages')
    for key in ['cluster_id', 'document_sha256']:
        sets = [{r[key] for r in rows if r['split'] == split} for split in ['screen', 'confirm']]
        if sets[0] & sets[1]:
            raise ValueError('Discovery/confirmation overlap')
    base = np.array([data['models'][name]['canonical_nats'] for name in MODELS])
    targets = np.array([data['models'][name]['transferred_nats'] for name in MODELS])
    if base.shape != (3, len(rows)) or targets.shape != (3, len(plan['indices']), len(rows)):
        raise ValueError('Missing model/document losses')
    if not np.isfinite(base).all() or not np.isfinite(targets).all() or (base < 0).any() or (targets < 0).any():
        raise ValueError('Invalid loss totals')
    selected = set(plan['context_selected_ids'])
    if not selected <= {r['id'] for r in rows}:
        raise ValueError('Missing selected passage')
    cohorts = {}
    representative = set(plan['representative_indices'])
    cohorts['representative36'] = [k for k, r in enumerate(data['tokenizers']) if r['index'] in representative]
    for label, pre in [('regex', ['regex']), ('line256', ['line256']), ('no-regex', ['no-regex']),
                       ('weak_pretokenization', ['line256', 'no-regex'])]:
        cohorts[label] = [k for k in cohorts['representative36'] if data['tokenizers'][k]['config']['pretokenizer'] in pre]
    cohorts['large_vocab_weak'] = [k for k in cohorts['weak_pretokenization']
        if data['tokenizers'][k]['config']['vocab_size'] >= plan['large_vocab_minimum']]
    regimes = dict(all=np.ones(len(rows), dtype=bool), context=np.array([r['id'] in selected for r in rows]))
    metrics, summaries, comparisons, sizes = [], [], [], []
    for regime, mask in regimes.items():
        for split in ['screen', 'confirm']:
            ids = np.flatnonzero(mask & np.array([r['split'] == split for r in rows]))
            subset = [rows[i] for i in ids]
            if not subset:
                raise ValueError('Empty regime')
            sizes.append(dict(regime=regime, split=split, passages=len(ids),
                documents=len({r['cluster_id'] for r in subset}), continuation_tokens=32 * len(ids),
                continuation_bytes=sum(r['continuation_bytes'] for r in subset),
                source_counts={domain: sum(r['domain'] == domain for r in subset)
                    for domain in ['web-en', 'code', 'math-en', 'medical-en', 'legal-en']}))
            n = len(plan['indices'])
            stack = np.concatenate([base[:, ids], targets[:, :, ids].reshape(3 * n, len(ids))])
            points, draws = bootstrap_balanced(subset, stack, plan['bootstrap_replicates'])
            canonical, trans = points[:3], points[3:].reshape(3, n)
            canonical_draws, trans_draws = draws[:, :3], draws[:, 3:].reshape(len(draws), 3, n)
            gaps, gap_draws = trans - canonical[:, None], trans_draws - canonical_draws[:, :, None]
            for m, name in enumerate(MODELS):
                for k, tokenizer in enumerate(data['tokenizers']):
                    ci = np.quantile(gap_draws[:, m, k], [.025, .975]).tolist()
                    metrics.append(dict(model=name, index=tokenizer['index'], regime=regime, split=split,
                        canonical_bpb=float(canonical[m]), transferred_bpb=float(trans[m, k]), gap_bpb=float(gaps[m, k]),
                        gap_ci95=ci, algorithm=tokenizer['config']['algorithm'],
                        pretokenizer=tokenizer['config']['pretokenizer'], vocab_size=tokenizer['config']['vocab_size']))
                for cohort, ks in cohorts.items():
                    median_draws = np.median(gap_draws[:, m, ks], axis=1)
                    summaries.append(dict(model=name, regime=regime, split=split, cohort=cohort, tokenizers=len(ks),
                        canonical_bpb=float(canonical[m]), mean_transferred_bpb=float(trans[m, ks].mean()),
                        median_transferred_bpb=float(np.median(trans[m, ks])),
                        mean_gap_bpb=float(gaps[m, ks].mean()), median_gap_bpb=float(np.median(gaps[m, ks])),
                        median_gap_ci95=np.quantile(median_draws, [.025, .975]).tolist(),
                        min_gap_bpb=float(gaps[m, ks].min()), max_gap_bpb=float(gaps[m, ks].max())))
            selections = {**cohorts, 'fixed0771': [plan['indices'].index(plan['primary_tokenizer'])]}
            for other in ['uniform20k', 'zett20k']:
                other_index = MODELS.index(other)
                for cohort, ks in selections.items():
                    absolute = trans[2, ks] - trans[other_index, ks]
                    gap = gaps[2, ks] - gaps[other_index, ks]
                    absolute_draws = (trans_draws[:, 2, ks] - trans_draws[:, other_index, ks]).mean(1)
                    difference_draws = (gap_draws[:, 2, ks] - gap_draws[:, other_index, ks]).mean(1)
                    comparisons.append(dict(comparison='rl20k-minus-' + other, regime=regime, split=split,
                        cohort=cohort, tokenizers=len(ks), mean_absolute_bpb_change=float(absolute.mean()),
                        mean_absolute_bpb_change_ci95=np.quantile(absolute_draws, [.025, .975]).tolist(),
                        mean_gap_change_bpb=float(gap.mean()),
                        mean_gap_change_ci95=np.quantile(difference_draws, [.025, .975]).tolist(),
                        canonical_bpb_change=float(canonical[2] - canonical[other_index])))
    return dict(sizes=sizes, metrics=metrics, summaries=summaries, paired_comparisons=comparisons,
        pool_regime_counts=data.get('pool_regime_counts'),
        cohort_indices={label: [data['tokenizers'][k]['index'] for k in ks] for label, ks in cohorts.items()},
        selection='Same frozen RL-canonical subset for every model; no per-model reselection.',
        limitations=['Subset was identified using RL20k canonical losses, so it is a conditional stress test, not a model-neutral population sample.',
                    'One training seed per method; bootstrap covers source-document sampling only.',
                    'Thirty-six representative settings plus two controls, not all1000 pool members.',
                    'Absolute loss and model-specific transfer gap measure different effects. Negative RL-minus-other changes favor RL.'])


def plot(report, output):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = ROOT / 'monitoring/web/Manrope.ttf'
    if font.exists():
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': 'Manrope' if font.exists() else 'DejaVu Sans', 'font.size': 9,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    names = ['ZeTT', 'Uniform pool', 'RL']
    fixed = [next(r for r in report['metrics'] if r['model'] == name and r['regime'] == 'context'
        and r['split'] == 'confirm' and r['index'] == 771) for name in MODELS]
    ax = axes[0, 0]
    ax.bar(np.arange(3) - .15, [r['canonical_bpb'] for r in fixed], width=.3, color='#D5DBE4', label='Canonical inputs')
    ax.bar(np.arange(3) + .15, [r['transferred_bpb'] for r in fixed], width=.3,
           color=[COLORS[name] for name in MODELS], label='Transferred inputs')
    ax.set(xticks=range(3), xticklabels=names, ylabel='Absolute BPB', title='A · Same tokenizer #0771, same stress passages')
    ax.legend(frameon=False)
    ax = axes[0, 1]
    for k, name in enumerate(MODELS):
        selected = [r for r in report['metrics'] if r['model'] == name and r['regime'] == 'context'
                    and r['split'] == 'confirm' and r['index'] in report['cohort_indices']['weak_pretokenization']]
        ax.scatter([k] * len(selected), [r['gap_bpb'] for r in selected], color=COLORS[name], alpha=.6, s=23)
        ax.plot(k, np.median([r['gap_bpb'] for r in selected]), '_', color='#273140', ms=20, mew=2)
    ax.axhline(0, color='#999999', lw=.8)
    ax.set(xticks=range(3), xticklabels=names, ylabel='Transfer gap · BPB', title='B · The broader weak regime: 24 tokenizer settings')
    ax = axes[1, 0]
    for k, cohort in enumerate(['regex', 'line256', 'no-regex']):
        for m, name in enumerate(MODELS):
            value = next(r for r in report['summaries'] if r['model'] == name and r['regime'] == 'context'
                         and r['split'] == 'confirm' and r['cohort'] == cohort)
            x = k + (m - 1) * .2
            ci = value['median_gap_ci95'];point = value['median_gap_bpb']
            ax.errorbar(x, point, yerr=[[max(0, point - ci[0])], [max(0, ci[1] - point)]],
                        fmt='o', color=COLORS[name], capsize=3, label=names[m] if k == 0 else None)
    ax.axhline(0, color='#999999', lw=.8)
    ax.set(xticks=range(3), xticklabels=['Regex', 'Line only', 'No regex'], ylabel='Median transfer gap · BPB',
           title='C · Pretokenization regimes, 12 settings each')
    ax.legend(frameon=False)
    ax = axes[1, 1]
    for m, name in enumerate(MODELS):
        values = [next(r for r in report['summaries'] if r['model'] == name and r['regime'] == regime
            and r['split'] == 'confirm' and r['cohort'] == 'weak_pretokenization') for regime in ['all', 'context']]
        ax.plot(range(2), [r['median_gap_bpb'] for r in values], 'o-', color=COLORS[name], label=names[m])
    ax.axhline(0, color='#999999', lw=.8)
    ax.set(xticks=range(2), xticklabels=['All natural text', 'Fixed context-dependent subset'],
           ylabel='Median transfer gap · BPB', title='D · Does the stress regime amplify each model’s gap?')
    ax.legend(frameon=False)
    fig.suptitle('Final trained models · identical canonical-boundary passages · 128 positions / 32 targets', fontsize=14)
    for suffix in ['png', 'pdf']:
        fig.savefig(output / f'model_comparison.{suffix}', dpi=180)
    plt.close(fig)


def write_report(report, data, output):
    lines = ['# Final trained models on the natural-text weakness regime', '',
        'The question is whether the substantial RL20k gap on context-dependent text is a broad weakness, and how the final ZeTT and uniform-pool models compare. All three models are evaluated on the same corrected raw prefixes, canonical continuations and native tokenizers.', '',
        '## The size of this setting', '',
        'The fixed confirmation stress set contains **229 passages from198 source documents**, **7,328 scored continuation tokens** and **23,074 scored UTF-8 bytes**. It is22.5% of the1,016 confirmation passages. Discovery contributes251 passages from203 separate documents. The source counts on confirmation are English web54, code61, math57, medical17 and legal40.', '',
        'This is a useful diagnostic panel, not a large benchmark or an estimate of the fraction of all internet text that fails. The selection rule can be applied to more held-out documents. It was fixed using RL’s canonical context benefit; membership is not recomputed for the other models. This model-dependent selection must be kept in mind when interpreting comparisons.', '',
        'The tokenizer basin is line-only/no-regex pretokenization: 24 representative settings spanning all three algorithms and four vocabulary scales, with 12 regex settings as controls. In the original 1,000-pool metadata, 648 settings use line-only/no-regex splitting, including 206 with vocabulary at least 96k. These counts describe the settings region; they do not establish that every member has a large gap. There are also two previously weak controls 0003/0635. The comparison does not rescore all 1,000 tokenizers.', '',
        '![Matched model comparison](model_comparison.png)', '', '## Fixed tokenizer0771: absolute loss and transfer gap', '',
        'Tokenizer0771 is WordPiece/no-regex/127k, selected using RL discovery documents before this comparison. Every row below uses exactly the same229 confirmation passages. Lower BPB is better.', '',
        '| Trained model | Canonical BPB | Transferred BPB | Gap | Paired95% interval for gap |',
        '|---|---:|---:|---:|---:|']
    for name in MODELS:
        r = next(r for r in report['metrics'] if r['model'] == name and r['regime'] == 'context'
                 and r['split'] == 'confirm' and r['index'] == 771)
        ci = r['gap_ci95']
        lines.append(f'| {LABELS[name]} | {r["canonical_bpb"]:.5f} | {r["transferred_bpb"]:.5f} | {r["gap_bpb"]:+.5f} | [{ci[0]:+.5f}, {ci[1]:+.5f}] |')
    lines.extend(['', 'The canonical baseline belongs to each trained LM. A lower gap does not by itself imply lower absolute loss: the LM’s canonical quality can change too. Absolute BPB and gap are both reported.', '',
        '## Does the broader tokenizer regime have headroom?', '',
        '| Model | Median regex gap (12) | Median line-only gap (12) | Median no-regex gap (12) | Median weak-regime gap (24) |',
        '|---|---:|---:|---:|---:|'])
    for name in MODELS:
        values = [next(r['median_gap_bpb'] for r in report['summaries'] if r['model'] == name
            and r['regime'] == 'context' and r['split'] == 'confirm' and r['cohort'] == cohort)
                  for cohort in ['regex', 'line256', 'no-regex', 'weak_pretokenization']]
        lines.append('| ' + LABELS[name] + ' | ' + ' | '.join(f'{value:+.5f}' for value in values) + ' |')
    lines.extend(['', 'Each entry first averages BPB within each source domain, then averages the five source scores equally for each tokenizer, and finally takes the median across the specified finite tokenizer group. It is not a median across individual passages.', '',
        '## Paired RL effects on the same stress passages', '',
        'The following differences average equally across the24 weak-pretokenization settings. Negative changes favor RL. Intervals resample paired source-document clusters within domains; two passages from one source document always move together.', '',
        '| Comparison | Absolute transferred BPB change | Paired95% interval | Transfer-gap change | Paired95% interval |',
        '|---|---:|---:|---:|---:|'])
    for other in ['uniform20k', 'zett20k']:
        r = next(r for r in report['paired_comparisons'] if r['comparison'] == 'rl20k-minus-' + other
                 and r['regime'] == 'context' and r['split'] == 'confirm' and r['cohort'] == 'weak_pretokenization')
        ci, gap_ci = r['mean_absolute_bpb_change_ci95'], r['mean_gap_change_ci95']
        lines.append(f'| RL minus {LABELS[other]} | {r["mean_absolute_bpb_change"]:+.5f} | [{ci[0]:+.5f}, {ci[1]:+.5f}] | {r["mean_gap_change_bpb"]:+.5f} | [{gap_ci[0]:+.5f}, {gap_ci[1]:+.5f}] |')
    lines.extend(['', '## Setup and verification', '',
        'The final checkpoints are the20k ZeTT-sampled-tokenizer run, the20k uniform1000-pool run and the20k shared-settings RL run, each consisting of a trained SmolLM2-1.7B backbone and input hypernetwork. The original canonical input embedding table and canonical output head are frozen and verified equal to the original model. This comparison does not run training.', '',
        'All38 native tokenizers and the entire2,032-passage corrected parent panel are reused. Each example contains BOS, an identical raw prefix encoded canonically or natively, and exactly32 canonical continuation targets. Total positions are at most128. All cuts preserve joint canonical tokenization. The hypernetwork reads the complete native vocabulary, with checksum-verified priors and surface features. Every batch’s per-token loss matches the model’s built-in causal loss. New weights are verified against their original step20k manifests; all large transfers occur cloud-to-cloud. Inference uses at most two B300 workers and no explicit CPU/memory reservations.', '',
        'We keep the original RL results unchanged. Context-dependent membership is pinned to480 original passage IDs across both splits, selected by RL-canonical prefix benefit and excluding image-link markup. We also report the unfiltered panel as a control. The paired bootstrap uses2000 replicates and accounts for source documents, but not model-training seeds or trying multiple subgroups. There is only one trained seed per method.', '',
        '## Reproduction and artifacts', '',
        '- `results.json.gz`: complete per-passage loss totals, frozen passage metadata, tokenizer settings, checkpoint hashes and fixed subset membership. Each loss total sums32 saved token losses in float64. This is sufficient to reproduce every aggregate and confidence interval here.',
        '- `analysis.json`: per-tokenizer scores, cohort summaries, sizes and paired model comparisons.',
        '- `metrics.csv`: absolute losses and gaps for each model/tokenizer/split/text setting.',
        '- `model_comparison.png` and `.pdf`: matched model plots.',
        '- `manifest.json`: SHA-256 hashes and sizes.', '',
        'Run `python scripts/analyze_natural_model_comparison.py --input reports/natural-model-comparison-20k/results.json.gz --output /tmp/model-comparison-reproduced` from the repository. The cloud entry point is `scripts/modal_natural_model_comparison.py`; the frozen plan is `plans/natural-model-comparison-v1.json`.', '',
        'Raw corpora, checkpoint weights, exact raw-text panel and full per-token scores remain in persistent Modal storage: blackpantherx86 / anytok-rl-all-pool-20k / natural-model-comparison-v1/. The corrected parent panel and original RL scores are in natural-transfer-aligned-v2/. Only score metadata and figures pass through the local caller.', ''])
    # Keep prose readable without altering numeric values or code paths.
    text = '\n'.join(lines)
    for old, new in [('from198', 'from 198'), ('is22.5%', 'is 22.5%'), ('the1,016', 'the 1,016'),
                     ('contributes251', 'contributes 251'), ('from203', 'from 203'), ('web54', 'web 54'),
                     ('code61', 'code 61'), ('math57', 'math 57'), ('medical17', 'medical 17'), ('legal40', 'legal 40'),
                     (':24', ': 24'), ('with12', 'with 12'), ('controls0003', 'controls 0003'), ('all1000', 'all 1,000'),
                     ('tokenizer0771', 'tokenizer 0771'), ('Tokenizer0771', 'Tokenizer 0771'), ('same229', 'same 229'),
                     ('Paired95%', 'Paired 95%'), ('the24', 'the 24'), ('the20k', 'the 20k'), ('uniform1000', 'uniform 1,000'),
                     ('All38', 'All 38'), ('entire2,032', 'entire 2,032'), ('exactly32', 'exactly 32'), ('most128', 'most 128'),
                     ('step20k', 'step 20k'), ('to480', 'to 480'), ('uses2000', 'uses 2,000'), ('sums32', 'sums 32')]:
        text = text.replace(old, new)
    (output / 'README.md').write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'modal_results/natural_model_comparison')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/natural-model-comparison-20k')
    args = parser.parse_args()
    if args.input.is_dir():
        data = pack(args.input)
    else:
        with gzip.open(args.input, 'rt') as stream:
            data = json.load(stream)
    report = analyze(data)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'results.json.gz').write_bytes(gzip.compress(json.dumps(data, sort_keys=True,
        ensure_ascii=False, separators=(',', ':')).encode(), mtime=0))
    (args.output / 'analysis.json').write_text(json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + '\n')
    with (args.output / 'metrics.csv').open('w') as stream:
        fields = [k for k in report['metrics'][0] if k != 'gap_ci95']
        writer = csv.DictWriter(stream, fieldnames=fields);writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in report['metrics'])
    plot(report, args.output);write_report(report, data, args.output)
    names = ['results.json.gz', 'analysis.json', 'metrics.csv', 'README.md', 'model_comparison.png', 'model_comparison.pdf']
    save_json(args.output / 'manifest.json', dict(files={name: dict(bytes=(args.output / name).stat().st_size,
        sha256=file_sha(args.output / name)) for name in names}))
    print(json.dumps([r for r in report['summaries'] if r['regime'] == 'context' and r['split'] == 'confirm'], indent=2))


if __name__ == '__main__':
    main()
