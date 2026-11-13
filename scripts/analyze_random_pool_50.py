"""Measured random50 pool baseline, paired with the fixed selected24."""
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import random

import numpy as np

from analyze_natural_transfer import bootstrap_balanced
from pool_generalization import digest, file_sha, save_json

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['zett20k', 'uniform20k', 'rl20k']
LABELS = ['ZeTT 20k', 'Uniform pool 20k', 'RL 20k']
COLORS = ['#8657BC', '#2F8FD0', '#36967B']


def pack(state):
    prepared = json.loads((state / 'preparation.json').read_text());plan = prepared['plan']
    source = json.loads((ROOT / 'reports/rl-pool-sweep-20k/spec.json').read_text())
    models = {}
    for name in MODELS:
        with gzip.open(state / (name + '.json.gz'), 'rt') as stream:
            raw = json.load(stream)
        if raw['panel_sha256'] != prepared['panel_sha256'] or raw['plan_sha256'] != digest(plan):
            raise ValueError('Model scores are not paired')
        ordered = sorted(raw['results'], key=lambda r: r['index'])
        if [r['index'] for r in ordered] != plan['indices']:
            raise ValueError('Missing native tokenizer')
        base = np.asarray(raw['canonical'], dtype=float)
        transferred = np.asarray([r['transferred'] for r in ordered], dtype=float)
        if base.shape != (len(prepared['metadata']), 32) or transferred.shape != (len(ordered), len(base), 32):
            raise ValueError('Missing continuation targets')
        models[name] = dict(canonical_nats=base.sum(1).tolist(), transferred_nats=transferred.sum(2).tolist())
        if name == 'rl20k':
            models[name]['no_prefix_nats'] = np.asarray(raw['no_prefix'], dtype=float).sum(1).tolist()
    return dict(plan=plan, preparation=prepared, models=models,
        tokenizers=[source['tokenizers'][i] for i in plan['indices']])


def analyze(data):
    plan = data['plan'];rows = data['preparation']['metadata'];indices = plan['indices']
    expected = random.Random(plan['seed']).sample(list(range(plan['population_size'])), 50)
    if plan['random_indices'] != expected or digest(expected) != plan['random_indices_sha256']:
        raise ValueError('The original uniform random draw changed')
    if len(indices) != len(set(indices)) or set(indices) != set(expected) | set(plan['hard_indices']):
        raise ValueError('Invalid union of random and selected settings')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate passage IDs')
    base = np.asarray([data['models'][name]['canonical_nats'] for name in MODELS])
    targets = np.asarray([data['models'][name]['transferred_nats'] for name in MODELS])
    if base.shape != (3, len(rows)) or targets.shape != (3, len(indices), len(rows)):
        raise ValueError('Missing model scores')
    if not np.isfinite(base).all() or not np.isfinite(targets).all() or (base < 0).any() or (targets < 0).any():
        raise ValueError('Invalid loss values')
    n = len(indices);replicates = plan['bootstrap_replicates']
    point, draws = bootstrap_balanced(rows, np.concatenate([base, targets.reshape(3 * n, len(rows))]), replicates)
    canonical = point[:3];trans = point[3:].reshape(3, n);gap = trans - canonical[:, None]
    canonical_draws = draws[:, :3];trans_draws = draws[:, 3:].reshape(replicates, 3, n)
    gap_draws = trans_draws - canonical_draws[:, :, None]
    sample_k = np.array([indices.index(i) for i in expected]);hard_k = np.array([indices.index(i) for i in plan['hard_indices']])
    rng = np.random.default_rng(20261118)
    sample_draws = rng.integers(50, size=(replicates, 50))
    correction = math.sqrt((plan['population_size'] - 50) / (plan['population_size'] - 1))
    models, metrics = [], []
    for m, name in enumerate(MODELS):
        reference = float(gap[m, sample_k].mean());hard = float(gap[m, hard_k].mean())
        document_reference = gap_draws[:, m, sample_k].mean(1)
        resampled = gap_draws[:, m, sample_k][np.arange(replicates)[:, None], sample_draws].mean(1)
        reference_draws = document_reference + correction * (resampled - document_reference)
        hard_draws = gap_draws[:, m, hard_k].mean(1)
        models.append(dict(model=name, canonical_bpb=float(canonical[m]),
            random50_mean_bpb=float(trans[m, sample_k].mean()), hard24_mean_bpb=float(trans[m, hard_k].mean()),
            random50_mean_gap=reference, hard24_mean_gap=hard, hard_minus_random=hard - reference,
            random50_ci95=np.quantile(reference_draws, [.025, .975]).tolist(),
            hard24_ci95=np.quantile(hard_draws, [.025, .975]).tolist(),
            difference_ci95=np.quantile(hard_draws - reference_draws, [.025, .975]).tolist(),
            random50_median_gap=float(np.median(gap[m, sample_k])), hard24_median_gap=float(np.median(gap[m, hard_k])),
            tokenizer_sample_standard_error=float(gap[m, sample_k].std(ddof=1) / math.sqrt(50) * correction)))
        for k, index in enumerate(indices):
            metrics.append(dict(model=name, index=index, in_random50=index in expected,
                in_selected24=index in plan['hard_indices'], canonical_bpb=float(canonical[m]),
                transferred_bpb=float(trans[m, k]), gap_bpb=float(gap[m, k])))
    by_index = {r['index']: r for r in data['tokenizers']}
    distribution = {field: dict(Counter(by_index[i]['config'][field] for i in expected))
                    for field in ['algorithm', 'pretokenizer']}
    return dict(models=models, metrics=metrics, sample_indices=expected, hard_indices=plan['hard_indices'],
        overlap=sorted(set(expected) & set(plan['hard_indices'])), distribution=distribution,
        passages=len(rows), documents=len({r['cluster_id'] for r in rows}), scored_tokens=32 * len(rows),
        scored_bytes=sum(r['continuation_bytes'] for r in rows), trimmed_passages=data['preparation']['trimmed_passages'],
        sampling='Uniform random50 without replacement from1000. No exclusions, stratification, reweighting or rerolling.',
        uncertainty='Paired source-document bootstrap within domains; additionally resample random50 tokenizer IDs, with finite-population correction on tokenizer-sampling deviations. The fixed24 are not treated as a random sample.',
        limitations='One trained checkpoint per method; intervals exclude training-seed variation. Text is the previously fixed RL-canonical stress subset, not a population sample of arbitrary text.')


def plot(report, output):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Patch
    font = ROOT / 'monitoring/web/Manrope.ttf'
    if font.exists():
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': 'Manrope' if font.exists() else 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    for k, r in enumerate(report['models']):
        for offset, field, ci_field, alpha in [(-.17, 'random50_mean_gap', 'random50_ci95', .45),
                                               (.17, 'hard24_mean_gap', 'hard24_ci95', 1.)]:
            v = r[field];ci = r[ci_field]
            axes[0].bar(k + offset, v, width=.32, color=COLORS[k], alpha=alpha)
            axes[0].errorbar(k + offset, v, yerr=[[max(0, v - ci[0])], [max(0, ci[1] - v)]],
                             fmt='none', color='#3F4957', capsize=3, lw=1)
        v = r['hard_minus_random'];ci = r['difference_ci95']
        axes[1].bar(k, v, width=.5, color=COLORS[k])
        axes[1].errorbar(k, v, yerr=[[max(0, v - ci[0])], [max(0, ci[1] - v)]],
                         fmt='none', color='#3F4957', capsize=3, lw=1)
    axes[0].set(title='Measured selected 24 versus measured random 50', ylabel='Mean transfer gap · BPB')
    axes[0].legend(handles=[Patch(color='#B5C3D2', label='Uniform random 50'), Patch(color='#526178', label='Selected 24')], frameon=False)
    axes[1].set(title='Additional gap for the selected 24', ylabel='Selected mean − random-sample mean · BPB')
    for ax in axes:
        ax.set_xticks(range(3), LABELS);ax.axhline(0, color='#999999', lw=.8)
    fig.suptitle('Same 229 stress passages · 50 tokenizers sampled uniformly from the pool of 1,000', fontsize=13)
    for suffix in ['png', 'pdf']:
        fig.savefig(output / f'random_pool_50.{suffix}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'modal_results/random_pool_50')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/random-pool-50')
    args = parser.parse_args()
    if args.input.is_dir():
        data = pack(args.input)
    else:
        with gzip.open(args.input, 'rt') as stream:
            data = json.load(stream)
    report = analyze(data);args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'results.json.gz').write_bytes(gzip.compress(json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode(), mtime=0))
    (args.output / 'analysis.json').write_text(json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + '\n')
    lines = ['# Selected 24 versus a uniform random sample of 50', '',
        '**The reference is measured, not imputed from representative settings.** Fifty distinct IDs were drawn uniformly without replacement from all 1,000 pool members using seed20261117, committed before scoring. The selected24 were not excluded. The draw overlaps that group at0011 and0912, giving72 distinct settings scored on all three trained models.', '',
        '| Model | Random50 mean gap | Selected24 mean gap | Additional gap | 95% interval for additional gap |',
        '|---|---:|---:|---:|---:|']
    for label, r in zip(LABELS, report['models']):
        ci = r['difference_ci95'];lines.append(f'| {label} | {r["random50_mean_gap"]:+.5f} | {r["hard24_mean_gap"]:+.5f} | {r["hard_minus_random"]:+.5f} | [{ci[0]:+.5f}, {ci[1]:+.5f}] |')
    lines += ['', '![Measured random50 comparison](random_pool_50.png)', '',
        'These are arithmetic means across tokenizers, not medians. Each tokenizer’s BPB is byte-weighted within source domains and then equally averaged over the five domains. Every row uses the same229 previously selected context-dependent confirmation passages from198 source documents, scoring7,328 canonical target tokens and23,074 UTF-8 bytes. The prefix is input only; the next32 canonical tokens are scored under teacher forcing, with at most128 positions including BOS.', '',
        f'Joint encoding validation required shortening **{report["trimmed_passages"]} prefixes**. All randomly selected settings were retained. Document IDs, cuts and scored continuation bytes are unchanged. All72 settings, including the selected24, are rescored on the shared prefixes; historical scores are not mixed into this comparison.', '',
        'The sample mean is an estimate of the full-pool mean, based on an actual uniform sample. It is not the measured mean of all1,000 tokenizers. The selected24 are a fixed configuration-selected group. Uncertainty combines paired document-cluster resampling within domains with resampling of the random50 IDs; tokenizer-sampling deviations use the finite-population correction. Shared random IDs and source draws preserve pairing between models. Intervals do not cover model-training seeds.', '',
        'The random sample’s settings distribution and exact IDs are in analysis.json. The committed draw and checkpoint hashes are in the plan inside results.json.gz. The same original trained canonical embedding tables and output heads are verified for each model. Every batch’s token loss agrees with Hugging Face’s built-in causal loss. Native embeddings are generated from complete vocabularies and checksum-verified feature arrays.', '',
        'The text subset was defined using the earlier RL-canonical context-benefit diagnostic and is fixed across models. It is a stress set, not a prevalence estimate for arbitrary text. The former36-setting proxy is superseded and removed from the active dashboard and report folder.', '',
        'Reproduce with `python scripts/analyze_random_pool_50.py --input reports/random-pool-50/results.json.gz --output /tmp/random-pool-50`.', '',
        'Full native tokenizer files, raw prefixes, checkpoints and per-token losses remain in persistent cloud storage: blackpantherx86 / anytok-rl-all-pool-20k / random-pool-50-v1/. The report stores all per-passage loss totals needed to reproduce the aggregates and confidence intervals.', '']
    text = '\n'.join(lines)
    for old, new in [('seed20261117', 'seed 20261117'), ('selected24', 'selected 24'), ('Selected24', 'Selected 24'),
        ('Random50', 'Random 50'), ('random50', 'random 50'), ('at0011', 'at 0011'), ('and0912', 'and 0912'),
        ('giving72', 'giving 72'), ('same229', 'same 229'), ('from198', 'from 198'), ('scoring7,328', 'scoring 7,328'),
        ('and23,074', 'and 23,074'), ('next32', 'next 32'), ('most128', 'most 128'), ('All72', 'All 72'),
        ('all1,000', 'all 1,000'), ('former36', 'former 36')]:
        text = text.replace(old, new)
    # Preserve file links after prose spacing normalization.
    text = text.replace('random 50_pool', 'random50_pool')
    (args.output / 'README.md').write_text(text)
    plot(report, args.output)
    names = ['results.json.gz', 'analysis.json', 'README.md', 'random_pool_50.png', 'random_pool_50.pdf']
    save_json(args.output / 'manifest.json', dict(files={name: dict(bytes=(args.output / name).stat().st_size,
        sha256=file_sha(args.output / name)) for name in names}))
    print(json.dumps(report['models'], indent=2))


if __name__ == '__main__':
    main()
