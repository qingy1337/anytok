"""Reproduce the natural-text transfer map from saved paired token losses."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np

from natural_transfer import DOMAINS, TAGS, metrics, validate_scores
from pool_generalization import digest, file_sha, save_json

ROOT = Path(__file__).resolve().parents[1]
COLORS = {'bpe': '#2F8FD0', 'wordpiece': '#8657BC', 'unigram': '#36967B'}
LABELS = dict(all='All natural text', no_image_markup='Exclude image-link markup',
    context_dependent='Context-dependent text', context_dependent_no_markup='Context-dependent; exclude image markup',
    **{'web-en': 'English web', 'code': 'Code', 'math-en': 'Math', 'medical-en': 'Medical', 'legal-en': 'Legal'},
    image_markup='Image-link markup', structured='Structured rows', numeric_dense='At least 8% digit characters',
    repeated_number='A number repeats from prefix to continuation', plain_prose='Plain prose (heuristic)')


def load(path):
    if path.is_dir():
        plan = json.loads((path / 'plan.json').read_text())
        result = dict(plan=plan, preparation=json.loads((path / 'preparation.json').read_text()), results=[])
        for index in plan['indices']:
            with gzip.open(path / f'{index:04d}.json.gz', 'rt') as stream:
                row = json.load(stream)
            for name in ['canonical', 'no_prefix']:
                if name not in result:
                    result[name] = row[name]
                if row.pop(name) != result[name]:
                    raise ValueError('Canonical worker baselines differ')
            result['results'].append(row)
        return result
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def ranks(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    cumulative = np.cumsum(counts)
    return ((cumulative - counts + cumulative - 1) / 2)[inverse]


def correlation(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def bootstrap_balanced(rows, nats, replicates=2000):
    """K×passage paired excess NLL -> replicate×K balanced-domain BPB."""
    total, means, domain_count = np.zeros((replicates, len(nats))), np.zeros(len(nats)), 0
    for domain in DOMAINS:
        rng = np.random.default_rng(int(digest(['natural-bootstrap', domain])[:12], 16))
        positions = [i for i, r in enumerate(rows) if r['domain'] == domain]
        if not positions:
            continue
        subrows = [rows[i] for i in positions]
        clusters = sorted({r['cluster_id'] for r in subrows})
        lookup = {cluster: j for j, cluster in enumerate(clusters)}
        inverse = np.array([lookup[r['cluster_id']] for r in subrows])
        # Both source windows move together, including in feature-defined subgroups.
        draws = rng.integers(len(clusters), size=(replicates, len(clusters)))
        offsets = np.arange(replicates)[:, None] * len(clusters)
        multiplicities = np.bincount((draws + offsets).ravel(), minlength=replicates * len(clusters))
        weights = multiplicities.reshape(replicates, len(clusters))[:, inverse].astype(float)
        bytes_ = np.array([r['continuation_bytes'] for r in subrows], dtype=float) * math.log(2)
        values = nats[:, positions]
        total += (weights @ values.T) / (weights @ bytes_)[:, None]
        means += values.sum(1) / bytes_.sum()
        domain_count += 1
    if not domain_count:
        raise ValueError('Cannot bootstrap an empty subgroup')
    return means / domain_count, total / domain_count


def subgroup_masks(rows, base, no_prefix):
    byte_counts = np.array([r['continuation_bytes'] for r in rows]) * math.log(2)
    benefit = (no_prefix - base).sum(1) / byte_counts
    thresholds = {domain: float(np.quantile([benefit[i] for i, r in enumerate(rows)
        if r['domain'] == domain and r['split'] == 'screen'], .75)) for domain in DOMAINS}
    high = np.array([benefit[i] >= thresholds[r['domain']] for i, r in enumerate(rows)])
    clean = np.array([not r['tags']['image_markup'] for r in rows])
    groups = dict(all=np.ones(len(rows), dtype=bool), no_image_markup=clean,
                  context_dependent=high, context_dependent_no_markup=high & clean)
    groups.update({domain: np.array([r['domain'] == domain for r in rows]) for domain in DOMAINS})
    groups.update({tag: np.array([r['tags'][tag] for r in rows]) for tag in TAGS})
    return groups, thresholds, benefit


def decompose(gaps, rows):
    weights = np.zeros(len(rows))
    for domain in DOMAINS:
        mask = np.array([r['domain'] == domain for r in rows])
        weights[mask] = 1 / (len(DOMAINS) * mask.sum())
    mean = float((gaps.mean(0) * weights).sum())
    tokenizer = (gaps * weights).sum(1) - mean
    passage = gaps.mean(0) - mean
    interaction = gaps - mean - tokenizer[:, None] - passage[None]
    components = dict(tokenizer=float(np.mean(tokenizer ** 2)),
        shared_passage=float(np.sum(passage ** 2 * weights)),
        interaction=float(np.mean((interaction ** 2 * weights).sum(1))))
    total = sum(components.values())
    return dict(balanced_mean_gap=mean, variances=components,
        variance_fractions={k: v / total if total else 0 for k, v in components.items()},
        interpretation='Descriptive two-way decomposition of passage BPB gaps. Shared-passage effects versus consistent tokenizer offsets versus tokenizer×passage residuals; finite-text noise is included, not isolated causally.')


def analyze(data):
    rows = data['preparation']['metadata']
    plan = data['plan']
    if digest(plan) != data['preparation']['plan_sha256']:
        raise ValueError('Frozen plan changed')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate passage IDs')
    for key in ['cluster_id', 'document_sha256']:
        cohorts = [{r[key] for r in rows if r['split'] == split} for split in ['screen', 'confirm']]
        if cohorts[0] & cohorts[1]:
            raise ValueError('Discovery/confirmation source overlap')
    context_lengths = {}
    if all('canonical_prefix_ids' in r and 'native_prefix_lengths' in r for r in rows):
        for split in ['screen', 'confirm']:
            selected = [r for r in rows if r['split'] == split]
            canonical_lengths = [len(r['canonical_prefix_ids']) for r in selected]
            native_lengths = [v for r in selected for v in r['native_prefix_lengths'].values()]
            context_lengths[split] = {name: dict(min=min(lengths), median=float(np.median(lengths)), max=max(lengths))
                for name, lengths in [('canonical_prefix_tokens', canonical_lengths), ('native_prefix_tokens', native_lengths)]}
    base = validate_scores(rows, data['canonical'])
    no_prefix = validate_scores(rows, data['no_prefix'])
    results = sorted(data['results'], key=lambda r: r['index'])
    if [r['index'] for r in results] != plan['indices']:
        raise ValueError('Missing/unexpected tokenizers')
    for result in results:
        if (result['plan_sha256'] != digest(plan)
                or result['panel_sha256'] != data['preparation']['panel_sha256']):
            raise ValueError('Unpaired evaluation result')
    transferred = np.array([validate_scores(rows, r['transferred']) for r in results])
    delta = transferred - base[None]
    groups, thresholds, benefit = subgroup_masks(rows, base, no_prefix)
    group_domains = {}
    for group, mask in groups.items():
        # A rare tag must not silently change domain weights between the two cohorts.
        common = [domain for domain in DOMAINS if all(any(mask[i] and r['domain'] == domain
            and r['split'] == split for i, r in enumerate(rows)) for split in ['screen', 'confirm'])]
        group_domains[group] = common
        groups[group] = mask & np.array([r['domain'] in common for r in rows])
    representative = np.array([r['index'] in plan['representative_indices'] for r in results])
    identifiers = np.array([r['index'] for r in results])
    metadata = {r['index']: r for r in results}
    all_metrics, draws_by_group, means_by_group = [], {}, {}
    for group, mask in groups.items():
        for split in ['screen', 'confirm']:
            positions = np.flatnonzero(mask & np.array([r['split'] == split for r in rows]))
            if not len(positions):
                continue
            selected = [rows[i] for i in positions]
            mean, draws = bootstrap_balanced(selected, delta[:, positions].sum(2),
                                             plan['bootstrap_replicates'])
            canonical_mean, _ = bootstrap_balanced(selected, base[positions].sum(1)[None], 1)
            means_by_group[group, split] = mean
            draws_by_group[group, split] = draws
            cis = np.quantile(draws, [.025, .975], axis=0)
            for k, result in enumerate(results):
                values = metrics(selected, delta[k, positions], base[positions], transferred[k, positions])
                all_metrics.append(dict(index=result['index'], algorithm=result['config']['algorithm'],
                    pretokenizer=result['config']['pretokenizer'], vocab_size=result['config']['vocab_size'],
                    representative=bool(representative[k]), group=group, split=split,
                    mean_domain_canonical_bpb=float(canonical_mean[0]),
                    mean_domain_transferred_bpb=float(canonical_mean[0] + mean[k]),
                    mean_domain_gap_bpb=float(mean[k]), mean_domain_gap_ci95=cis[:, k].tolist(), **values))
    summaries = []
    for group in groups:
        for split in ['screen', 'confirm']:
            selected = [r for r in all_metrics if r['group'] == group and r['split'] == split and r['representative']]
            if not selected:
                continue
            summaries.append(dict(group=group, split=split, passages=selected[0]['passages'],
                documents=selected[0]['documents'], canonical_bpb=selected[0]['canonical_bpb'],
                domains=group_domains[group],
                mean_domain_canonical_bpb=selected[0]['mean_domain_canonical_bpb'],
                median_mean_domain_transferred_bpb=float(np.median([r['mean_domain_transferred_bpb'] for r in selected])),
                median_gap_bpb=float(np.median([r['mean_domain_gap_bpb'] for r in selected])),
                min_gap_bpb=min(r['mean_domain_gap_bpb'] for r in selected),
                max_gap_bpb=max(r['mean_domain_gap_bpb'] for r in selected),
                median_worst_decile_mean=float(np.median([r['worst_decile_mean_gap_bpb'] for r in selected])),
                median_p90_gap=float(np.median([r['p90_passage_gap_bpb'] for r in selected])),
                median_fraction_above_01=float(np.median([r['fraction_gap_above_01'] for r in selected])),
                median_fraction_above_025=float(np.median([r['fraction_gap_above_025'] for r in selected]))))
    selections = []
    for group in groups:
        if (group, 'screen') not in means_by_group or (group, 'confirm') not in means_by_group:
            continue
        a, b = [means_by_group[group, split][representative] for split in ['screen', 'confirm']]
        candidate_ids = identifiers[representative]
        selected_local = int(np.argmax(a))
        selected_global = int(np.flatnonzero(representative)[selected_local])
        confirm_draws = draws_by_group[group, 'confirm']
        advantage_draws = confirm_draws[:, selected_global] - confirm_draws[:, representative].mean(1)
        top = np.argsort(a)[-math.ceil(len(a) * .2):]
        top_global = np.flatnonzero(representative)[top]
        top_draws = confirm_draws[:, top_global].mean(1) - confirm_draws[:, representative].mean(1)
        selections.append(dict(group=group, spearman_screen_confirm=correlation(a, b),
            screen_selected_index=int(candidate_ids[selected_local]),
            screen_selected_gap_bpb=float(a[selected_local]), confirmation_selected_gap_bpb=float(b[selected_local]),
            confirmation_uniform_gap_bpb=float(b.mean()),
            confirmation_selection_advantage_bpb=float(b[selected_local] - b.mean()),
            confirmation_advantage_ci95=np.quantile(advantage_draws, [.025, .975]).tolist(),
            screen_top20_indices=candidate_ids[top].tolist(),
            confirmation_top20_advantage_bpb=float(b[top].mean() - b.mean()),
            confirmation_top20_advantage_ci95=np.quantile(top_draws, [.025, .975]).tolist()))
    # A policy that sees the source domain can select a separate setting for each.
    domain_picks = [next(r for r in selections if r['group'] == d) for d in DOMAINS]
    domain_draws = []
    for domain, selected in zip(DOMAINS, domain_picks):
        k = int(np.flatnonzero(identifiers == selected['screen_selected_index'])[0])
        draws = draws_by_group[domain, 'confirm']
        domain_draws.append(draws[:, k] - draws[:, representative].mean(1))
    domain_advantage = dict(selections={r['group']: r['screen_selected_index'] for r in domain_picks},
        confirmation_advantage_bpb=float(np.mean([r['confirmation_selection_advantage_bpb'] for r in domain_picks])),
        confirmation_advantage_ci95=np.quantile(np.mean(domain_draws, axis=0), [.025, .975]).tolist())
    positions = []
    for group in ['all', 'no_image_markup', 'context_dependent_no_markup']:
        for split in ['screen', 'confirm']:
            selected = np.flatnonzero(groups[group] & np.array([r['split'] == split for r in rows]))
            for start in [0, 8, 16, 24]:
                # Bits per scored token avoids inventing per-token byte boundaries.
                by_domain = []
                for domain in DOMAINS:
                    ids = [i for i in selected if rows[i]['domain'] == domain]
                    if ids:
                        by_domain.append(delta[:, ids, start:start + 8].mean((1, 2)) / math.log(2))
                values = np.mean(by_domain, axis=0)
                positions.append(dict(group=group, split=split, start=start + 1, end=start + 8,
                    median_excess_bits_per_token=float(np.median(values[representative])),
                    per_tokenizer={str(i): float(v) for i, v in zip(identifiers, values)}))
    decomposition, oracle = {}, {}
    for split in ['screen', 'confirm']:
        ids = [i for i, r in enumerate(rows) if r['split'] == split]
        byte_counts = np.array([rows[i]['continuation_bytes'] for i in ids]) * math.log(2)
        decomposition[split] = decompose(delta[representative][:, ids].sum(2) / byte_counts, [rows[i] for i in ids])
        values = delta[representative][:, ids].sum(2)
        selected_rows = [rows[i] for i in ids]
        oracle_mean, _ = bootstrap_balanced(selected_rows, np.stack([values.max(0), values.mean(0)]), 10)
        oracle[split] = dict(hindsight_best_per_passage_gap_bpb=float(oracle_mean[0]),
            uniform_gap_bpb=float(oracle_mean[1]),
            hindsight_selection_advantage_bpb=float(oracle_mean[0] - oracle_mean[1]),
            limitation='Selects a different tokenizer after seeing every passage loss. Optimistic finite-panel diagnostic, not a deployable policy or held-out generalization result.')
    # Select a small auditable set; retain IDs, request raw snippets only in cloud.
    example_requests = []
    confirmation = np.array([r['split'] == 'confirm' for r in rows])
    mean_gap = delta[representative].sum(2).mean(0) / (np.array([r['continuation_bytes'] for r in rows]) * math.log(2))
    for domain in DOMAINS:
        ids = [i for i, r in enumerate(rows) if r['domain'] == domain and confirmation[i] and not r['tags']['image_markup']]
        for i in sorted(ids, key=lambda j: -mean_gap[j])[:2]:
            example_requests.append(dict(id=rows[i]['id'], mean_gap_bpb=float(mean_gap[i]),
                canonical_bpb=float(base[i].sum() / (rows[i]['continuation_bytes'] * math.log(2))),
                context_benefit_bpb=float(benefit[i]), tags=rows[i]['tags']))
    return dict(summaries=summaries, metrics=all_metrics, selections=selections,
        domain_conditioned_selection=domain_advantage, positions=positions, decomposition=decomposition,
        hindsight_oracle=oracle,
        context_lengths=context_lengths,
        context_thresholds=thresholds, example_requests=example_requests,
        subgroup_domains=group_domains,
        canonical_worker_check='All worker per-token canonical and BOS-only baselines agree exactly.',
        limits=plan['limits'])


def plot(report, output):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = ROOT / 'monitoring/web/Manrope.ttf'
    if font.exists():
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': 'Manrope' if font.exists() else 'DejaVu Sans',
        'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    metrics_ = report['metrics']
    def subset(group, split='confirm'):
        return [r for r in metrics_ if r['group'] == group and r['split'] == split and r['representative']]
    labels = ['English web', 'Code', 'Math', 'Medical', 'Legal']
    ax = axes[0, 0]
    for k, domain in enumerate(DOMAINS):
        values = subset(domain)
        for algorithm, color in COLORS.items():
            rows = [r for r in values if r['algorithm'] == algorithm]
            offset = list(COLORS).index(algorithm) * .16 - .16
            ax.scatter([k + offset] * len(rows), [r['mean_domain_gap_bpb'] for r in rows],
                       color=color, alpha=.65, s=20, label=algorithm.title() if k == 0 else None)
    ax.axhline(0, color='#999999', lw=.8)
    ax.set(xticks=range(5), xticklabels=labels, ylabel='Transfer gap · BPB', title='A · Fresh natural text, 36 tokenizer settings')
    ax.legend(frameon=False, fontsize=8)
    ax = axes[0, 1]
    names = ['All text', 'No image\nmarkup', 'Context-dependent,\nno image markup']
    for k, group in enumerate(['all', 'no_image_markup', 'context_dependent_no_markup']):
        values = subset(group)
        average = [r['mean_domain_gap_bpb'] for r in values]
        tail = [r['worst_decile_mean_gap_bpb'] for r in values]
        ax.scatter([k - .12] * len(values), average, s=15, color='#2F8FD0', alpha=.5)
        ax.scatter([k + .12] * len(values), tail, s=15, color='#8657BC', alpha=.5)
        ax.plot(k - .12, np.median(average), '_', color='#185A83', ms=16, mew=2)
        ax.plot(k + .12, np.median(tail), '_', color='#52326F', ms=16, mew=2)
    ax.scatter([], [], color='#2F8FD0', label='Five-domain average')
    ax.scatter([], [], color='#8657BC', label='Worst 10% of passages')
    ax.axhline(0, color='#999999', lw=.8)
    ax.set(xticks=range(3), xticklabels=names, ylabel='Transfer gap · BPB', title='B · Averages and tails answer different questions')
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 0]
    a, b = subset('all', 'screen'), subset('all')
    for left, right in zip(a, b):
        ax.scatter(left['mean_domain_gap_bpb'], right['mean_domain_gap_bpb'],
            color=COLORS[left['algorithm']], s=25)
    pick = next(r for r in report['selections'] if r['group'] == 'all')
    ax.set(xlabel='Discovery gap · BPB', ylabel='Confirmation gap · BPB',
        title=f'C · Do hard tokenizer rankings repeat?  ρ = {pick["spearman_screen_confirm"]:.2f}')
    ax = axes[1, 1]
    for group, label, color in [('all', 'All text', '#2F8FD0'),
            ('context_dependent_no_markup', 'Context-dependent, no image markup', '#8657BC')]:
        values = [r for r in report['positions'] if r['group'] == group and r['split'] == 'confirm']
        ax.plot(range(4), [r['median_excess_bits_per_token'] for r in values], 'o-', label=label, color=color)
    ax.axhline(0, color='#999999', lw=.8)
    ax.set(xticks=range(4), xticklabels=['1–8', '9–16', '17–24', '25–32'],
        xlabel='Scored continuation token positions', ylabel='Excess bits per token',
        title='D · Where does the prefix-transfer penalty appear?')
    ax.legend(frameon=False, fontsize=8)
    figure.suptitle('Natural text · final RL20k model · 128 total positions / 32 scored tokens', fontsize=14)
    for suffix in ['png', 'pdf']:
        figure.savefig(output / f'natural_transfer.{suffix}', dpi=180)
    plt.close(figure)


def write_report(report, data, output):
    def summary(group):
        return next(r for r in report['summaries'] if r['group'] == group and r['split'] == 'confirm')
    overall = summary('all')
    selection = next(r for r in report['selections'] if r['group'] == 'all')
    lines = ['# Natural-text transfer gaps: is there a useful adversarial tokenizer signal?', '',
        'This tests the final RL20k model on ordinary held-out text. No new training or synthetic copying prompts are used.', '',
        f'Across 36 settings selected by generator configuration, the median five-domain confirmation gap is **{overall["median_gap_bpb"]:+.5f} BPB**. The median gap on each setting’s worst 10% of confirmation passages is **{overall["median_worst_decile_mean"]:+.5f} BPB**.', '',
        f'The tokenizer selected as hardest on discovery is **{selection["screen_selected_index"]:04d}**. On independent confirmation documents it is **{selection["confirmation_selection_advantage_bpb"]:+.5f} BPB** harder than uniform over these 36 settings; paired 95% interval **[{selection["confirmation_advantage_ci95"][0]:+.5f}, {selection["confirmation_advantage_ci95"][1]:+.5f}]**. Discovery/confirmation rank correlation is **{selection["spearman_screen_confirm"]:.3f}**.', '',
        '![Natural-text transfer map](natural_transfer.png)', '', '## What these numbers mean', '',
        'A transfer gap is transferred BPB minus canonical BPB on exactly the same text. Both use the same trained LM and canonical output head; only prefix tokenization and prefix input embeddings differ. A positive gap is additional loss introduced by transfer. A difficult passage with high canonical loss is not automatically a transfer failure.', '',
        'The average equally weights English web, code, math, medical and legal text. Within a domain, BPB is total negative log likelihood divided by total scored UTF-8 bytes and ln(2). Tail summaries first compute each passage’s gap, then average the worst 10% of passages equally. Thus the average and tail use different explicit weighting; the tail describes a minority of text and is not overall performance.', '',
        '## Confirmation aggregates', '',
        '| Text setting | Passages / documents | Median average gap | Min–max average gap | Median worst 10% gap | Median fraction above+.1 BPB |',
        '|---|---:|---:|---:|---:|---:|']
    for group in ['all', 'no_image_markup', 'context_dependent_no_markup', *DOMAINS, *TAGS]:
        matches = [r for r in report['summaries'] if r['group'] == group and r['split'] == 'confirm']
        if not matches:
            continue
        r = matches[0]
        lines.append(f'| {LABELS[group]} | {r["passages"]} / {r["documents"]} | {r["median_gap_bpb"]:+.5f} | {r["min_gap_bpb"]:+.5f}–{r["max_gap_bpb"]:+.5f} | {r["median_worst_decile_mean"]:+.5f} | {r["median_fraction_above_01"]:.1%} |')
    lines.extend(['', 'Medians/minima/maxima are across 36 representative settings, not across documents or all 1,000 pool members. The two previously weak controls 0003/0635 are included in the data but excluded from these summaries. Tags overlap. Numeric density means at least 8% digit characters; repeated numbers means a suffix number containing at least three consecutive digits also appears in the prefix. Structured text is detected by fixed punctuation/row patterns. Image-link markup is retained and also removed in the named sensitivity.', '',
        '## Does selecting hard tokenizers generalize?', '',
        'Choose the hardest setting on discovery, then measure its gap on confirmation. The reference is the mean gap from uniformly choosing among the 36 evaluated representative settings. This is an offline selection test, not a trained RL policy or a comparison with uniform-trained model weights.', '',
        '| Setting | Discovery-selected tokenizer | Confirmation selection advantage | Paired 95% interval | Rank correlation |',
        '|---|---:|---:|---:|---:|'])
    for r in report['selections']:
        ci = r['confirmation_advantage_ci95']
        rho = r['spearman_screen_confirm']
        label = LABELS[r['group']]
        lines.append(f'| {label} | {r["screen_selected_index"]:04d} | {r["confirmation_selection_advantage_bpb"]:+.5f} | [{ci[0]:+.5f}, {ci[1]:+.5f}] | {rho:.3f} |' if rho is not None else f'| {label} | {r["screen_selected_index"]:04d} | {r["confirmation_selection_advantage_bpb"]:+.5f} | [{ci[0]:+.5f}, {ci[1]:+.5f}] | undefined |')
    domain = report['domain_conditioned_selection']
    lines.extend(['', f'Allowing a different discovery-selected setting for each domain gives a confirmation advantage of **{domain["confirmation_advantage_bpb"]:+.5f} BPB**, with paired 95% interval [{domain["confirmation_advantage_ci95"][0]:+.5f}, {domain["confirmation_advantage_ci95"][1]:+.5f}]. This is a contextual selector that knows the source domain; our previous settings-only policy did not condition on text/domain.', '',
        'A positive, repeatable selection advantage is the useful signal for an adversarial tokenizer policy. A large tail shared by every tokenizer instead motivates finding hard text. A high tail alone does not show that selecting tokenizers is valuable. The top 20% selection sensitivity and a descriptive tokenizer/passage/interaction variance decomposition are in analysis.json.', '',
        '## Context and scoring, explicitly', '',
        'Each example is BOS + raw prefix + 32 canonical continuation tokens, with at most 128 positions total. Prefixes start at at most 64 canonical tokens and are jointly shortened only if necessary for all 38 native encoders; at least 16 canonical prefix tokens remain. The native prefix can have a different number of tokens, but every setting sees identical raw prefix text and identical scored continuation IDs/bytes. Teacher forcing supplies earlier canonical continuation tokens, so later positions can rely less on the transferred prefix. Panel D shows excess bits/token in successive eight-token blocks; it does not assign invented byte boundaries to individual tokens.', '',
        '```text\nCanonical: BOS (1) | raw prefix as ≤64 canonical tokens | 32 scored canonical tokens\nTransfer:  BOS (1) | same raw prefix as ≤95 native tokens | same 32 scored canonical tokens\n```', '',
        'For the context-dependent stratum, score the same continuation with BOS alone and with its canonical prefix. Context benefit is BOS-only BPB minus canonical-prefix BPB. For each domain, freeze the 75th-percentile discovery threshold and apply that numerical threshold to confirmation. This uses only canonical losses, never transferred losses, and changes no text. It is a diagnostic stratum, not the original training distribution.', '',
        '## Data independence and verification', '',
        'The panel contains 2,032 passages from 1,016 source documents. Each split has 1,016 passages from 508 documents: 128 documents each for web/code/math, 96 legal and 28 medical documents, with two disjoint source windows per document. We use deterministic ID hashing and deterministic locations in opposite halves of each full document. Documents must contain at least 1,024 characters to support both windows; this excludes shorter documents and limits the population represented. All source IDs from the previous all-pool evaluation are excluded. Discovery/confirmation source IDs and full-document text hashes are disjoint. Rejections and every surviving passage ID/hash are in results.json.gz. The five-domain average compensates for unequal domain sample counts; the pooled passage tail retains those sample counts.', '',
        'All model/checkpoint/tokenizer/feature checksums are verified in the cloud. Hypernetwork inference uses the complete native vocabulary. Canonical input/output tables remain equal to their original values. Every batch’s per-token loss agrees with the model’s built-in causal loss, and canonical/BOS-only per-token losses agree exactly across workers. Confidence intervals resample paired document clusters within domains, so the two windows from one document move together. There are 2,000 replicates. Intervals condition on the chosen discovery setting and one trained model; they do not cover model-training seeds or correct for trying multiple subgroups.', '',
        '## Files and reproduction', '',
        '- `results.json.gz`: frozen plan, passage metadata, full paired 32-token losses for all 38 settings and canonical/BOS-only controls.',
        '- `analysis.json`: all aggregates, paired intervals, held-out selection tests, context thresholds and example requests.',
        '- `metrics.csv`: per-setting, per-split, per-subgroup metrics.',
        '- `natural_transfer.png` / `.pdf`: four-panel results.',
        '- `manifest.json`: SHA-256 checksums and file sizes.', '',
        'Run `python scripts/analyze_natural_transfer.py --input reports/natural-transfer-20k/results.json.gz --output /tmp/natural-transfer-reproduced` from the repository. The cloud launcher is `scripts/modal_natural_transfer.py`; design and selection are frozen in `plans/natural-transfer-v1.json` and `.md`. Raw corpora, full raw-text panel, tokenizer files and model weights remain in persistent Modal storage: `blackpantherx86 / anytok-rl-all-pool-20k / natural-transfer-v1/`.', '',
        'This is a balanced five-source English/code diagnostic and a stratified tokenizer subset, not an estimate for arbitrary internet text or every possible tokenizer. It does not establish that continuous generation helps, that RL training beats uniform training, or a causal mechanism for any failure.', ''])
    if report.get('example_inspection'):
        lines.extend(['## Inspectable natural passages', '',
            'See [examples.json](examples.json) for ten actual prefix/continuation pairs: two per source with large average transfer gaps on confirmation, excluding image markup. These are selected illustrations, not independent evidence or a prevalence estimate. Their scores are already included in the complete aggregates above.', ''])
    (output / 'README.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'modal_results/natural_transfer')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/natural-transfer-20k')
    args = parser.parse_args()
    data = load(args.input)
    report = analyze(data)
    args.output.mkdir(parents=True, exist_ok=True)
    examples = (args.input if args.input.is_dir() else args.input.parent) / 'examples.json'
    if examples.exists():
        snippets = json.loads(examples.read_text())
        if {r['id'] for r in snippets} != {r['id'] for r in report['example_requests']}:
            raise ValueError('Unmatched inspected examples')
        save_json(args.output / 'examples.json', snippets)
        report['example_inspection'] = dict(examples=len(snippets), file='examples.json')
    save_json(args.output / 'analysis.json', report)
    packed = json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    (args.output / 'results.json.gz').write_bytes(gzip.compress(packed, mtime=0))
    with (args.output / 'metrics.csv').open('w') as stream:
        fields = [k for k in report['metrics'][0] if k != 'mean_domain_gap_ci95']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in report['metrics'])
    plot(report, args.output)
    write_report(report, data, args.output)
    files = ['README.md', 'results.json.gz', 'analysis.json', 'metrics.csv', 'natural_transfer.png', 'natural_transfer.pdf']
    if report.get('example_inspection'):
        files.append('examples.json')
    save_json(args.output / 'manifest.json', {'files': {name: dict(bytes=(args.output / name).stat().st_size,
        sha256=file_sha(args.output / name)) for name in files}})
    print(json.dumps({'report': str(args.output), 'summaries': report['summaries'],
                      'selection': report['selections'][0]}, indent=2))


if __name__ == '__main__':
    main()
