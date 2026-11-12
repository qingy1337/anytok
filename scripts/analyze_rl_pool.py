"""Export and plot the exhaustive RL20k tokenizer-pool evaluation."""
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

from pool_generalization import digest
from pool_rl_sweep import rank_results, summarize_candidate
from uniform_trajectory import PROFILE_DOMAINS

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'modal_results/rl_pool_20k'
OUT = ROOT / 'reports/rl-pool-sweep-20k'
DOMAIN_NAMES = {'web-en': 'English', 'code': 'Code', 'web-zh': 'Chinese',
    'web-ja': 'Japanese', 'web-ar': 'Arabic', 'web-hi': 'Hindi',
    'math-en': 'Math', 'medical-en': 'Medical', 'legal-en': 'Legal'}


def read(path):
    return json.loads(Path(path).read_text())


def paired_interval(a, b, metadata, replicates=5000):
    """Equal-domain BPB delta, resampling shared documents within each domain."""
    delta = np.asarray(a) - np.asarray(b)
    byte_counts = np.asarray(metadata['byte_counts'], dtype=float)
    if delta.shape != byte_counts.shape or not np.isfinite(delta).all():
        raise ValueError('Invalid paired losses')
    rng = np.random.default_rng(20261112)
    samples = np.zeros(replicates)
    estimate = 0.
    for domain in PROFILE_DOMAINS:
        ids = np.flatnonzero(np.asarray(metadata['domains']) == domain)
        draws = rng.choice(ids, size=(replicates, len(ids)))
        estimate += delta[ids].sum() / (byte_counts[ids].sum() * math.log(2)) / 9
        samples += delta[draws].sum(1) / (byte_counts[draws].sum(1) * math.log(2)) / 9
    return {'delta_bpb': float(estimate), 'ci95': np.quantile(samples, [.025, .975]).tolist()}


def verify(spec, result, split, expected):
    if result['evaluation_sha256'] != digest(spec):
        raise ValueError('Evaluation specification changed')
    metadata = spec['panels'][split]
    panel = [{'domain': domain, 'continuation_bytes': count}
             for domain, count in zip(metadata['domains'], metadata['byte_counts'])]
    for row in result['rows']:
        if row['panel_sha256'] != spec['panel_hashes'][split]:
            raise ValueError('Context changed')
        recomputed = summarize_candidate(row['index'], row['config'], panel,
            row['transferred_nats'], row['canonical_nats'])
        for key in ['mean_domain_bpb', 'mean_domain_canonical_bpb', 'mean_domain_gap_bpb',
                    'pooled_bpb', 'pooled_canonical_bpb', 'pooled_gap_bpb']:
            if abs(row[key] - recomputed[key]) > 1e-10:
                raise ValueError('Saved aggregate does not match per-document losses')
        if row['domains'] != recomputed['domains']:
            raise ValueError('Domain aggregates changed')
        if row['maximum_positions'] > 128:
            raise ValueError('Exceeded position cap')
        if row['tokenizer_sha256'] != spec['tokenizers'][row['index']]['sha256']:
            raise ValueError('Native tokenizer changed')
    return rank_results(result['rows'], expected)


def plot(screen, confirmation, exposures):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    # Same Manrope typeface and colors as the established experiment figures.
    font_path = ROOT / 'monitoring/web/Manrope.ttf'
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({'font.family': 'Manrope' if font_path.exists() else 'DejaVu Sans',
        'font.size': 9, 'axes.titlesize': 11, 'axes.labelsize': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'text.color': '#333333', 'axes.labelcolor': '#333333', 'pdf.fonttype': 42})
    colors = {'bpe': '#2F8FD0', 'wordpiece': '#8657BC', 'unigram': '#36967B'}
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5))
    fig.subplots_adjust(left=.09, right=.97, top=.89, bottom=.10, wspace=.31, hspace=.45)
    ax = axes[0, 0]
    for algorithm, color in colors.items():
        vals = sorted(r['mean_domain_gap_bpb'] for r in screen if r['config']['algorithm'] == algorithm)
        ax.plot(vals, np.arange(1, len(vals) + 1) / len(vals), color=color, label=f'{algorithm} (n={len(vals)})')
    ax.set(xlabel='Mean domain transfer gap (BPB)', ylabel='Fraction of tokenizer family', title='All 1,000 training-pool tokenizers')
    ax.legend(frameon=False)
    ax.grid(alpha=.13)
    ax = axes[0, 1]
    selected = screen[:10]
    confirmed = {r['index']: r for r in confirmation}
    y = np.arange(len(selected))
    ax.barh(y - .16, [r['mean_domain_gap_bpb'] for r in selected], height=.3,
            color='#2F8FD0', label='Screen · 288 passages')
    ax.barh(y + .16, [confirmed[r['index']]['mean_domain_gap_bpb'] for r in selected], height=.3,
            color='#8657BC', label='Independent check · 288 passages')
    ax.set_yticks(y, [f"{r['index']:04d} · {r['config']['algorithm']}" for r in selected])
    ax.invert_yaxis()
    ax.set(xlabel='Mean domain transfer gap (BPB)', title='Ten worst screen candidates')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis='x', alpha=.13)
    ax = axes[1, 0]
    top = screen[:15]
    image = ax.imshow([[r['domains'][d]['gap_bpb'] for d in PROFILE_DOMAINS] for r in top],
                       aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xticks(range(9), [DOMAIN_NAMES[d] for d in PROFILE_DOMAINS], rotation=45, ha='right')
    ax.set_yticks(range(len(top)), [f"{r['index']:04d}" for r in top])
    ax.set(title='Where the worst candidates struggle', ylabel='Tokenizer index · screen rank order')
    fig.colorbar(image, ax=ax, fraction=.04, pad=.025, label='Transfer gap (BPB)')
    ax = axes[1, 1]
    for algorithm, color in colors.items():
        rows = [r for r in screen if r['config']['algorithm'] == algorithm]
        ax.scatter([int(exposures[r['index']]['rl_updates']) for r in rows],
            [r['mean_domain_gap_bpb'] for r in rows], s=13, alpha=.40, color=color, label=algorithm)
    ax.set(xlabel='Times selected during 20k RL updates', ylabel='Mean domain transfer gap (BPB)',
           title='Training exposure and final transfer')
    ax.grid(alpha=.13)
    ax.legend(frameon=False)
    fig.suptitle('RL20k · exhaustive training-tokenizer pool evaluation', fontsize=17, y=.97)
    fig.text(.09, .925, '1,000 native vocabularies · identical held-out text · 9 domains · next 32 canonical tokens · ≤128 positions', fontsize=10, color='#73808E')
    fig.text(.09, .025, 'Higher gap is worse. Domains have equal weight; BPB within each domain is weighted by scored bytes.\nCandidates selected on screen passages are checked on separate documents. One model-training seed; exposure correlations are descriptive.', fontsize=8, color='#73808E')
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / f'pool_sweep.{ext}', dpi=180)
    plt.close(fig)


def main():
    spec = read(STATE / 'spec.json')
    screen_file = read(STATE / 'screen.json')
    confirm_file = read(STATE / 'confirmation.json')
    screen = verify(spec, screen_file, 'screen', range(1000))
    selected = confirm_file['selected_by_screen']
    confirmation = verify(spec, confirm_file, 'confirm', selected)
    screen_ids = set(spec['panels']['screen']['document_ids'])
    if screen_ids.intersection(spec['panels']['confirm']['document_ids']):
        raise ValueError('Screen and confirmation source IDs overlap')
    exposures_path = ROOT / 'reports/rl-tokenizer-selection-20k/data/policy/exposures.csv'
    with exposures_path.open() as stream:
        exposures = {int(r['index']): r for r in csv.DictReader(stream)}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, value in [('spec.json', spec), ('screen.json', screen_file),
                        ('confirmation.json', confirm_file)]:
        (OUT / name).write_text(json.dumps(value, indent=2) + '\n')
    fields = ['rank', 'index', 'algorithm', 'pretokenizer', 'vocab_size', 'rl_updates',
              'uniform_updates', 'final_rl_probability', 'mean_domain_bpb',
              'mean_domain_canonical_bpb', 'mean_domain_gap_bpb', 'pooled_bpb']
    fields += [d + '_gap_bpb' for d in PROFILE_DOMAINS]
    with (OUT / 'ranking.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields, lineterminator='\n')
        writer.writeheader()
        for rank, row in enumerate(screen, 1):
            export = {k: row[k] for k in fields if k in row}
            export.update({k: row['config'][k] for k in ['algorithm', 'pretokenizer', 'vocab_size']})
            export.update({k: exposures[row['index']][k] for k in ['rl_updates', 'uniform_updates', 'final_rl_probability']})
            export.update({d + '_gap_bpb': row['domains'][d]['gap_bpb'] for d in PROFILE_DOMAINS})
            export['rank'] = rank
            writer.writerow(export)
    winner = screen[0]
    confirmed = {r['index']: r for r in confirmation}
    check_winner = confirmed[winner['index']]
    comparisons = []
    for row in confirmation:
        if row['index'] == winner['index']:
            continue
        comparisons.append({'index': row['index'],
            **paired_interval(check_winner['transferred_nats'], row['transferred_nats'], spec['panels']['confirm'])})
    analysis = {'screen_worst_index': winner['index'], 'confirmation_worst_index': confirmation[0]['index'],
        'screen_winner_confirmation_gap': paired_interval(check_winner['transferred_nats'], check_winner['canonical_nats'], spec['panels']['confirm']),
        'confirmation_comparisons_against_screen_winner': comparisons,
        'uncertainty': 'Paired document bootstrap stratified by domain. Fixed selected candidates, one training seed. Intervals are not adjusted for multiple comparisons.'}
    (OUT / 'analysis.json').write_text(json.dumps(analysis, indent=2) + '\n')
    plot(screen, confirmation, exposures)
    config = winner['config']
    lines = ['# RL20k: all 1,000 training-pool tokenizers', '',
        f"The largest screen loss belongs to **tokenizer {winner['index']:04d}**: "
        f"{config['algorithm']}, {config['pretokenizer']}, requested vocabulary {config['vocab_size']:,}. "
        f"It was selected {exposures[winner['index']]['rl_updates']} times during RL training.", '',
        f"Its mean domain BPB is **{winner['mean_domain_bpb']:.5f}**, versus "
        f"**{winner['mean_domain_canonical_bpb']:.5f}** for the same trained LM with canonical inputs: "
        f"a **+{winner['mean_domain_gap_bpb']:.5f} BPB** transfer gap.", '',
        f"On separate confirmation passages, its gap is **{check_winner['mean_domain_gap_bpb']:+.5f} BPB**. "
        f"The largest confirmation loss among the selected candidates belongs to tokenizer **{confirmation[0]['index']:04d}**. "
        'Treat this as a ranking on fixed held-out samples, not a proven unique worst tokenizer for all text.', '',
        '![Pool evaluation](pool_sweep.png)', '',
        '## How the evaluation works', '',
        'We use the final 20,000-step RL-trained language model and input hypernetwork. '
        'For each tokenizer, the hypernetwork reads its entire native vocabulary and produces input embeddings. '
        'The output head and continuation tokenization stay canonical.', '',
        'Every tokenizer sees exactly the same 288 held-out passages: 32 each from English web, code, '
        'Chinese, Japanese, Arabic, Hindi, math, medical English and legal English. '
        'We score only the next 32 canonical tokens after each cut. The total context is at most 128 positions, including BOS. '
        'We shorten each raw prefix jointly until all 1,000 tokenizers fit and preserve the text exactly. '
        'Neither the scored continuation nor the document selection changes between tokenizers.', '',
        'BPB means bits per UTF-8 byte; lower is better. First we compute BPB within each domain, '
        'then average the nine domain scores equally. This prevents a domain with longer byte strings '
        'from dominating the overall result. The canonical baseline is identical for every tokenizer, '
        'so ranking by BPB or by excess BPB gives the same order. The CSV also includes a pooled-byte score.', '',
        'We selected the ten worst overall candidates and the worst candidate in every domain for '
        'a separate 288-passage confirmation set. The sets have disjoint source-document IDs. '
        'Confidence intervals resample paired passages within each domain. They describe uncertainty '
        'over these texts, not different model-training seeds, and are not corrected for multiple comparisons.', '',
        'These shared contexts differ from the earlier OOD and training-sanity panels. '
        'Do not compare their absolute BPB values directly.', '',
        '## Ten largest screen losses', '',
        '| Rank | Tokenizer | Algorithm / pretokenizer | RL updates | BPB | Canonical | Gap | Confirmation gap |',
        '|---:|---:|---|---:|---:|---:|---:|---:|']
    for rank, row in enumerate(screen[:10], 1):
        index = row['index']
        lines.append(f"| {rank} | {index:04d} | {row['config']['algorithm']} / {row['config']['pretokenizer']} | "
            f"{exposures[index]['rl_updates']} | {row['mean_domain_bpb']:.5f} | "
            f"{row['mean_domain_canonical_bpb']:.5f} | {row['mean_domain_gap_bpb']:+.5f} | "
            f"{confirmed[index]['mean_domain_gap_bpb']:+.5f} |")
    lines += ['', '## Worst by domain', '', '| Domain | Tokenizer | BPB | Canonical | Gap |', '|---|---:|---:|---:|---:|']
    for domain in PROFILE_DOMAINS:
        row = max(screen, key=lambda r: r['domains'][domain]['bpb'])
        score = row['domains'][domain]
        lines.append(f"| {DOMAIN_NAMES[domain]} | {row['index']:04d} | {score['bpb']:.5f} | {score['canonical_bpb']:.5f} | {score['gap_bpb']:+.5f} |")
    lines += ['', '## Files and reproduction', '',
        '* [ranking.csv](ranking.csv): all 1,000 screen ranks, settings, training exposures and domain scores.',
        '* [screen.json](screen.json), [confirmation.json](confirmation.json): all evaluated per-passage losses and aggregates.',
        '* [spec.json](spec.json): checkpoint/tokenizer hashes, settings, document IDs, counts and context hashes.',
        '* [analysis.json](analysis.json): confirmation comparisons and uncertainty.',
        '* [PDF plot](pool_sweep.pdf).', '',
        'Run `python scripts/analyze_rl_pool.py` from the source repository with the saved metadata '
        'under `modal_results/rl_pool_20k/` to verify and regenerate this report. '
        'The inference entry point is `scripts/modal_eval_rl_pool.py`; the frozen design is '
        '`plans/rl-all-pool-20k.md`.', '',
        'Raw text, tokenizer files and weights remain in the persistent Modal Volume '
        '`blackpantherx86 / anytok-rl-all-pool-20k`. The shared raw-text panel is `panel.json`; '
        'full features are `features/NNNN/`; individual scores are `screen/NNNN.json` '
        'and `confirm/NNNN.json`. No large artifacts were downloaded through the local caller.', '']
    (OUT / 'README.md').write_text('\n'.join(lines))
    print(json.dumps(analysis, indent=2))


if __name__ == '__main__':
    main()
