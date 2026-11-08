"""Show all screened pairs and independent confirmation of useful weaknesses."""
import argparse
import csv
import json
from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
from plot_joint_hypernetwork_learning import style

ROOT = Path(__file__).resolve().parents[1]


def plot_rerun(step):
    report = json.loads((ROOT / f'modal_results/uniform_weakness/reruns/step-{step:06d}/report.json').read_text())
    intermediate = []
    for path in sorted((ROOT / 'modal_results/uniform_weakness/reruns').glob('step-*/report.json')):
        saved = json.loads(path.read_text())
        if report['previous_step'] < saved['step'] < report['step']:
            if saved['previous_report_digest'] != report['previous_report_digest']:
                raise ValueError('Intermediate checkpoint uses a different reference panel')
            intermediate.append(saved)
    reports = [*intermediate, report]
    style()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), sharey=True)
    names = {'ood48-chinese35': 'Chinese35 (48k) · Chinese',
             'ood48-japanese35': 'Japanese35 (48k) · Japanese',
             'base64': 'Base (64k) · Medical English'}
    colors = ['#2F8FD0', '#55A868', '#8657BC']
    steps = [report['previous_step'] / 1000, *[r['step'] / 1000 for r in reports]]
    for ax, row, color in zip(axes, report['rows'], colors):
        scores = [row['comparison']['previous']]
        for saved in reports:
            same = next(r for r in saved['rows'] if r['tokenizer'] == row['tokenizer'] and r['domain'] == row['domain'])
            if (same['context_sha256'] != row['context_sha256']
                    or same['current_scores']['document_ids'] != row['current_scores']['document_ids']
                    or same['current_scores']['continuation_bytes'] != row['current_scores']['continuation_bytes']):
                raise ValueError('Checkpoint contexts or scored documents/bytes differ')
            scores.append(same['comparison']['current'])
        values = [r['gap_bpb'] for r in scores]
        errors = [[r['gap_bpb'] - r['gap_ci95'][0] for r in scores],
                  [r['gap_ci95'][1] - r['gap_bpb'] for r in scores]]
        ax.errorbar(steps, values, yerr=errors, fmt='o-', color=color, capsize=3, linewidth=1.5, markersize=5)
        ax.axhline(.1, color='#D05879', linewidth=.9, linestyle='--')
        ax.set_title(names[row['tokenizer']]); ax.set_xlabel('Training steps (thousands)')
        ax.set_xticks(steps); ax.margins(x=.25); ax.grid(axis='y', alpha=.13)
        change = scores[-1]['gap_bpb'] - scores[-2]['gap_bpb']
        ax.text(.5, .94, f"{steps[-2]:g}k → {steps[-1]:g}k: {change:+.3f} BPB",
                transform=ax.transAxes, ha='center', va='top', fontsize=8)
    axes[0].set_ylabel('Transfer gap versus trained canonical (BPB)')
    axes[0].set_ylim(bottom=0)
    labels = ' · '.join(f'{s:g}k' for s in steps)
    fig.suptitle(f"Fixed tokenizer edge cases · {labels}", fontsize=12)
    fig.text(.5, .025, '96 identical documents per case · 32 scored suffix tokens · ≤128 total positions · paired 95% intervals · dashed line: +0.1 BPB',
             ha='center', fontsize=7, color='#73808E')
    fig.tight_layout(rect=(0, .07, 1, .92))
    output = ROOT / 'plots/uniform_weakness_rerun'
    for suffix in ('.png', '.pdf'):
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=suffix, delete=False) as stream:
            temporary = Path(stream.name)
        fig.savefig(temporary, dpi=180); temporary.replace(output.with_suffix(suffix))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rerun-step', type=int, help='Compare this saved rerun checkpoint with the fixed 12k reference')
    args = parser.parse_args()
    if args.rerun_step is not None: return plot_rerun(args.rerun_step)
    report = json.loads((ROOT / 'modal_results/uniform_weakness/report.json').read_text())
    style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={'width_ratios': [1, 1.3]})
    rows = report['screen']; confirmed = report['confirmation']
    axes[0].scatter([r['utility']['token_reduction_vs_canonical_percent'] for r in rows],
                    [r['score']['gap_bpb'] for r in rows], s=14, color='#9BA5AF', alpha=.45,
                    label='Screening documents')
    axes[0].axvline(10, color='#73808E', linewidth=.8, linestyle=':')
    axes[0].axhline(.1, color='#D05879', linewidth=1, linestyle='--', label='Requested +0.1 BPB')
    for r in confirmed:
        useful = r['classification']['confirmed_weak_and_useful']
        axes[0].scatter(r['utility']['token_reduction_vs_canonical_percent'], r['score']['gap_bpb'],
                        s=27, color='#55A868' if useful else '#2F8FD0', zorder=3)
    axes[0].scatter([], [], s=27, color='#55A868', label='Confirmed gap ≥0.1, token saving ≥10%')
    axes[0].set(title='Loss versus compression', xlabel='Fewer tokens than canonical (%)',
                ylabel='Transfer gap against trained canonical (BPB)')
    axes[0].legend(frameon=False, fontsize=6.5)
    # Illustrative use cases, rather than twelve versions of the same language.
    examples = [('ood48-chinese35', 'web-zh'), ('ood48-japanese35', 'web-ja'),
                ('base64', 'medical-en'), ('ood48-base', 'web-zh'),
                ('qwen25coder', 'web-zh'), ('starcoder2', 'web-zh')]
    selected = [r for key in examples for r in confirmed if (r['tokenizer'], r['domain']) == key]
    labels = ['Chinese35 (48k) · Chinese', 'Japanese35 (48k) · Japanese',
              'Base (64k) · Medical English', 'Base (48k) · Chinese',
              'Qwen2.5-Coder (~152k) · Chinese', 'StarCoder2 (49k) · Chinese']
    for index, r in enumerate(selected):
        score = r['score']; lo, hi = score['gap_ci95']
        color = '#55A868' if r['classification']['confirmed_weak_and_useful'] else '#2F8FD0'
        axes[1].errorbar(score['gap_bpb'], index,
                         xerr=[[max(0, score['gap_bpb'] - lo)], [max(0, hi - score['gap_bpb'])]],
                         fmt='o', color=color, capsize=2, markersize=4)
    axes[1].set_yticks(range(len(selected)), labels[:len(selected)], fontsize=7)
    axes[1].invert_yaxis(); axes[1].axvline(.1, color='#D05879', linewidth=1, linestyle='--')
    axes[1].set(title='Examples · independent confirmation · 95% CI', xlabel='Transfer gap (BPB)')
    for ax in axes: ax.grid(alpha=.12)
    fig.suptitle('Practical tokenizer search · uniform-pool checkpoint 12k', fontsize=12)
    fig.text(.5, .012, 'Native IDs and priors · fixed 32-token suffix · at most 128 total positions · screening and confirmation documents are disjoint',
             ha='center', fontsize=7, color='#73808E')
    fig.tight_layout(rect=(0, .05, 1, .94))
    output = ROOT / 'plots/uniform_weakness'; output.parent.mkdir(exist_ok=True)
    for suffix in ('.png', '.pdf'):
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=suffix, delete=False) as stream:
            temporary = Path(stream.name)
        fig.savefig(temporary, dpi=180); temporary.replace(output.with_suffix(suffix))
    plt.close(fig)
    with output.with_suffix('.csv').open('w') as stream:
        fields = ['tokenizer', 'domain', 'gap_bpb', 'ci_low', 'ci_high', 'canonical_bpb', 'transferred_bpb',
                  'token_reduction_percent', 'documents', 'qualified']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for r in confirmed:
            s = r['score']; writer.writerow({'tokenizer': r['tokenizer'], 'domain': r['domain'],
                'gap_bpb': s['gap_bpb'], 'ci_low': s['gap_ci95'][0], 'ci_high': s['gap_ci95'][1],
                'canonical_bpb': s['canonical_bpb'], 'transferred_bpb': s['bpb'],
                'token_reduction_percent': r['utility']['token_reduction_vs_canonical_percent'],
                'documents': s['documents'], 'qualified': r['classification']['confirmed_weak_and_useful']})


if __name__ == '__main__': main()
