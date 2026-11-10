"""Render fixed-cohort BPB, transfer gaps and late slopes for every saved checkpoint.

    python plotting_scripts/plot_uniform_trajectory.py
"""
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'plotting_scripts'))
from plot_joint_hypernetwork_learning import style, BLUE, PURPLE, GRAY

GREEN = '#36967B'
NAMES = {
    'ood48-chinese35': 'Chinese35 · BPE 48k / Chinese',
    'ood48-japanese35': 'Japanese35 · BPE 48k / Japanese',
    'base64': 'Base BPE 64k / medical English',
    'pool-0416': 'Pool 416 · WordPiece 102k / medical',
    'pool-0822': 'Pool 822 · Unigram 123k / Japanese',
    'pool-0616': 'Pool 616 · BPE 115k / Chinese',
    'starcoder2': 'StarCoder2 · 49k / Japanese',
    'wordpiece-2gb': 'WordPiece 150k† / medical English',
    'unigram-2gb': 'Unigram 150k† / Japanese',
    'qwen25coder': 'Qwen2.5-Coder · 152k† / Chinese',
    'pool-0635': 'Pool 635 · BPE 120k / medical',
    'ood48-base': 'Base BPE 48k / English control',
}


def save(fig, name):
    path = ROOT / 'plots' / name
    fig.savefig(path.with_suffix('.pdf'))
    fig.savefig(path.with_suffix('.png'), dpi=180)
    plt.close(fig)


def series(row, metric):
    return np.asarray([p[metric + '_bpb'] for p in row['points']]), np.asarray([p[metric + '_ci95'] for p in row['points']])


def subtitle(case):
    d = case['details']; utility = case['utility']['token_reduction_vs_canonical_percent']
    text = f"{d['examples']} passages / {d['source_documents']} docs · {utility:.0f}% fewer tokens"
    if case['cohort'] == 'historical': text += ' · original panel'
    elif d['later_passages']: text += ' · later passages'
    if 'training_exposures_at_20k' in case: text += f"\nIn pool: sampled {case['training_exposures_at_20k']} times by 20k"
    elif case['candidate']['vocab_size'] > 128000: text += '\n† Vocabulary outside training range'
    else: text += '\nTokenizer absent from training pool'
    return text


def grid(report, mode):
    late = mode == 'late'; metric = 'transferred' if mode == 'bpb' else 'gap'
    fig, axes = plt.subplots(4, 3, figsize=(14.6, 15.5))
    fig.subplots_adjust(left=.065, right=.98, top=.82, bottom=.14, hspace=.95, wspace=.28)
    if mode == 'bpb':
        title = 'Uniform pool training · absolute continuation BPB'
        desc = 'Transferred input embeddings and each checkpoint’s trained canonical input embeddings'
    elif late:
        title = 'Uniform pool training · is transfer flattening late?'
        desc = 'Zoom: all 11 checkpoints from 15k to 20k · same fixed passages within each panel'
    else:
        title = 'Uniform pool training · transfer gap across all checkpoints'
        desc = 'Transferred BPB − trained canonical BPB at the same checkpoint · lower is better'
    fig.text(.065, .98, title, va='top', fontsize=19, weight='bold')
    fig.text(.065, .943, desc, va='top', fontsize=11)
    handles = [Line2D([], [], color=BLUE, lw=2, label='Transferred input' if mode == 'bpb' else 'Transfer gap')]
    if mode == 'bpb': handles.append(Line2D([], [], color=GRAY, ls='--', lw=1.7, label='Trained canonical input'))
    else: handles.append(Line2D([], [], color=PURPLE, ls=':', lw=1.2, label='+0.10 BPB weakness threshold'))
    handles.append(Patch(facecolor=BLUE, alpha=.12, label='95% source-document cluster interval'))
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.06, .916), ncol=3, frameon=False, fontsize=9)
    for ax, row in zip(axes.flat, report['cases']):
        case = row['case']; x = np.array([p['step'] for p in row['points']]) / 1000
        y, ci = series(row, metric); mask = x >= 15 if late else np.ones(len(x), dtype=bool)
        ax.fill_between(x[mask], ci[mask, 0], ci[mask, 1], color=BLUE, alpha=.12, lw=0)
        ax.plot(x[mask], y[mask], color=BLUE, lw=1.7, marker='o', ms=2.4)
        if mode == 'bpb':
            canonical, _ = series(row, 'canonical')
            ax.plot(x, canonical, color=GRAY, ls='--', lw=1.5)
        else:
            ax.axhline(0, color=GRAY, lw=.7, alpha=.65)
            ax.axhline(.1, color=PURPLE, lw=1, ls=':')
        if late: ax.set_xlim(14.8, 20.2); ax.set_xticks([15, 16, 17, 18, 19, 20])
        else: ax.set_xlim(0, 20.3); ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_title(NAMES[case['tokenizer']], loc='left', pad=42, fontsize=10, weight='bold')
        ax.text(0, 1.045, subtitle(case), transform=ax.transAxes, va='bottom', fontsize=7.6, color=GRAY, linespacing=1.35)
        ax.grid(axis='y', alpha=.13); ax.set_axisbelow(True); ax.set_xlabel('Training step (thousands)', fontsize=8)
        ax.set_ylabel('Continuation BPB' if mode == 'bpb' else 'Gap (BPB)', fontsize=8)
        if late:
            s = row['slopes']['gap']['late']; lo, hi = s['ci95']
            low = ci[mask, 0].min(); high = ci[mask, 1].max(); spread = max(high-low, .005)
            ax.set_ylim(low - .12*spread, high + .37*spread)
            ax.text(.03, .96, f"Late slope {s['slope_bpb_per_1k']:+.4f} / 1k\n95% CI [{lo:+.4f}, {hi:+.4f}]",
                    transform=ax.transAxes, va='top', fontsize=7.3,
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .85, 'pad': 2})
        ax.margins(y=.12)
    fig.text(.065, .065,
             '32 canonical continuation tokens · BOS + prefix + continuation ≤128 positions · canonical output head fixed\n'
             'Original panels: canonical prefix ≤95 tokens. New panels: ≤64. Medical later passages cluster by source article.\n'
             '† 150k/152k vocabularies exceed the training pool’s 32k–128k range. Curves describe one training run.',
             fontsize=9, color=GRAY, linespacing=1.5, va='top')
    save(fig, 'uniform_trajectory_' + mode)


def forest(report):
    fig, ax = plt.subplots(figsize=(13.9, 9.8))
    fig.subplots_adjust(left=.37, right=.81, top=.79, bottom=.22)
    fig.text(.045, .975, 'No strict transfer plateau established by 20k', va='top', fontsize=20, weight='bold')
    fig.text(.045, .921, 'Paired slope fits across all 11 checkpoints from 15k–20k · 95% source-document cluster intervals', fontsize=10.5, va='top')
    handles = [Line2D([], [], color=color, marker=marker, lw=1.5, label=name)
               for name, color, marker in [('Transferred BPB', BLUE, 'o'), ('Canonical BPB', GREEN, 's'), ('Transfer gap', PURPLE, 'D')]]
    handles.append(Patch(facecolor=GRAY, alpha=.12, label='Flatness band: ±2 mBPB / 1k'))
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.038, .884), ncol=4, frameon=False, fontsize=9)
    ax.axvspan(-2, 2, color=GRAY, alpha=.12, lw=0); ax.axvline(0, color=GRAY, lw=.8)
    labels = []
    for i, row in enumerate(report['cases']):
        c = row['case']; p = row['points'][-1]
        labels.append(NAMES[c['tokenizer']])
        if i % 2 == 0: ax.axhspan(i-.48, i+.48, color='#EEF1F4', alpha=.45, lw=0, zorder=-2)
        for metric, color, marker, offset in [('transferred', BLUE, 'o', -.22), ('canonical', GREEN, 's', 0), ('gap', PURPLE, 'D', .22)]:
            s = row['slopes'][metric]['late']; v = s['slope_bpb_per_1k'] * 1000; lo, hi = np.asarray(s['ci95']) * 1000
            ax.errorbar(v, i+offset, xerr=[[v-lo], [hi-v]], color=color, fmt=marker, ms=4.3, lw=1.4, capsize=2)
        weak = p['gap_bpb'] >= .1 and c['utility']['token_reduction_vs_canonical_percent'] >= 10
        label = f"{p['gap_bpb']:+.3f} BPB" + ('  • weak' if weak else '')
        ax.text(1.035, i, label, transform=ax.get_yaxis_transform(), va='center', fontsize=9,
                color=PURPLE if weak else GRAY, weight='bold' if weak else 'normal')
    ax.text(1.035, -.8, '20k transfer gap', transform=ax.get_yaxis_transform(), fontsize=9, weight='bold')
    ax.set_yticks(range(len(labels)), labels); ax.set_ylim(len(labels)-.4, -.6)
    ax.set_xlim(-13.5, 3.2); ax.set_xticks([-12, -10, -8, -6, -4, -2, 0, 2])
    ax.set_xlabel('Slope (milli-BPB per 1,000 training steps)   ← improving', labelpad=10)
    ax.grid(axis='x', alpha=.12); ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0, pad=12)
    fig.text(.045, .105,
             'A strict practical plateau requires the entire purple interval inside the shaded band. None of the seven weak cases qualifies.\n'
             'All seven weak cases still improve in absolute transferred BPB. A residual gap alone does not establish a training wall.\n'
             '† Outside the training vocabulary range. Intervals describe text variation for one training seed; per-case inference.',
             fontsize=9.5, color=GRAY, linespacing=1.5, va='top')
    save(fig, 'uniform_trajectory_plateau')


def main():
    report = json.loads((ROOT / 'modal_results/uniform_trajectories/analysis.json').read_text())
    if report['checkpoint_count'] != 40 or len(report['cases']) != 12: raise ValueError('Expected complete fixed grid')
    style()
    for mode in ['bpb', 'gap', 'late']: grid(report, mode)
    forest(report)
    print('Wrote four PNG/PDF figure pairs for all 40 checkpoints and 12 cases.')


if __name__ == '__main__': main()
