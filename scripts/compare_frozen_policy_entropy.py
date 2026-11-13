"""Same full reward table and policy seed, entropy 0 versus 0.01."""
from pathlib import Path

import numpy as np


def plot_comparison(baseline, current, refs, path):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import PercentFormatter
    root = Path(__file__).resolve().parents[1];font = root / 'monitoring/web/Manrope.ttf'
    if font.exists():
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': 'Manrope' if font.exists() else 'DejaVu Sans', 'font.size': 9,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True)
    for curve, color, label in [(baseline, '#8657BC', 'Entropy 0.01'), (current, '#36967B', 'Entropy 0')]:
        x = [r['step'] for r in curve]
        axes[0, 0].plot(x, [r['screen']['expected_bpb'] - refs['screen']['bpb'].mean() for r in curve], color=color, label=label)
        axes[0, 1].plot(x, [r['screen']['top10_mass'] for r in curve], color=color, label=label)
        axes[1, 0].plot(x, [r['screen']['top1_mass'] for r in curve], color=color, label=label)
        axes[1, 1].plot(x, [r['confirm']['expected_bpb'] - refs['confirm']['canonical_bpb'] for r in curve], color=color, label=label)
    axes[0, 0].axhline(float(refs['screen']['bpb'].max() - refs['screen']['bpb'].mean()), color='#87919D', ls=':', label='True highest reward')
    axes[0, 0].set(title='Does removing entropy improve hard-tokenizer selection?', ylabel='Reward-text expected BPB above uniform')
    axes[0, 1].axhline(.01, color='#87919D', ls=':', label='Uniform: 1%')
    axes[0, 1].set(title='Probability on the hardest ten reward-text tokenizers', ylabel='Selection probability', ylim=(0, 1.05))
    axes[1, 0].axhline(.001, color='#87919D', ls=':', label='Uniform: 0.1%')
    top1_limit = max(.02, 1.1 * max(r['screen']['top1_mass'] for curve in [baseline, current] for r in curve))
    axes[1, 0].set(title='Probability on the exact worst tokenizer #0635', ylabel='Selection probability', ylim=(0, top1_limit))
    for ax in [axes[0, 1], axes[1, 0]]:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[1, 1].axhline(float(refs['confirm']['bpb'].mean() - refs['confirm']['canonical_bpb']), color='#87919D', ls=':', label='Uniform')
    axes[1, 1].set(title='Do harder selections persist on held-out check text?', ylabel='Expected transfer gap · BPB')
    for ax in axes.flat:
        ax.set_xlabel('Policy updates');ax.legend(frameon=False, fontsize=8)
        ax.grid(axis='y', color='#E9EDF1', lw=.6);ax.set_axisbelow(True)
    fig.suptitle('Entropy ablation · same 1,000 tokenizers, 288 + 288 passages and initialization', fontsize=13)
    for suffix in ['png', 'pdf']:
        target = path.with_suffix('.' + suffix);target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.stem + '.tmp.' + suffix)
        fig.savefig(temporary, dpi=170, facecolor='white');temporary.replace(target)
    plt.close(fig)
