"""Plot the saved held-out 48k comparison, separating LM and transfer changes."""
import json
from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from plot_joint_hypernetwork_learning import style

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = json.loads((ROOT / 'modal_results/pool_ood_48k/report.json').read_text())
    rows = report['rows']; steps = [r['step'] / 1000 for r in rows]
    style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
    axes[0].plot(steps, [r['canonical_bpb'] for r in rows], 'o-', color='#2F8FD0', label='Trained canonical')
    axes[0].plot(steps, [r['mean_bpb'] for r in rows], 'o-', color='#8657BC', label='Mean held-out transfer')
    axes[0].set(title='Absolute loss', ylabel='Bits per byte (lower is better)')
    colors = ['#2F8FD0', '#E78042', '#8657BC', '#55A868', '#D05879', '#9F7955',
              '#59A9A7', '#BD9C36', '#6975B9', '#B16CB2', '#8B919A']
    for color, name in zip(colors, rows[0]['tokenizers']):
        axes[1].plot(steps, [r['tokenizers'][name]['gap_bpb'] for r in rows], 'o-',
                     color=color, linewidth=1.2, markersize=3, label=name)
    axes[1].plot(steps, [r['mean_gap_bpb'] for r in rows], color='#333333', linewidth=2, label='Mean')
    axes[1].axhline(0, color='#999999', linewidth=.6)
    axes[1].set(title='Transfer gap versus trained canonical', ylabel='Excess bits per byte')
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False, ncol=3, fontsize=6.3, loc='upper right')
    for ax in axes:
        ax.set_xlabel('Training steps (thousands)'); ax.set_xticks(steps)
        ax.yaxis.set_major_locator(MaxNLocator(5)); ax.grid(axis='y', alpha=.13)
    fig.suptitle('Uniform pool training · 11 unseen 48k BPE tokenizers', fontsize=12)
    fig.text(.5, .015, '256 fixed documents · 32 scored suffix tokens each · BOS + prefix + suffix ≤128 positions',
             ha='center', fontsize=8, color='#73808E')
    fig.tight_layout(rect=(0, .045, 1, .94))
    output = ROOT / 'plots/pool_ood_48k'; output.parent.mkdir(exist_ok=True)
    for suffix in ('.png', '.pdf'):
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=suffix, delete=False) as temporary:
            temp = Path(temporary.name)
        fig.savefig(temp, dpi=180); temp.replace(output.with_suffix(suffix))
    plt.close(fig)


if __name__ == '__main__': main()
