"""Draw the inputs behind the length diagnostic and the 20k transfer plot."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'plots/zett_eval_context_explained'
INK, MUTED = '#263342', '#637486'
BLUE, PURPLE, PALE = '#2F8FD0', '#8657BC', '#E9EDF2'


def main():
    for path in font_manager.findSystemFonts([str(Path.home() / '.local/share/fonts/manrope')]):
        font_manager.fontManager.addfont(path)
    plt.rcParams.update({'font.family': 'Manrope', 'text.color': INK, 'pdf.fonttype': 42})
    with (ROOT / 'plots/zett_length_swap.csv').open() as f:
        length = {(int(r['positions']), int(r['hypernetwork_step']), int(r['lm_step'])): r
                  for r in csv.DictReader(f)}
    with (ROOT / 'plots/zett_20k_transfer.csv').open() as f:
        transfer = {int(r['step']): r for r in csv.DictReader(f)}
    fig = plt.figure(figsize=(14, 9.1), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 100), ylim=(0, 100)); ax.axis('off')
    ax.text(5, 95, 'What goes into the model?', fontsize=22, fontweight='bold', va='top')
    ax.text(5, 88.4, 'Both evaluations score the continuation after the cut. They give the model different amounts of prefix.',
            fontsize=10.5, color=MUTED, va='top')

    def card(y, title, detail, kind, note, start, end):
        ax.add_patch(FancyBboxPatch((5, y), 90, 22, boxstyle='round,pad=0.35,rounding_size=1.2',
                                   facecolor='white', edgecolor='#D8E0E7', linewidth=1))
        ax.text(7.4, y + 19, title, fontsize=13.1, fontweight='bold', va='top')
        ax.text(7.4, y + 15.4, detail, fontsize=9.2, color=MUTED, va='top')
        yy, hh = y + 7.0, 5.1
        ax.add_patch(Rectangle((8, yy), 4.4, hh, facecolor='#F5F7F9', edgecolor='#AAB8C6'))
        ax.text(10.2, yy + hh/2, 'BOS', ha='center', va='center', fontsize=8.1)
        if kind == 'short':
            ax.add_patch(Rectangle((12.4, yy), 26, hh, facecolor=BLUE, alpha=.19, edgecolor=BLUE, linestyle='--'))
            ax.text(25.4, yy + hh/2, 'recent prefix: ≤ 95 tokens', ha='center', va='center', fontsize=9, color=INK)
            split = 38.4
            ax.add_patch(Rectangle((split, yy), 14.5, hh, facecolor=PURPLE, alpha=.82))
            ax.text(split+7.25, yy+hh/2, 'next 32 scored', ha='center', va='center', fontsize=9, color='white')
            ax.add_patch(Rectangle((split+14.5, yy), 15.5, hh, facecolor=PALE, edgecolor='white'))
            ax.text(split+22.25, yy+hh/2, 'rest not fed', ha='center', va='center', fontsize=8.4, color=MUTED)
        else:
            ax.add_patch(Rectangle((12.4, yy), 42, hh, facecolor=BLUE, alpha=.19, edgecolor=BLUE))
            ax.text(33.4, yy + hh/2, 'longer prefix before the cut', ha='center', va='center', fontsize=9, color=INK)
            split = 54.4
            if kind == 'long':
                ax.add_patch(Rectangle((split, yy), 14.5, hh, facecolor=PURPLE, alpha=.82))
                ax.text(split+7.25, yy+hh/2, 'next 32 scored', ha='center', va='center', fontsize=9, color='white')
                ax.add_patch(Rectangle((split+14.5, yy), 15.5, hh, facecolor=PALE, edgecolor='white'))
                ax.text(split+22.25, yy+hh/2, 'rest not fed', ha='center', va='center', fontsize=8.4, color=MUTED)
            else:
                ax.add_patch(Rectangle((split, yy), 30, hh, facecolor=PURPLE, alpha=.82))
                ax.text(split+15, yy+hh/2, 'next up to 128 scored', ha='center', va='center', fontsize=9, color='white')
        ax.plot([split, split], [yy-.5, yy+hh+.5], color=INK, lw=1.15)
        ax.text(split, yy+hh+0.9, 'CUT', fontsize=8.1, fontweight='bold', ha='center', va='bottom')
        ax.text(7.4, y+3.8, note, fontsize=8.5, color=MUTED, va='center')
        ax.text(92.2, y+3.8, f'5k → 20k mean gap: {start:+.4f} → {end:+.4f}',
                fontsize=9, color=INK, fontweight='bold', ha='right', va='center')

    def gap(cap, step):
        return float(length[(cap, step, step)]['gap_bpb'])
    card(61, 'A  ·  Length test: 128-position cap',
         '256 documents · same first 32 continuation tokens at every cap', 'short',
         '1 BOS + prefix + 32 scored tokens ≤ 128. Older prefix text is removed when needed.', gap(128,5000), gap(128,20000))
    card(36, 'B  ·  Length test: 8,192-position cap',
         'Same 256 documents and same 32 scored continuation tokens as A', 'long',
         'More earlier prefix text can fit. The cap is a maximum; actual inputs can be shorter.', gap(8192,5000), gap(8192,20000))
    card(11, 'C  ·  Top right of zett_20k_transfer.pdf',
         '1,936 documents · full saved prefix · first up to 128 continuation tokens scored', 'full',
         'The full prefix plus BOS and scored continuation fits within 8,192 positions.',
         float(transfer[5000]['zett_mean_gap_bpb']), float(transfer[20000]['zett_mean_gap_bpb']))
    ax.text(5, 5.9, 'BLUE = prefix context (not scored)     PURPLE = continuation (scored)     GREY = not fed to the model     ·     bars are schematic',
            fontsize=9, fontweight='bold', color=INK)
    ax.text(5, 2.4, 'A vs B changes prefix context while keeping the scored tokens fixed. C also changes the document set and scoring window.',
            fontsize=8.7, color=MUTED)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT.with_suffix(f'.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(OUT.with_suffix('.png'))


if __name__ == '__main__':
    main()
