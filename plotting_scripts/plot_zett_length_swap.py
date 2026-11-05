"""Collect and plot the matched length and hypernetwork/LM swap diagnostic.

    python plotting_scripts/plot_zett_length_swap.py --sync
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
import os
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'results/zett_length_swap'
OUTPUT = ROOT / 'plots/zett_length_swap'
CAPS = (128, 512, 2048, 8192)
PAIRS = ((5000, 5000), (10000, 10000), (20000, 20000), (5000, 20000), (20000, 5000))
INDICES = (0, 4, 8, 12, 16, 20, 24, 28)
PROFILES = {'qingy1337': (128, 2048), 'code-group2-evcc': (512, 8192)}


def sync(profile, directory):
    directory.mkdir(parents=True, exist_ok=True)
    subprocess.run(['modal', 'volume', 'get', '--force', 'anytok-zett-length-swap', '/', str(directory)],
                   env={**os.environ, 'MODAL_PROFILE': profile}, check=True)


def load(source):
    rows, identities = [], {}
    for profile, caps in PROFILES.items():
        for cap in caps:
            for hn, lm in PAIRS:
                folder = source / profile / f'positions-{cap:04d}' / f'hn-{hn:06d}_lm-{lm:06d}'
                marker = json.loads((folder / 'complete.json').read_text())
                if (marker['position_cap'], marker['hypernetwork_step'], marker['lm_step'], tuple(marker['indices'])) != (cap, hn, lm, INDICES):
                    raise ValueError(f'Incorrect completion marker: {folder}')
                meta = json.loads((folder / 'metadata.json').read_text())
                summary = json.loads((folder / 'summary.json').read_text())
                baseline = json.loads((folder / 'baselines.json').read_text())
                vals = [json.loads(x) for x in (folder / 'tokenizers.jsonl').read_text().splitlines() if x]
                if (tuple(meta['selected_indices']) != INDICES or summary['tokenizers'] != 8
                        or len(vals) != 8 or {v['index'] for v in vals} != set(INDICES)
                        or baseline['examples'] != 256):
                    raise ValueError(f'Incomplete diagnostic: {folder}')
                if any((v['overall']['tokens'], v['overall']['bytes']) != (8192, 38534) for v in vals):
                    raise ValueError(f'Mismatched continuation counts: {folder}')
                trained = baseline['trained']['overall']
                if (trained['tokens'], trained['bytes']) != (8192, 38534):
                    raise ValueError(f'Mismatched canonical baseline: {folder}')
                key = (cap, lm)
                if key in identities and abs(identities[key] - trained['bpb']) > 1e-7:
                    raise ValueError(f'Canonical baseline changed between swaps: {folder}')
                identities[key] = trained['bpb']
                if not math.isfinite(summary['transferred_mean_bpb']):
                    raise ValueError(f'Non-finite result: {folder}')
                rows.append({'positions': cap, 'hypernetwork_step': hn, 'lm_step': lm,
                             'canonical_bpb': trained['bpb'],
                             'transfer_bpb': summary['transferred_mean_bpb'],
                             'gap_bpb': summary['mean_transfer_gap_vs_trained_canonical_bpb'],
                             'corpus_sha256': meta['corpus_sha256'],
                             'examples_sha256': meta['examples_sha256'],
                             'hypernetwork_manifest_sha256': marker['hypernetwork_manifest_sha256'],
                             'lm_manifest_sha256': marker['lm_manifest_sha256']})
    for cap in CAPS:
        group = [r for r in rows if r['positions'] == cap]
        if len({(r['corpus_sha256'], r['examples_sha256']) for r in group}) != 1:
            raise ValueError(f'Panel mismatch at position cap {cap}')
    return sorted(rows, key=lambda r: (r['positions'], r['hypernetwork_step'], r['lm_step']))


def draw(rows, output):
    font_dir = Path.home() / '.local/share/fonts/manrope'
    for f in font_manager.findSystemFonts([str(font_dir)]):
        font_manager.fontManager.addfont(f)
    plt.rcParams.update({'font.family': 'Manrope', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'text.color': '#333333',
                         'axes.labelcolor': '#333333', 'pdf.fonttype': 42})
    grid = {(r['positions'], r['hypernetwork_step'], r['lm_step']): r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6))
    fig.subplots_adjust(left=.09, right=.97, top=.78, bottom=.19, wspace=.27)
    for step, color in ((5000, '#2F8FD0'), (10000, '#D86B2D'), (20000, '#8657BC')):
        axes[0].plot(range(4), [grid[(c, step, step)]['gap_bpb'] for c in CAPS],
                     color=color, marker='o', lw=2, label=f'{step//1000}k matched')
    axes[0].set_title('Matched checkpoints', loc='left')
    axes[0].set_ylabel('Mean transfer − canonical BPB  ↓')
    axes[0].legend(frameon=False)
    for hn, color in ((5000, '#2F8FD0'), (20000, '#8657BC')):
        for lm, style in ((5000, '-'), (20000, '--')):
            axes[1].plot(range(4), [grid[(c, hn, lm)]['gap_bpb'] for c in CAPS],
                         color=color, ls=style, marker='o', lw=1.9,
                         label=f'{hn//1000}k HN / {lm//1000}k LM')
    axes[1].set_title('Hypernetwork and LM swaps', loc='left')
    axes[1].set_ylabel('Mean transfer − that LM’s canonical BPB  ↓')
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xticks(range(4), ['128', '512', '2,048', '8,192'])
        ax.set_xlabel('Maximum positions: prefix + 32-token continuation + BOS')
        ax.axhline(0, color='#73808E', lw=.8)
        ax.grid(axis='y', color='#DCE1E5', lw=.7)
    fig.text(.09, .95, 'Does transfer degrade with longer context?', fontsize=16, fontweight='bold', va='top')
    fig.text(.09, .885, 'Same 256 documents and continuation tokens · 8 sampled ZeTT tokenizers', fontsize=9, color='#73808E', va='top')
    fig.text(.09, .055, 'Panels differ in prefix length; each swap uses the canonical baseline of its LM. Lower gaps are better.', fontsize=8, color='#73808E')
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(output.with_suffix(f'.{suffix}'), dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sync', action='store_true')
    parser.add_argument('--input', type=Path, default=INPUT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.sync:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(sync, p, args.input / p) for p in PROFILES]
            for f in futures:
                f.result()
    rows = load(args.input)
    draw(rows, args.output)
    with args.output.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} diagnostic cells to {args.output}.png')


if __name__ == '__main__':
    main()
