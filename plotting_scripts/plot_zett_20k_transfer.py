"""Plot the 1k ZeTT checkpoint grid from the saved Modal evaluation results.

Refresh local inputs and plot all completed steps:
    python plotting_scripts/plot_zett_20k_transfer.py --sync
Plot the first ten checkpoints from already downloaded inputs:
    python plotting_scripts/plot_zett_20k_transfer.py --max-step 10000

Outputs: plots/zett_20k_transfer.{png,pdf,csv,json}.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'results/zett_20k_eval'
OUTPUT = ROOT / 'plots/zett_20k_transfer'
PROFILES = ('qingy1337', 'code-group2-evcc')
INDICES = (0, 4, 8, 12, 16, 20, 24, 28)
LABELS = ('Base', 'Code50', 'Papers25', 'Math50', 'QA25', 'Books25',
          'Medical40', 'Legal40', 'Spanish35', 'Chinese35', 'Japanese35')
EXPECTED = (1936, 235503, 1088789)
BLUE, ORANGE, PURPLE = '#2F8FD0', '#D86B2D', '#8657BC'
GRAY, INK, GRID = '#73808E', '#333333', '#DCE1E5'


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sync_profile(profile, dest):
    dest.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'MODAL_PROFILE': profile}
    subprocess.run(['modal', 'volume', 'get', '--force', 'anytok-zett-grid-eval',
                    '/corrected-v2', str(dest)], check=True, env=env)


def score_counts(score):
    return score['tokens'], score['bytes']


def load_data(source, max_step):
    rows, provenance = [], []
    panel_identity = None
    tokenizer_identity = None
    for step in range(1000, max_step + 1, 1000):
        profile = PROFILES[0] if step // 1000 % 2 else PROFILES[1]
        directory = source / profile / 'corrected-v2' / f'step-{step:06d}'
        complete = directory / 'complete.json'
        if not complete.is_file():
            continue
        marker = read_json(complete)
        if marker['step'] != step or tuple(marker['indices']) != INDICES:
            raise ValueError(f'Unexpected checkpoint marker at {step}')
        name = f'step-{step:06d}'
        zdir, odir = directory / 'zett32' / name, directory / 'ood48' / name
        meta, baseline = read_json(zdir / 'metadata.json'), read_json(zdir / 'baselines.json')
        ometa = read_json(odir / 'metadata.json')
        base = read_json(directory / 'base64' / f'{name}.json')
        identity = (meta['corpus_sha256'], meta['examples_sha256'], meta['specs_sha256'])
        if panel_identity is None:
            panel_identity = identity
        if identity != panel_identity or tuple(meta['selected_indices']) != INDICES:
            raise ValueError(f'Panel or sampled tokenizer selection changed at {step}')
        if (ometa['corpus_sha256'], ometa['examples_sha256']) != identity[:2]:
            raise ValueError(f'48k evaluation panel differs at {step}')
        if (ometa['documents'], ometa['continuation_tokens'], ometa['continuation_bytes']) != EXPECTED:
            raise ValueError(f'48k panel counts differ at {step}')
        bm = base['metadata']
        if (bm['corpus_sha256'], bm['examples_sha256']) != identity[:2]:
            raise ValueError(f'64k evaluation panel differs at {step}')
        if (bm['documents'], bm['continuation_tokens'], bm['continuation_bytes']) != EXPECTED:
            raise ValueError(f'64k panel counts differ at {step}')
        tok_identity = (tuple(sorted(ometa['tokenizer_sha256'].items())), bm['tokenizer_sha256'])
        if tokenizer_identity is None:
            tokenizer_identity = tok_identity
        if tok_identity != tokenizer_identity:
            raise ValueError(f'BPE tokenizer files changed at {step}')
        if baseline['examples'] != EXPECTED[0]:
            raise ValueError(f'Document count changed at {step}')
        for kind in ('original', 'trained'):
            if score_counts(baseline[kind]['overall']) != EXPECTED[1:]:
                raise ValueError(f'Canonical scored counts changed at {step}')
        canonical = baseline['trained']['overall']['bpb']
        original = baseline['original']['overall']['bpb']
        if abs(base['canonical_bpb'] - canonical) > 1e-9 or score_counts(base['score']) != EXPECTED[1:]:
            raise ValueError(f'64k canonical baseline or counts differ at {step}')
        zrows, orows = read_jsonl(zdir / 'tokenizers.jsonl'), read_jsonl(odir / 'results.jsonl')
        if len(zrows) != len(INDICES) or {r['index'] for r in zrows} != set(INDICES):
            raise ValueError(f'Incomplete ZeTT tokenizers at {step}')
        if len(orows) != len(LABELS) or {r['label'] for r in orows} != set(LABELS):
            raise ValueError(f'Incomplete 48k tokenizers at {step}')
        if any(score_counts(r['overall']) != EXPECTED[1:] for r in zrows):
            raise ValueError(f'ZeTT scored counts differ at {step}')
        if any(score_counts(r['score']) != EXPECTED[1:] for r in orows):
            raise ValueError(f'48k scored counts differ at {step}')
        if any(r['tokenizer_sha256'] != ometa['tokenizer_sha256'][r['label']] for r in orows):
            raise ValueError(f'48k tokenizer hash differs at {step}')
        if not all(math.isfinite(v) for v in [canonical, original, base['score']['bpb']]
                   + [r['overall']['bpb'] for r in zrows] + [r['score']['bpb'] for r in orows]):
            raise ValueError(f'Non-finite BPB at {step}')
        zgap = {r['index']: r['overall']['bpb'] - canonical for r in zrows}
        ogap = {r['label']: r['score']['bpb'] - canonical for r in orows}
        row = {'step': step, 'canonical_bpb': canonical, 'original_canonical_bpb': original,
               'zett_mean_gap_bpb': statistics.mean(zgap.values()),
               'ood48_mean_gap_bpb': statistics.mean(ogap.values()),
               'base64_gap_bpb': base['score']['bpb'] - canonical}
        row.update({f'zett_{i}_gap_bpb': zgap[i] for i in INDICES})
        row.update({f'ood48_{label}_gap_bpb': ogap[label] for label in LABELS})
        rows.append(row)
        provenance.append({'step': step, 'profile': profile,
                           'checkpoint_manifest_sha256': marker['manifest_sha256'],
                           'completion_sha256': hashlib.sha256(complete.read_bytes()).hexdigest()})
    if not rows:
        raise ValueError(f'No complete checkpoints found under {source}; run with --sync')
    originals = [r['original_canonical_bpb'] for r in rows]
    if max(originals) - min(originals) > 1e-8:
        raise ValueError('Original canonical score changed across checkpoints')
    return rows, {'panel_hashes': dict(zip(('corpus_sha256', 'examples_sha256', 'specs_sha256'), panel_identity)),
                  'documents': EXPECTED[0], 'continuation_tokens': EXPECTED[1],
                  'continuation_bytes': EXPECTED[2], 'zett_indices': INDICES,
                  'ood_tokenizer_sha256': dict(tokenizer_identity[0]),
                  'base64_tokenizer_sha256': tokenizer_identity[1], 'checkpoints': provenance}


def apply_style():
    font_dir = Path.home() / '.local/share/fonts/manrope'
    if not font_dir.is_dir():
        raise FileNotFoundError(f'Manrope font not found: {font_dir}')
    for path in font_manager.findSystemFonts([str(font_dir)]):
        font_manager.fontManager.addfont(path)
    plt.rcParams.update({'font.family': 'Manrope', 'font.size': 9,
        'axes.titlesize': 10.5, 'axes.labelsize': 9, 'legend.fontsize': 7.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#555B65', 'axes.labelcolor': INK,
        'xtick.color': INK, 'ytick.color': INK, 'text.color': INK,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'pdf.fonttype': 42, 'ps.fonttype': 42})


def draw(rows, output):
    apply_style()
    # Leave gaps in lines where a checkpoint hasn't finished yet.
    steps = list(range(1000, max(r['step'] for r in rows) + 1, 1000))
    by_step = {r['step']: r for r in rows}
    def series(key):
        return [by_step[s][key] if s in by_step else float('nan') for s in steps]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.subplots_adjust(left=.085, right=.97, top=.87, bottom=.10, hspace=.39, wspace=.26)
    a = axes[0, 0]
    a.plot(steps, series('canonical_bpb'), color=BLUE, lw=2.3, marker='o', ms=4, label='Trained canonical')
    a.axhline(rows[0]['original_canonical_bpb'], color=GRAY, ls='--', lw=1.4, label='Original canonical')
    a.set_title('Canonical language model', loc='left')
    a.set_ylabel('Bits per byte  ↓')
    a.legend(frameon=False, loc='best')
    a = axes[0, 1]
    zcolors = [plt.get_cmap('tab10')(i) for i in range(len(INDICES))]
    for i, color in zip(INDICES, zcolors):
        a.plot(steps, series(f'zett_{i}_gap_bpb'), color=color, lw=1.2, marker='o', ms=2.4, label=f'#{i}')
    a.plot(steps, series('zett_mean_gap_bpb'), color=INK, lw=2.3, label='Mean')
    a.set_title('8 sampled ZeTT tokenizers', loc='left')
    a.legend(frameon=False, ncol=3, loc='upper left', fontsize=7)
    a = axes[1, 0]
    palette = [plt.get_cmap('tab20')(i) for i in range(len(LABELS))]
    for label, color in zip(LABELS, palette):
        a.plot(steps, series(f'ood48_{label}_gap_bpb'), color=color, lw=1.2, label=label)
    a.plot(steps, series('ood48_mean_gap_bpb'), color=INK, lw=2.3, label='Mean')
    a.set_title('11 heldout 48k tokenizers', loc='left')
    a.legend(frameon=False, ncol=3, loc='lower left', fontsize=6.6)
    a = axes[1, 1]
    a.plot(steps, series('ood48_Base_gap_bpb'), color=BLUE, lw=2, marker='o', ms=4, label='Base 48k')
    a.plot(steps, series('base64_gap_bpb'), color=PURPLE, lw=2, marker='o', ms=4, label='Base 64k')
    a.set_title('Base vocabulary size', loc='left')
    a.legend(frameon=False, loc='best')
    for a in (axes[0, 1], axes[1, 0], axes[1, 1]):
        a.axhline(0, color=GRAY, lw=.7, alpha=.7)
        a.set_ylabel('Transfer − canonical BPB  ↓')
    axes[1, 0].set_ylim(bottom=0.08)
    for a in axes.flat:
        a.set_xlabel('Joint-training updates after identity warmup')
        a.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
        a.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1000:g}k'))
        a.grid(axis='y', color=GRID, lw=.65)
        a.tick_params(length=3, width=.6)
    fig.text(.085, .965, 'Tokenizer transfer across training', fontsize=17, fontweight='bold', va='top')
    fig.text(.085, .925, f'FineWeb-Edu · {len(rows)} checkpoints · 8 ZeTT, 11 heldout 48k, and Base 64k tokenizers', fontsize=9, color=GRAY, va='top')
    fig.text(.085, .035, '1,936 documents · up to 128 canonical continuation tokens after each cut · identical scored bytes · canonical panel uses its own scale', fontsize=8, color=GRAY)
    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(output.with_suffix(f'.{ext}'), dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sync', action='store_true', help='Download the latest results from both Modal profiles')
    parser.add_argument('--input', type=Path, default=INPUT)
    parser.add_argument('--output', type=Path, default=OUTPUT, help='Output path without extension')
    parser.add_argument('--max-step', type=int, default=20000)
    args = parser.parse_args()
    if not 1000 <= args.max_step <= 20000 or args.max_step % 1000:
        parser.error('--max-step must be a multiple of 1000 between 1000 and 20000')
    if args.sync:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(sync_profile, p, args.input / p) for p in PROFILES]
            for future in futures:
                future.result()
    rows, provenance = load_data(args.input, args.max_step)
    draw(rows, args.output)
    with args.output.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    provenance.update({'max_requested_step': args.max_step, 'completed_steps': [r['step'] for r in rows],
                       'continuation_limit_tokens': 128,
                       'measurement': 'generated input embeddings on prefix; fixed canonical output head; gap against trained canonical BPB'})
    args.output.with_suffix('.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f"Plotted {len(rows)} checkpoints through {rows[-1]['step']:,}: {args.output.with_suffix('.png')}")
    missing = sorted(set(range(1000, rows[-1]['step'] + 1, 1000)) - {r['step'] for r in rows})
    if missing:
        print(f'Missing completed checkpoints (shown as gaps): {missing}')


if __name__ == '__main__':
    main()
