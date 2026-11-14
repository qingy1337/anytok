"""Plot representative memory trials; retain every attempt in results.json."""
from pathlib import Path

SMALL = [
    ('baseline-43k', 'Original'),
    ('optimizer-only-43k', 'Adam on CPU'),
    ('fine-43k', 'Recompute'),
    ('fine-offload-43k', 'Recompute + RAM'),
    ('split-norm-fixed-43k-expand', 'Split norms + RAM'),
    ('split-disk-fixed-43k-host80-expand', 'RAM + NVMe'),
    ('split-disk-fixed-43k-host80-optimizer-expand', 'RAM + NVMe + Adam'),
]
LARGE = [
    ('baseline-128k', 'Original'),
    ('fine-offload-128k-expand', 'Recompute + RAM'),
    ('split-norm-fixed-128k-expand', 'Split norms + RAM'),
    ('split-disk-fixed-128k-host80-expand', 'RAM + NVMe'),
    ('split-disk-fixed-128k-host80-optimizer-expand', 'RAM + NVMe + Adam'),
]


def plot(rows, path):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    root = Path(__file__).resolve().parents[1]
    font_manager.fontManager.addfont(str(root / 'monitoring/web/Manrope.ttf'))
    plt.rcParams.update({'font.family': 'Manrope', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    data = {r['id']: r['result'] for r in rows}
    capacity = next((r['inputs']['gpu_total_bytes'] / 2**30 for r in data.values() if 'inputs' in r), 95)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), constrained_layout=True)
    for ax, selection, title, color in [
        (axes[0, 0], SMALL, '43k native vocabulary · complete training step', '#36967B'),
        (axes[0, 1], LARGE, '128k native vocabulary · complete training step', '#8657BC'),
    ]:
        selected = [(name, label) for name, label in selection if name in data]
        for x, (name, _) in enumerate(selected):
            result = data[name]
            if result['outcome'] == 'complete':
                height = result['peak_allocated_bytes'] / 2**30
                ax.bar(x, height, color=color, width=.66)
                ax.text(x, height+1.5, f'{height:.1f}', ha='center', fontsize=9)
            else:
                ax.scatter(x, capacity, marker='x', s=65, color='#BB675A', zorder=4)
                ax.text(x, capacity+2, 'OOM', ha='center', color='#A85349', fontsize=8)
        ax.axhline(capacity, color='#89919C', ls=':', label=f'Local capacity: {capacity:.1f} GiB')
        ax.set_xticks(range(len(selected)), [label for _, label in selected], rotation=25, ha='right', fontsize=8)
        ax.set(title=title, ylabel='Peak allocated GPU memory · GiB', ylim=(0, capacity+15))
        ax.legend(frameon=False, fontsize=8, loc='upper left')
    timings = [(n, label) for n, label in SMALL if n in data and data[n]['outcome'] == 'complete']
    if LARGE[-1][0] in data and data[LARGE[-1][0]]['outcome'] == 'complete':
        timings.append((LARGE[-1][0], '128k: RAM + NVMe + Adam'))
    ax = axes[1, 0]
    for i, (name, _) in enumerate(timings):
        seconds = sum(p['seconds'] for p in data[name]['phases'] if not p['phase'].startswith('load '))
        ax.barh(i, seconds, color='#8657BC' if '128k' in name else '#36967B')
        ax.text(seconds+1.5, i, f'{seconds:.1f}s', va='center', fontsize=8)
    ax.set_yticks(range(len(timings)), [label for _, label in timings], fontsize=8)
    ax.invert_yaxis()
    ax.set(title='Runtime · one measured step per variant', xlabel='Seconds · includes offloading; excludes loading and numeric checks')
    ax.set_xlim(0, max([sum(p['seconds'] for p in data[n]['phases'] if not p['phase'].startswith('load ')) for n, _ in timings] or [1])*1.22)
    ax = axes[1, 1]
    ax.axis('off')
    table = []
    for name, label in [('repeat-43k', 'Original repeated'), *SMALL[1:]]:
        result = data.get(name, {}); numeric = result.get('numeric', {}); gradients = numeric.get('gradients', {})
        if not gradients.get('relative_l2'): continue
        table.append([label, 'Yes' if numeric.get('output_bitwise_equal') else 'No',
                      f"{gradients['relative_l2']:.2g}",
                      'Yes' if numeric.get('cuda_rng_sha256') == numeric.get('reference_numeric', {}).get('cuda_rng_sha256') else 'No'])
    ax.set_title('43k correctness · full outputs + all 349 parameter gradients', pad=15)
    if table:
        t = ax.table(cellText=table, colLabels=['Variant', 'Output exact', 'Gradient ΔL2', 'RNG exact'],
                     bbox=[0, .16, 1, .70], cellLoc='center', colWidths=[.38, .2, .23, .19])
        t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.65)
        for cell in t.get_celld().values(): cell.set_edgecolor('#E1E6EB')
    ax.text(.5, .03, '128k forward: separate full-table check. No original 128k gradient comparison.\nAll 26 attempts, including superseded wrappers and failures: results.json',
            ha='center', va='bottom', fontsize=8, transform=ax.transAxes)
    for ax in axes.flat:
        if ax.axison:
            ax.grid(axis='x' if ax is axes[1, 0] else 'y', color='#E9EDF1', lw=.6); ax.set_axisbelow(True)
    fig.suptitle('Local memory diagnostics · real step-2,000 checkpoint · batch 128 · microbatch 32', fontsize=13)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ['png', 'pdf']:
        target = path.with_suffix('.'+suffix); temporary = target.with_name(target.stem+'.tmp.'+suffix)
        fig.savefig(temporary, dpi=170, facecolor='white'); temporary.replace(target)
    plt.close(fig)
