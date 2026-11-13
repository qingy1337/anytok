"""Quick pool-composition reference using existing36-tokenizer model scores."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from pool_generalization import file_sha, save_json

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['zett20k', 'uniform20k', 'rl20k']
LABELS = ['ZeTT 20k', 'Uniform pool 20k', 'RL 20k']
COLORS = ['#8657BC', '#2F8FD0', '#36967B']


def inputs():
    pool_path = ROOT / 'reports/rl-pool-sweep-20k/spec.json'
    score_path = ROOT / 'reports/natural-model-comparison-20k/analysis.json'
    pool = json.loads(pool_path.read_text())['tokenizers']
    score = json.loads(score_path.read_text())
    representatives = score['cohort_indices']['representative36']
    return dict(pool=[{k: r[k] for k in ['index', 'vocab_size', 'config']} for r in pool],
        representatives=representatives, hard_indices=score['cohort_indices']['weak_pretokenization'],
        scores=[r for r in score['metrics'] if r['index'] in representatives and r['regime'] == 'context' and r['split'] == 'confirm'],
        source_hashes={str(p.relative_to(ROOT)): file_sha(p) for p in [pool_path, score_path]})


def analyze(data):
    by_index = {r['index']: r for r in data['pool']}
    representatives = [by_index[i] for i in data['representatives']]
    assignments = {}
    for row in data['pool']:
        group = [r for r in representatives if r['config']['algorithm'] == row['config']['algorithm']
                 and r['config']['pretokenizer'] == row['config']['pretokenizer']]
        selected = min(group, key=lambda r: (abs(r['vocab_size'] - row['vocab_size']), r['index']))
        assignments[str(row['index'])] = selected['index']
    weights = Counter(assignments.values())
    if len(assignments) != 1000 or sum(weights.values()) != 1000:
        raise ValueError('Expected the complete1000-member pool')
    result = []
    for name in MODELS:
        rows = {r['index']: r for r in data['scores'] if r['model'] == name}
        if set(rows) != set(data['representatives']):
            raise ValueError('Missing representative score')
        hard_mean = math.fsum(rows[i]['gap_bpb'] for i in data['hard_indices']) / len(data['hard_indices'])
        proxy = math.fsum(weights[i] * rows[i]['gap_bpb'] for i in weights) / 1000
        result.append(dict(model=name, measured24_mean_gap=hard_mean,
            measured36_mean_gap=math.fsum(r['gap_bpb'] for r in rows.values()) / len(rows),
            estimated_pool_mean_gap=proxy, hard_minus_estimated_pool=hard_mean - proxy,
            hard_to_estimated_pool_ratio=hard_mean / proxy))
    return dict(models=result, representative_weights={str(i): weights[i] for i in sorted(weights)}, assignments=assignments,
        method='Assign each pool member to the nearest vocabulary-size representative within its algorithm/pretokenizer cell; average the36 measured scores using these1000 assignment counts.',
        limitation='A configuration-based proxy from36 nonrandom representatives, not an observed mean over all1000. It does not capture within-cell domain-mixture, merge-noise or seed variation. No extra GPU evaluation was run.',
        passages=229, documents=198, scored_tokens=7328, scored_bytes=23074)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/pool-reference-estimate')
    args = parser.parse_args();args.output.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.input.read_text()) if args.input else inputs()
    result = analyze(data)
    for name, value in [('inputs.json', data), ('analysis.json', result)]:
        (args.output / name).write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + '\n')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Patch
    font = ROOT / 'monitoring/web/Manrope.ttf'
    if font.exists():
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': 'Manrope' if font.exists() else 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    values = result['models']
    for k, row in enumerate(values):
        axes[0].bar(k - .17, row['estimated_pool_mean_gap'], width=.32, color=COLORS[k], alpha=.45)
        axes[0].bar(k + .17, row['measured24_mean_gap'], width=.32, color=COLORS[k])
        axes[0].text(k - .17, row['estimated_pool_mean_gap'] + .004, f'{row["estimated_pool_mean_gap"]:.3f}', ha='center', fontsize=9)
        axes[0].text(k + .17, row['measured24_mean_gap'] + .004, f'{row["measured24_mean_gap"]:.3f}', ha='center', fontsize=9)
        axes[1].bar(k, row['hard_minus_estimated_pool'], width=.5, color=COLORS[k])
    axes[0].set(xticks=range(3), xticklabels=LABELS, ylabel='Mean transfer gap · BPB', title='Selected 24 versus the estimated pool mean')
    axes[0].legend(handles=[Patch(color='#B5C3D2', label='Pool estimate from 36 settings'),
                            Patch(color='#526178', label='Measured mean of the 24 settings')], frameon=False)
    axes[0].margins(y=.16)
    axes[1].set(xticks=range(3), xticklabels=LABELS, ylabel='Additional transfer gap · BPB', title='How much harder are the 24?')
    figure.suptitle('229 fixed stress passages · existing scores only · pool average is estimated', fontsize=13)
    for suffix in ['png', 'pdf']:
        figure.savefig(args.output / f'pool_reference.{suffix}', dpi=180)
    plt.close(figure)
    lines = ['# Selected24 versus an estimated average pool tokenizer', '',
        '**No new GPU evaluation.** The full sweep was cancelled at the user’s request. This is a quick reference based on existing scores for36 representative pool tokenizers, reweighted to match the composition of the1000-member pool.', '',
        '| Model | Estimated pool mean gap | Measured24 mean gap | Additional gap for the24 |',
        '|---|---:|---:|---:|']
    for label, row in zip(LABELS, values):
        lines.append(f'| {label} | {row["estimated_pool_mean_gap"]:+.5f} | {row["measured24_mean_gap"]:+.5f} | {row["hard_minus_estimated_pool"]:+.5f} |')
    lines += ['', '![Pool reference](pool_reference.png)', '',
        'These are arithmetic means across tokenizers, not the medians reported previously. Every score uses the same229 context-dependent confirmation passages from198 documents. For each tokenizer, BPB is byte-weighted within source domains and the five sources receive equal weights.', '',
        'To estimate the full pool average, assign each of its1000 tokenizers to the closest vocabulary-size representative with the same algorithm and pretokenizer. The36 representatives cover all nine algorithm/pretokenizer combinations and four vocabulary scales. Weight their measured scores by the number of assigned pool members.', '',
        '**This is not the measured full-pool mean.** The representatives were selected by configuration, not sampled uniformly at random. The estimate ignores domain-mixture, BPE-noise and seed differences among tokenizers assigned to the same representative. Treat it as a settings-based reference, not a precise population estimate. The exact unweighted mean of the36 is also saved in analysis.json.', '',
        'The input score table, pool settings and source checksums are in inputs.json; all1000 assignments and36 weights are in analysis.json. The original paired model results remain in ../natural-model-comparison-20k/.', '',
        'Reproduce with `python scripts/analyze_pool_reference.py --input reports/pool-reference-estimate/inputs.json --output /tmp/pool-reference`.', '']
    text = '\n'.join(lines)
    for old, new in [('Selected24', 'Selected 24'), ('selected24', 'selected 24'), ('for36', 'for 36'), ('the1000', 'the 1,000'),
                     ('Measured24', 'Measured 24'), ('the24', 'the 24'), ('same229', 'same 229'), ('from198', 'from 198'),
                     ('its1000', 'its 1,000'), ('The36', 'The 36'), ('the36', 'the 36'), ('all1000', 'all 1,000'), ('and36', 'and 36')]:
        text = text.replace(old, new)
    (args.output / 'README.md').write_text(text)
    names = ['inputs.json', 'analysis.json', 'README.md', 'pool_reference.png', 'pool_reference.pdf']
    save_json(args.output / 'manifest.json', {'files': {name: dict(bytes=(args.output / name).stat().st_size,
        sha256=file_sha(args.output / name)) for name in names}})
    print(json.dumps(values, indent=2))


if __name__ == '__main__':
    main()
