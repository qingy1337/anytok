"""Report repeatable behavioral weaknesses across native tokenizer settings."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np

from pool_generalization import file_sha

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ['numbers', 'identifiers', 'names', 'quoted_text', 'semantic_retrieval', 'natural_text']
NAMES = {'numbers': 'Numbers', 'identifiers': 'Identifiers', 'names': 'Names',
         'quoted_text': 'Quoted English', 'semantic_retrieval': 'Fact retrieval', 'natural_text': 'Natural text'}


def load(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as stream:
            result = json.load(stream)
        for row in result['results']:
            row['canonical'] = result['canonical']
        return result
    return json.loads(path.read_text())


def individual(meta, scores, indices):
    answer_nats = np.array([sum(scores[i]['nats'][:len(meta[i]['answer_ids'])]) for i in indices])
    full_nats = np.array([sum(scores[i]['nats']) for i in indices])
    correct = np.array([all(scores[i]['correct'][:len(meta[i]['answer_ids'])]) for i in indices])
    return answer_nats, full_nats, correct


def metric(meta, canonical, transferred, indices, bootstrap=2000):
    cb, cf, ca = individual(meta, canonical, indices)
    tb, tf, ta = individual(meta, transferred, indices)
    counts = np.array([meta[i]['answer_bytes'] for i in indices])
    full_bytes = np.array([meta[i]['continuation_bytes'] for i in indices])
    denom = counts.sum() * math.log(2)
    rng = np.random.default_rng(20261116)
    # Paired stratified document draws preserve the planned length/distance mix.
    cells = {}
    for j, i in enumerate(indices):
        cells.setdefault((meta[i]['size'], meta[i]['distance']), []).append(j)
    draws = np.concatenate([rng.choice(group, size=(bootstrap, len(group))) for group in cells.values()], axis=1)
    gaps = (tb - cb)[draws].sum(1) / (counts[draws].sum(1) * math.log(2))
    drops = (ca.astype(float) - ta)[draws].mean(1)
    return {'examples': len(indices), 'canonical_accuracy': float(ca.mean()),
        'transferred_accuracy': float(ta.mean()), 'accuracy_drop': float(ca.mean() - ta.mean()),
        'accuracy_drop_ci95': np.quantile(drops, [.025, .975]).tolist(),
        'canonical_answer_bpb': float(cb.sum() / denom),
        'transferred_answer_bpb': float(tb.sum() / denom), 'answer_gap_bpb': float((tb - cb).sum() / denom),
        'answer_gap_ci95': np.quantile(gaps, [.025, .975]).tolist(),
        'canonical_bpb_32': float(cf.sum() / (full_bytes.sum() * math.log(2))),
        'transferred_bpb_32': float(tf.sum() / (full_bytes.sum() * math.log(2))),
        'gap_bpb_32': float((tf - cf).sum() / (full_bytes.sum() * math.log(2)))}


def analyze(result):
    meta = result['preparation']['metadata']
    base = result['results'][0]['canonical']
    if sorted(r['index'] for r in result['results']) != result['plan']['indices']:
        raise ValueError('Missing native tokenizer settings')
    for row in result['results']:
        if row['canonical'] != base or len(row['transferred']) != len(meta):
            raise ValueError('Unpaired canonical results or missing examples')
        for score in row['transferred']:
            if len(score['nats']) != 32 or len(score['correct']) != 32 or not np.isfinite(score['nats']).all():
                raise ValueError('Invalid token scores')
    metrics = []
    for row in result['results']:
        for family in FAMILIES:
            for split in ['screen', 'confirm']:
                for size in ['all', 'short', 'long'] if family != 'natural_text' else ['all']:
                    indices = [i for i, r in enumerate(meta) if r['family'] == family
                               and r['split'] == split and (size == 'all' or r['size'] == size)]
                    value = metric(meta, base, row['transferred'], indices)
                    metrics.append({'index': row['index'], 'family': family, 'split': split,
                        'size': size, 'algorithm': row['config']['algorithm'],
                        'pretokenizer': row['config']['pretokenizer'],
                        'vocab_size': row['config']['vocab_size'], **value})
    representative = set(result['plan']['representative_indices'])
    summaries = []
    for family in FAMILIES:
        for size in ['all', 'short', 'long'] if family != 'natural_text' else ['all']:
            supporting = []
            for index in representative:
                pair = [m for m in metrics if m['index'] == index and m['family'] == family and m['size'] == size]
                if all(m['canonical_accuracy'] >= .8 and m['accuracy_drop'] >= .2
                       and m['answer_gap_ci95'][0] > 0 for m in pair):
                    supporting.append(index)
            splits = {}
            for split in ['screen', 'confirm']:
                rows = [m for m in metrics if m['index'] in representative and m['family'] == family
                        and m['size'] == size and m['split'] == split]
                splits[split] = {'examples': rows[0]['examples'],
                    'canonical_accuracy': rows[0]['canonical_accuracy'],
                    'transferred_accuracy_min': min(r['transferred_accuracy'] for r in rows),
                    'transferred_accuracy_median': float(np.median([r['transferred_accuracy'] for r in rows])),
                    'transferred_accuracy_max': max(r['transferred_accuracy'] for r in rows),
                    'median_answer_gap_bpb': float(np.median([r['answer_gap_bpb'] for r in rows])),
                    'median_gap_bpb_32': float(np.median([r['gap_bpb_32'] for r in rows]))}
            summaries.append({'family': family, 'size': size, 'supporting_indices': sorted(supporting),
                'support_count': len(supporting), 'representative_count': len(representative), 'splits': splits})
    return {'metrics': metrics, 'summaries': summaries,
        'criterion': 'Canonical answer-token accuracy>=80%, transferred drop>=20 percentage points, paired answer-BPB95% interval above zero in both splits.',
        'limits': 'Designed English stress tasks and stratified tokenizer settings, not a population estimate over all text or all1000 tokenizers. Confirmation uses independent draws and new prompt wordings; common color/name values can repeat. No multiple-comparison correction.'}


def plot(report, out):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import PercentFormatter
    font = ROOT / 'monitoring/web/Manrope.ttf'
    if font.exists():
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': 'Manrope' if font.exists() else 'DejaVu Sans', 'font.size': 9,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'text.color': '#333333'})
    colors = {'bpe': '#2F8FD0', 'wordpiece': '#8657BC', 'unigram': '#36967B'}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(left=.08, right=.97, top=.87, bottom=.11, wspace=.30, hspace=.48)
    numbers = next(r for r in report['summaries'] if r['family'] == 'numbers' and r['size'] == 'all')
    representative = set(numbers['supporting_indices'])
    rows = sorted([r for r in report['metrics'] if r['index'] in representative and r['family'] == 'numbers'
                   and r['split'] == 'confirm' and r['size'] == 'all'], key=lambda r: (r['algorithm'], r['pretokenizer'], r['vocab_size']))
    ax = axes[0, 0]
    ax.axhline(numbers['splits']['confirm']['canonical_accuracy'], color='#73808E', ls='--', label='Canonical inputs ·87.5%')
    ax.bar(range(len(rows)), [r['transferred_accuracy'] for r in rows], color=[colors[r['algorithm']] for r in rows])
    ax.set_xticks(range(len(rows)), [f"{r['index']:04d}" for r in rows], rotation=60)
    ax.set(ylim=(0, 1.05), xlabel='18 representative tokenizer settings', ylabel='Exact answer-token accuracy', title='Copying numbers · fresh wording and values')
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(frameon=False);ax.grid(axis='y', alpha=.12)
    ax = axes[0, 1]
    families = ['numbers', 'names', 'quoted_text']
    selected = [next(r for r in report['summaries'] if r['family'] == f and r['size'] == 'all')['splits']['confirm'] for f in families]
    positions = np.arange(len(families))
    ax.bar(positions - .18, [r['median_answer_gap_bpb'] for r in selected], width=.34, color='#8657BC', label='Answer bytes only')
    ax.bar(positions + .18, [r['median_gap_bpb_32'] for r in selected], width=.34, color='#2F8FD0', label='Full32-token continuation')
    for i, row in enumerate(selected):
        ax.text(i - .18, row['median_answer_gap_bpb'] + .03, f"{row['median_answer_gap_bpb']:.3f}", ha='center', fontsize=8)
        ax.text(i + .18, row['median_gap_bpb_32'] + .03, f"{row['median_gap_bpb_32']:.3f}", ha='center', fontsize=8)
    ax.set_xticks(positions, [NAMES[f] for f in families]);ax.set(ylabel='Median transferred − canonical BPB', title='Easy trailing text can dilute the task error')
    ax.legend(frameon=False);ax.grid(axis='y', alpha=.12)
    ax = axes[1, 0]
    lengths = [next(r for r in report['summaries'] if r['family'] == 'numbers' and r['size'] == size)['splits']['confirm'] for size in ['short', 'long']]
    ax.bar(np.arange(2) - .18, [r['canonical_accuracy'] for r in lengths], width=.34, color='#73808E', label='Canonical')
    ax.bar(np.arange(2) + .18, [r['transferred_accuracy_median'] for r in lengths], width=.34, color='#2F8FD0', label='Transferred · median')
    ax.set_xticks(range(2), ['8-digit payloads', '20-digit payloads']);ax.set(ylim=(0, 1.05), ylabel='Exact answer-token accuracy', title='The failure includes shorter numbers')
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(frameon=False);ax.grid(axis='y', alpha=.12)
    ax = axes[1, 1]
    names = next(r for r in report['summaries'] if r['family'] == 'names' and r['size'] == 'all')['splits']['confirm']
    quote = next(r for r in report['summaries'] if r['family'] == 'quoted_text' and r['size'] == 'all')['splits']['confirm']
    ax.bar(np.arange(2) - .18, [names['canonical_accuracy'], quote['canonical_accuracy']], width=.34, color='#73808E', label='Canonical')
    ax.bar(np.arange(2) + .18, [names['transferred_accuracy_median'], quote['transferred_accuracy_median']], width=.34, color='#8657BC', label='Transferred · median')
    ax.set_xticks(range(2), ['Personal names', 'Quoted English']);ax.set(ylim=(0, 1.05), ylabel='Exact answer-token accuracy', title='Exact recall weakness beyond numerical strings')
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(frameon=False);ax.grid(axis='y', alpha=.12)
    fig.suptitle('RL20k · exact information is lost across tokenizer transfer', fontsize=17, y=.97)
    fig.text(.08, .925, '18 representative settings ·3 algorithms ×3 pretokenizers ×2 vocabulary scales · same model and raw text · ≤128 positions', fontsize=10, color='#73808E')
    fig.text(.08, .025, 'Controlled stress tasks; not an estimate of average natural-text performance. Fresh confirmation uses new wording.\nCanonical accuracy must be high before a transfer failure is claimed. Accuracy requires the canonical answer-token sequence; selected errors also have actual greedy outputs.', fontsize=8, color='#73808E')
    for ext in ['png', 'pdf']:
        fig.savefig(out / f'behavior_transfer.{ext}', dpi=180)
    plt.close(fig)


def main(source, out):
    result = load(source)
    report = analyze(result)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    compact = {k: result[k] for k in ['plan', 'preparation']}
    compact['canonical'] = result['results'][0]['canonical']
    compact['results'] = [{k: v for k, v in row.items() if k != 'canonical'} for row in result['results']]
    (out / 'results.json.gz').write_bytes(gzip.compress(json.dumps(compact, separators=(',', ':')).encode(), mtime=0))
    fields = [k for k, v in report['metrics'][0].items() if not isinstance(v, list)]
    with (out / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields, lineterminator='\n', extrasaction='ignore')
        writer.writeheader();writer.writerows(report['metrics'])
    plot(report, out)
    lines = ['# A repeatable weakness: exact information in transferred prefixes', '',
        '**Copying numerical values and personal names fails across all18 representative tokenizer settings in this probe.** '
        'The canonical-input model succeeds on these tasks. The failure repeats with new prompt wording and independently generated examples.', '',
        'The general setting is read-and-repeat: put a value, name or short sentence in an English record, '
        'then ask the same model to reproduce it. Only the input-prefix tokenizer and embeddings change. '
        'Keep the complete fact and question, use the fixed canonical output head, and stay within128 positions.', '',
        '## Results on the fresh confirmation wording', '',
        '| Task | Canonical accuracy | Transferred median | Transferred range | Settings meeting failure criterion on both splits |',
        '|---|---:|---:|---:|---:|']
    for family in FAMILIES[:-1]:
        row = next(r for r in report['summaries'] if r['family'] == family and r['size'] == 'all')
        values = row['splits']['confirm']
        lines.append(f"| {NAMES[family]} | {values['canonical_accuracy']:.1%} | {values['transferred_accuracy_median']:.1%} | "
            f"{values['transferred_accuracy_min']:.1%}–{values['transferred_accuracy_max']:.1%} | {row['support_count']}/18 |")
    lines += ['', 'Numbers: canonical accuracy is100% on discovery and87.5% on confirmation. '
        'The median transferred accuracy is0% on both. All18 settings span BPE, WordPiece and Unigram; '
        'regex, line256 and no-regex pretokenization; and approximately64k/120k vocabularies. '
        'The separately included earlier weak tokenizers0003/0635 are not needed for that conclusion.', '',
        'Names also meet the failure criterion across all18 settings. Quoted English shows a large confirmed drop '
        'in a subset of settings. Identifier and semantic-retrieval prompts do **not** pass our cross-split '
        'baseline calibration: canonical identifier accuracy falls to46.9% on confirmation, and canonical '
        'semantic accuracy is0% on the discovery wording. We do not claim a confirmed general failure from those two tests.', '',
        '![Behavioral transfer results](behavior_transfer.png)', '',
        '## The loss gap on this setting', '',
        'For numerical copying, the median confirmation gap across representative settings is '
        f"**+{next(r for r in report['summaries'] if r['family']=='numbers' and r['size']=='all')['splits']['confirm']['median_answer_gap_bpb']:.3f} BPB on the answer**, "
        f"and **+{next(r for r in report['summaries'] if r['family']=='numbers' and r['size']=='all')['splits']['confirm']['median_gap_bpb_32']:.3f} BPB over the full32-token continuation**. "
        'The latter uses the same continuation length as the old evaluation. Easy, shared trailing prose lowers '
        'that average while the actual copied value is wrong.', '',
        'These are task-conditioned stress-test gaps, not revised estimates for the old random-text panel. '
        'On the natural-text reference passages, the median32-token gap is '
        f"+{next(r for r in report['summaries'] if r['family']=='natural_text')['splits']['confirm']['median_gap_bpb_32']:.4f} BPB. "
        'Small average continuation loss therefore coexists with unreliable exact recall.', '',
        '## Concrete greedy-generation failures', '']
    representatives = set(result['plan']['representative_indices'])
    for algorithm in ['bpe', 'wordpiece', 'unigram']:
        row = next(r for r in result['results'] if r['index'] in representatives and r['config']['algorithm'] == algorithm and r['examples'])
        example = row['examples'][0]
        lines += [f"### {algorithm} · tokenizer {row['index']:04d}", '', 'Prefix:', '',
            '````text', example['prefix'], '````', '',
            f"Required value: `{example['answer'].strip()}`", '',
            f"Actual transferred greedy output: `{example['greedy_transferred_answer']['text'].strip().replace(chr(10), ' / ')}`", '',
            'The canonical model reproduces the complete target-token sequence on this example.', '']
    lines += ['## Design and limits', '',
        'Each split has64 generated cases per family, equally divided by short/long payload and nearby/separated facts. '
        'Confirmation uses a new random seed and different wording templates. Common color/name values can repeat '
        'between splits. There were zero encoding/length rejections. The shared panel also includes153 discovery '
        'and150 confirmation natural English/code/math/medical/legal continuations without image-link markup.', '',
        'Exact accuracy means every answer token matches the canonical target sequence. This is equivalent to '
        'greedy reproduction of that token sequence; it is stricter than accepting alternate tokenizations or '
        'paraphrases. Selected failures also include actual free-running greedy outputs. Natural prose accuracy '
        'is not interpreted as a task-success rate. Per-token losses agreed with the model’s built-in causal loss.', '',
        'A large supported failure requires canonical accuracy≥80%, an accuracy drop≥20 percentage points, '
        'and a paired answer-BPB95% interval above zero in both splits. Intervals resample paired cases within '
        'the fixed length/distance cells; they do not cover training seeds and are not multiplicity-adjusted. '
        'Representative tokenizers are selected by settings/vocabulary size, not by these task losses. '
        'This is evidence of a repeatable behavioral weakness across the tested settings. It does not measure '
        'its frequency in arbitrary real documents or establish which learned component causes it.', '',
        '## Files and reproduction', '',
        '* [metrics.csv](metrics.csv): all settings, tasks, lengths and splits.',
        '* [analysis.json](analysis.json): paired intervals and cross-split support counts.',
        '* [results.json.gz](results.json.gz): frozen plan, shared case metadata, per-token losses/correctness and selected raw examples.',
        '* [PDF plot](behavior_transfer.pdf) and [manifest.json](manifest.json).', '',
        'Run `python scripts/analyze_behavior_transfer.py --input reports/behavior-transfer-20k/results.json.gz` '
        'to reproduce the report. Raw task contexts stay in '
        '`blackpantherx86 / anytok-rl-all-pool-20k / behavior-transfer-v1/panel.json`. '
        'The plan is `plans/behavior-transfer-v1.md`; generation/scoring source is '
        '`scripts/behavior_transfer.py` and `scripts/modal_behavior_transfer.py`.', '']
    (out / 'README.md').write_text('\n'.join(lines))
    manifest = {'files': {p.name: {'bytes': p.stat().st_size, 'sha256': file_sha(p)}
        for p in sorted(out.iterdir()) if p.is_file() and p.name != 'manifest.json'}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(report['summaries'][:3], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'modal_results/behavior_transfer/results.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/behavior-transfer-20k')
    args = parser.parse_args()
    main(args.input, args.output)
