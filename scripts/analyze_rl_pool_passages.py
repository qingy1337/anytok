"""Explain the exact scored text behind the RL pool evaluation gap."""
import argparse
import json
from pathlib import Path

from pool_generalization import file_sha

ROOT = Path(__file__).resolve().parents[1]


def analyze(result):
    screen = result['scores']['screen']
    confirm = result['scores']['confirm']
    shift = confirm['mean_gaps']['635'] - screen['mean_gaps']['635']
    domains = []
    for domain, old in screen['domain_scores']['635'].items():
        new = confirm['domain_scores']['635'][domain]
        domains.append({'domain': domain, 'screen_gap_bpb': old['gap_bpb'],
                        'confirmation_gap_bpb': new['gap_bpb'],
                        'contribution_to_gap_increase': (new['gap_bpb'] - old['gap_bpb']) / 9})
    if abs(sum(r['contribution_to_gap_increase'] for r in domains) - shift) > 1e-12:
        raise ValueError('Domain contributions do not reconstruct the gap increase')
    for split in ['screen', 'confirm']:
        rows = result['scores'][split]['rows']
        for index in ['635', '3']:
            total = sum(r['scores'][index]['contribution_to_mean_gap_bpb'] for r in rows)
            if abs(total - result['scores'][split]['mean_gaps'][index]) > 1e-12:
                raise ValueError('Passage contributions do not reconstruct the reported gap')
    domains.sort(key=lambda r: -r['contribution_to_gap_increase'])
    examples = sorted(result['examples'], key=lambda r:
        (r['split'] != 'confirm', -r['scores']['635']['contribution_to_mean_gap_bpb']))
    worst = examples[0]
    medical = next(r for r in domains if r['domain'] == 'medical-en')
    return {'tokenizer_index': 635, 'screen_mean_gap_bpb': screen['mean_gaps']['635'],
            'confirmation_mean_gap_bpb': confirm['mean_gaps']['635'], 'gap_increase': shift,
            'medical_fraction_of_increase': medical['contribution_to_gap_increase'] / shift,
            'domain_decomposition': domains, 'largest_confirmation_contributor': worst,
            'limits': 'Selected high-loss passages explain these saved scores. They do not establish which token or model component causes the loss. Selection and textual interpretation are post hoc.'}


def main(source, out):
    result = json.loads(source.read_text())
    report = analyze(result)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'passages.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    (out / 'analysis.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    worst = report['largest_confirmation_contributor']
    score = worst['scores']['635']
    lines = ['# Exact text behind the RL transfer gap', '',
        'The largest contributors are passages of repeated image-link markup in the medical corpus. '
        'The selected examples also include Chinese blog-interface text, Japanese prose and a decimal-number copying passage.', '',
        f"For tokenizer 0635, the average gap rises from **+{report['screen_mean_gap_bpb']:.5f}** on the first sample "
        f"to **+{report['confirmation_mean_gap_bpb']:.5f} BPB** on the separate confirmation sample. "
        f"The medical-domain change accounts for **{100 * report['medical_fraction_of_increase']:.1f}%** of that increase.", '',
        f"The largest confirmation contributor is **{worst['id']}**: canonical **{score['canonical_bpb']:.3f} BPB**, "
        f"transferred **{score['transferred_bpb']:.3f} BPB**, gap **+{score['gap_bpb']:.3f} BPB**. "
        f"It has {worst['continuation_bytes']} scored bytes and 32 scored canonical tokens.", '',
        'The +0.101 number is an average across the complete evaluation sample. '
        'An individual passage can have a much larger gap. Each passage contributes according '
        'to its scored bytes within its domain, and each of the nine domains has equal weight.', '',
        '## What changed between the two text samples', '',
        '| Domain | First-sample gap | Confirmation gap | Contribution to average-gap increase |',
        '|---|---:|---:|---:|']
    for row in report['domain_decomposition']:
        lines.append(f"| {row['domain']} | {row['screen_gap_bpb']:+.5f} | "
            f"{row['confirmation_gap_bpb']:+.5f} | {row['contribution_to_gap_increase']:+.5f} |")
    lines += ['', '## Exact passages', '',
        'The prefix is the text fed through the transferred input embeddings. '
        'The continuation is the exact next 32 canonical tokens that were scored. '
        'Cuts can fall inside a word or filename. Line breaks and spelling below come from '
        'the stored evaluation data. The JSON retains exact whitespace and native prefix token IDs.', '',
        'Examples are ordered by contribution to tokenizer 0635’s total gap, with confirmation '
        'examples first. We selected the ten largest contributors from each sample and the '
        'largest Chinese, Japanese and math contributors. This is a post-hoc inspection of '
        'saved losses; it does not establish a causal explanation.', '']
    for example in sorted(result['examples'], key=lambda r:
            (r['split'] != 'confirm', -r['scores']['635']['contribution_to_mean_gap_bpb'])):
        value = example['scores']['635']
        lines += [f"### {example['split']} · {example['domain']} · {example['id']}", '',
            f"Canonical {value['canonical_bpb']:.4f} BPB; transferred {value['transferred_bpb']:.4f}; "
            f"gap {value['gap_bpb']:+.4f}. Contribution to overall mean gap: "
            f"{value['contribution_to_mean_gap_bpb']:+.6f} BPB.", '', 'Prefix:', '',
            '````text', example['prefix'], '````', '', 'Scored continuation:', '',
            '````text', example['continuation'], '````', '']
    lines += ['## Data and reproduction', '',
        '* [passages.json](passages.json): the selected exact text, native prefix tokenization, '
        'and all 576 passage-level losses/contributions for both tokenizers.',
        '* [analysis.json](analysis.json): domain decomposition and the leading passage.',
        '* [manifest.json](manifest.json): file sizes and SHA-256 hashes.', '',
        'The source panel stays in `blackpantherx86 / anytok-rl-all-pool-20k / panel.json`. '
        'The cloud extraction validated the complete panel hashes, ordered document IDs, '
        'canonical/native tokenizer file hashes, prefix round trips, and 32-token continuation byte counts. '
        'Only selected short passages were retrieved locally to answer the request.', '',
        'Recreate this report with '
        '`python scripts/analyze_rl_pool_passages.py --input reports/rl-pool-passages-20k/passages.json`.', '']
    (out / 'README.md').write_text('\n'.join(lines))
    manifest = {'files': {p.name: {'bytes': p.stat().st_size, 'sha256': file_sha(p)}
        for p in sorted(out.iterdir()) if p.is_file() and p.name != 'manifest.json'}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['screen_mean_gap_bpb', 'confirmation_mean_gap_bpb', 'gap_increase', 'medical_fraction_of_increase']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'modal_results/rl_pool_passages/result.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/rl-pool-passages-20k')
    args = parser.parse_args()
    main(args.input, args.output)
