"""Compare corrected broad-text scores with the original full-pool screen."""
import argparse
import gzip
import json
from pathlib import Path
import random

import numpy as np

from analyze_natural_transfer import bootstrap_balanced
from pool_generalization import digest, file_sha, save_json
from uniform_trajectory import PROFILE_DOMAINS

ROOT = Path(__file__).resolve().parents[1]


def pack(state):
    prep = json.loads((state / 'preparation.json').read_text());plan = prep['plan']
    raw = json.loads(gzip.decompress((state / 'scores.json.gz').read_bytes()))
    if raw['panel_sha256'] != prep['panel_sha256'] or raw['plan_sha256'] != digest(plan):
        raise ValueError('Scores do not match the frozen corrected panel')
    spec = json.loads((ROOT / 'reports/rl-pool-sweep-20k/spec.json').read_text())
    if digest(spec) != plan['source_spec_sha256']:
        raise ValueError('Original specification changed')
    results = sorted(raw['results'], key=lambda r: r['index'])
    if [r['index'] for r in results] != plan['indices']:
        raise ValueError('Missing tokenizer scores')
    base = np.asarray(raw['canonical'], dtype=float)
    native = np.asarray([r['transferred'] for r in results], dtype=float)
    if base.shape != (576, 32) or native.shape != (50, 576, 32):
        raise ValueError('Missing scored tokens')
    old = {}
    for split, filename in [('screen', 'screen.json'), ('confirm', 'confirmation.json')]:
        full = json.loads((ROOT / 'reports/rl-pool-sweep-20k' / filename).read_text())
        if full['evaluation_sha256'] != digest(spec):
            raise ValueError('Original scores changed')
        subset = {str(r['index']): r['transferred_nats'] for r in full['rows'] if r['index'] in plan['indices']}
        old[split] = dict(metadata=spec['panels'][split], canonical_nats=full['rows'][0]['canonical_nats'],
                          transferred_nats=subset)
    audit = json.loads((ROOT / 'modal_results/original_pool_boundary_audit/audit.json').read_text())
    if audit['panel_sha256'] != plan['parent_panel_sha256'] or audit['source_spec_sha256'] != digest(spec):
        raise ValueError('Boundary audit belongs to another original panel')
    return dict(plan=plan, preparation=prep, audit=audit, canonical_nats=base.sum(1).tolist(),
        transferred_nats=native.sum(2).tolist(), old=old,
        tokenizers=[spec['tokenizers'][i] for i in plan['indices']])


def analyze(data):
    plan = data['plan'];rows = data['preparation']['metadata'];indices = plan['indices']
    sample = random.Random(plan['seed']).sample(list(range(1000)), 50)
    if plan['random_indices'] != sample or digest(sample) != plan['random_indices_sha256']:
        raise ValueError('Frozen random sample changed')
    if indices != sorted(set(sample + [635])) or 635 not in sample:
        raise ValueError('Invalid comparison settings')
    base = np.asarray(data['canonical_nats'], dtype=float)
    native = np.asarray(data['transferred_nats'], dtype=float)
    if base.shape != (576,) or native.shape != (50, 576):
        raise ValueError('Missing passages')
    if not np.isfinite(base).all() or not np.isfinite(native).all() or (base < 0).any() or (native < 0).any():
        raise ValueError('Invalid losses')
    if len({r['id'] for r in rows}) != 576 or not all(r['maximum_positions'] <= 128 and r['canonical_joint_encoding_matches'] for r in rows):
        raise ValueError('Invalid corrected contexts')
    configs = {r['index']: r['config'] for r in data['tokenizers']};report = dict(splits={}, tokenizers=[])
    for split in ['screen', 'confirm']:
        positions = [j for j, r in enumerate(rows) if r['split'] == split];sub = [rows[j] for j in positions]
        old = data['old'][split];meta = old['metadata']
        if ([r['original_id'] for r in sub] != meta['document_ids']
                or [r['domain'] for r in sub] != meta['domains'] or len(sub) != 288):
            raise ValueError('Original document IDs, domains or order changed')
        point, draws = bootstrap_balanced(sub, np.concatenate([base[None, positions], native[:, positions]]),
            plan['bootstrap_replicates'], domains=PROFILE_DOMAINS)
        gaps = point[1:] - point[0];gap_draws = draws[:, 1:] - draws[:, :1]
        old_rows = [dict(domain=dom, cluster_id=r['cluster_id'], continuation_bytes=b)
                    for r, dom, b in zip(sub, meta['domains'], meta['byte_counts'])]
        old_indices = sorted(map(int, old['transferred_nats']))
        old_values = np.asarray([old['transferred_nats'][str(i)] for i in old_indices]) - np.asarray(old['canonical_nats'])
        old_points, _ = bootstrap_balanced(old_rows, old_values, 1, domains=PROFILE_DOMAINS)
        old_gaps = dict(zip(old_indices, old_points.tolist()))
        mean = float(gaps.mean());worst = indices.index(635);difference = gap_draws[:, worst] - gap_draws.mean(1)
        info = dict(passages=len(sub), scored_tokens=32 * len(sub), scored_bytes=sum(r['continuation_bytes'] for r in sub),
            changed_cuts=sum(r['cut_shift_chars'] != 0 for r in sub),
            max_absolute_cut_shift_chars=max(abs(r['cut_shift_chars']) for r in sub),
            already_valid_contexts=sum(r['original_joint_encoding_matches'] for r in sub),
            canonical_bpb=float(point[0]), random50_mean_gap=mean,
            random50_ci95=np.quantile(gap_draws.mean(1), [.025, .975]).tolist(),
            old_worst_gap=float(gaps[worst]), old_worst_ci95=np.quantile(gap_draws[:, worst], [.025, .975]).tolist(),
            old_worst_minus_random=float(gaps[worst] - mean),
            difference_ci95=np.quantile(difference, [.025, .975]).tolist(),
            old_protocol_worst_gap=old_gaps[635],
            old_protocol_random50_mean_gap=float(np.mean([old_gaps[i] for i in sample])) if set(sample) <= set(old_gaps) else None,
            worst_sampled_index=indices[int(np.argmax(gaps))], worst_sampled_gap=float(gaps.max()), groups={})
        for pretokenizer in ['regex', 'line256', 'no-regex']:
            mask = [k for k, i in enumerate(indices) if configs[i]['pretokenizer'] == pretokenizer]
            info['groups'][pretokenizer] = dict(count=len(mask), mean_gap=float(gaps[mask].mean()),
                ci95=np.quantile(gap_draws[:, mask].mean(1), [.025, .975]).tolist())
        report['splits'][split] = info
        report['tokenizers'] += [dict(split=split, index=i, config=configs[i], gap_bpb=float(gaps[k]),
            old_protocol_gap_bpb=old_gaps.get(i)) for k, i in enumerate(indices)]
    report['scope'] = 'Fixed uniform random50, already including original worst0635; final RL20k only; original broad nine-domain screen and confirmation documents. No context-dependent text filtering. This is not a reranking of all1000.'
    report['uncertainty'] = '95% paired document-cluster bootstrap within each of nine domains, conditional on the fixed50 and one checkpoint. No tokenizer-sampling or training-seed uncertainty.'
    return report


def plot(report, output):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = ROOT / 'monitoring/web/Manrope.ttf'
    if font.exists():
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': 'Manrope' if font.exists() else 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for ax, (split, r) in zip(axes, report['splits'].items()):
        values = [r['random50_mean_gap'], r['old_worst_gap']]
        for j, (v, key, color) in enumerate(zip(values, ['random50_ci95', 'old_worst_ci95'], ['#36967B', '#8657BC'])):
            ci = r[key];ax.bar(j, v, width=.5, color=color)
            ax.errorbar(j, v, yerr=[[max(0, v - ci[0])], [max(0, ci[1] - v)]], fmt='none', color='#3F4957', capsize=4)
        ax.scatter(1, r['old_protocol_worst_gap'], marker='_', s=500, color='#67717D', linewidths=3, label='Old score (where available)')
        if r['old_protocol_random50_mean_gap'] is not None:
            ax.scatter(0, r['old_protocol_random50_mean_gap'], marker='_', s=500, color='#67717D', linewidths=3)
        ax.set_xticks([0, 1], ['Uniform random 50', 'Original worst #0635'])
        ax.set(title=f'{split.capitalize()} · original 288 broad-text passages', ylabel='Transfer gap versus trained canonical · BPB', ylim=(0, .19))
        ax.legend(frameon=False, loc='upper left', fontsize=9)
        ax.axhline(0, color='#999999', lw=.8)
    fig.suptitle('RL 20k · corrected token boundaries · no context-dependent passage selection', fontsize=13)
    for suffix in ['png', 'pdf']:
        fig.savefig(output / ('corrected_pool_check.' + suffix), dpi=180, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'modal_results/corrected_pool_check')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/corrected-pool-check')
    args = parser.parse_args()
    data = pack(args.input) if args.input.is_dir() else json.loads(gzip.decompress(args.input.read_bytes()))
    report = analyze(data);out = args.output;out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json.gz').write_bytes(gzip.compress(json.dumps(data, sort_keys=True, separators=(',', ':')).encode(), mtime=0))
    (out / 'analysis.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
    lines = ['# Corrected broad-text check: random 50 including the original worst', '',
        '**The old +0.066 BPB result was understated by the cutting protocol.** On the same original screen documents, correcting token boundaries raises tokenizer #0635 to about +0.104 BPB. On the separate confirmation documents its corrected gap is about +0.138 BPB.', '',
        'This is the final RL20k model, not a new training run. These are the original broad-text samples, not the 229 context-dependent passages used in the recent diagnostic. The exact same frozen random 50 from that experiment are reused. The old worst, #0635, was already in this draw, so there are 50 distinct tokenizers, not 51.', '',
        '| Text sample | Old random-50 mean | Corrected random-50 mean | Old #0635 | Corrected #0635 |',
        '|---|---:|---:|---:|---:|']
    for split, r in report['splits'].items():
        old = 'Not measured' if r['old_protocol_random50_mean_gap'] is None else f'{r["old_protocol_random50_mean_gap"]:+.5f}'
        lines.append(f'| {split.capitalize()} | {old} | {r["random50_mean_gap"]:+.5f} | {r["old_protocol_worst_gap"]:+.5f} | {r["old_worst_gap"]:+.5f} |')
    lines += ['', 'Gap = transferred BPB minus the same trained LM’s canonical BPB on identical targets within each protocol. Lower is better. BPB is byte-weighted within each of nine domains, then equally averaged across domains. The random sample mean includes #0635.', '',
        '![Corrected broad-text comparison](corrected_pool_check.png)', '',
        'The original worst is also worst among these 50 on both corrected samples. Its gap exceeds the random mean by about +0.036 BPB on screen and +0.048 on confirmation. This does not establish the corrected worst across all 1,000. The random mean remains positive; this is not a case of near-zero loss for the typical sampled tokenizer.', '',
        '## What is kept fixed and what changes', '',
        'Both screen and confirmation contain 288 source documents, 32 from each of English web, code, Chinese, Japanese, Arabic, Hindi, math, medical and legal text. No passage was selected because of context benefit, transferred loss or model weakness. All original source IDs and split membership are preserved.', '',
        'The audit found 128 mismatched seams on screen and 118 on confirmation. All already-valid contexts are preserved exactly, including prefix and all 32 targets. Invalid cuts move to the nearest reversible canonical token end within 64 Unicode characters, using encoding validity only. The old prefix start is retained, with trimming only when needed for the position limit. Because a cut moves, its continuation bytes may change: this is a corrected protocol on matched documents, not an identical-target comparison between protocols.', '',
        'Each model input is BOS + prefix + 32 canonical continuation tokens, at most 128 positions. The prefix is input only; teacher-forced loss is scored on the continuation. Canonical prefixes are 16–64 tokens; native prefixes are at most 95. Full native vocabularies generate transferred input embeddings. The canonical output head is fixed. Checkpoint, native-tokenizer and surface-feature hashes are verified, and token loss is checked against Hugging Face’s built-in causal loss.', '',
        '| Sample | Changed cuts | Largest shift (characters) | Scored tokens | Corrected continuation bytes |',
        '|---|---:|---:|---:|---:|']
    for split, r in report['splits'].items():
        lines.append(f'| {split.capitalize()} | {r["changed_cuts"]} | {r["max_absolute_cut_shift_chars"]} | {r["scored_tokens"]:,} | {r["scored_bytes"]:,} |')
    lines += ['', '## Pretokenizer groups within the random sample', '',
        '| Group | Count | Corrected screen gap | Corrected confirmation gap |', '|---|---:|---:|---:|']
    for key, r in report['splits']['screen']['groups'].items():
        other = report['splits']['confirm']['groups'][key]
        lines.append(f'| {key} | {r["count"]} | {r["mean_gap"]:+.5f} | {other["mean_gap"]:+.5f} |')
    lines += ['', 'These groups contain different sampled tokenizers; this is an association, not a matched pretokenization ablation. The screen and confirmation are different text samples and are not pooled.', '',
        'Plot intervals and intervals in analysis.json use paired document-cluster resampling within domains, conditional on these fixed 50 tokenizers and one trained checkpoint. They exclude tokenizer-sampling and training-seed uncertainty. No claim of a unique full-pool worst follows from this rerun.', '',
        '## Data and reproduction', '',
        '`results.json.gz` contains per-passage summed losses for all 50, old reference scores, source metadata and the frozen plan. Raw text, per-token scores, full tokenizer files and checkpoints remain in persistent Modal Volume `blackpantherx86 / anytok-rl-all-pool-20k`, under `corrected-pool-check-v1/` for the new evaluation and `original-pool-boundary-audit-v1/` for the audit.', '',
        'Run `python scripts/analyze_corrected_pool_check.py --input reports/corrected-pool-check/results.json.gz --output /tmp/corrected-pool-check` to reproduce. Worker: `scripts/modal_corrected_pool_check.py`; committed design: `plans/corrected-pool-check-v1.json` and `.md`.', '']
    (out / 'README.md').write_text('\n'.join(lines));plot(report, out)
    names = ['results.json.gz', 'analysis.json', 'README.md', 'corrected_pool_check.png', 'corrected_pool_check.pdf']
    save_json(out / 'manifest.json', dict(files={n: dict(bytes=(out / n).stat().st_size, sha256=file_sha(out / n)) for n in names}))
    print(json.dumps(report['splits'], indent=2))


if __name__ == '__main__':
    main()
