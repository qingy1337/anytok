"""Audit the complete frozen sweep, compare policy probabilities and draw plots."""
import csv
import gzip
import json
from pathlib import Path
import shutil

import numpy as np

from package_training_inputs import atomic_json, sha256
from pool_policy_ranking import compare_policy

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'modal_results/regret_pool_ranking'
REPORT = ROOT / 'reports/regret-pool-ranking-6500'
PLOT = ROOT / 'plots/regret_pool_ranking_6500'


def analyze(records, policy, manifest):
    records = sorted(records, key=lambda row: row['index'])
    if [row['index'] for row in records] != list(range(1000)):
        raise ValueError('Missing or duplicated tokenizer scores')
    p = np.asarray(policy['probabilities'], dtype=float)
    for row in records:
        if (row['checkpoint_step'] != 6500 or row['panel_sha256'] != manifest['panel_sha256']
                or len(row['batches']) != 32 or row['policy_probability'] != p[row['index']]):
            raise ValueError('Score provenance or complete batch coverage mismatch')
        for index, batch in enumerate(row['batches']):
            if (batch['index'] != index or batch['valid_examples'] != 128
                    or batch['maximum_positions'] > 128 or batch['scored_bytes'] <= 0
                    or batch['dropout_seed'] != 2026111500 + index
                    or batch['split'] != ('screen' if index < 16 else 'confirm')
                    or abs(batch['gap_bpb'] - (batch['raw_bpb'] - batch['canonical_bpb'])) > 1e-12):
                raise ValueError('Training reward protocol changed')
    gaps = np.asarray([[batch['gap_bpb'] for batch in row['batches']] for row in records])
    raw = np.asarray([[batch['raw_bpb'] for batch in row['batches']] for row in records])
    canonical = np.asarray([[batch['canonical_bpb'] for batch in row['batches']] for row in records])
    if (not np.isfinite(gaps).all() or not np.isfinite(raw).all() or not np.isfinite(canonical).all()
            or (raw < 0).any() or (canonical < 0).any()):
        raise ValueError('Nonfinite score; no tokenizer may be excluded')
    summaries = {name: compare_policy(gaps[:, selection].mean(1), p) for name, selection in
                 [('all', slice(None)), ('screen', slice(0, 16)), ('confirm', slice(16, 32))]}
    dominant = int(p.argmax()); winner = summaries['screen']['hardest_index']
    difference = gaps[winner, 16:] - gaps[dominant, 16:]
    rng = np.random.default_rng(20261115)
    draws = rng.integers(0, 16, size=(10000, 16))
    means = difference[draws].mean(1)
    interval = np.quantile(means, [.025, .975]).tolist()
    confirmation = dict(discovery_winner=winner, dominant_index=dominant,
        mean_gap_difference=float(difference.mean()), paired_batch_bootstrap_95pct=interval,
        batches=16, resamples=10000,
        interpretation=('The dominant action is also the discovery winner.' if winner == dominant else
            'Discovery winner has higher regret on confirmation batches.' if interval[0] > 0 else
            'The difference is unresolved on the confirmation batches.' if interval[1] >= 0 else
            'Discovery winner does not beat the dominant action on confirmation batches.'))
    result = dict(checkpoint_step=6500, tokenizers=1000, batches_per_tokenizer=32,
        initial_text_examples=4096, corpus_buckets=23, dropout_active=True,
        aggregation=manifest['aggregation'], panel_sha256=manifest['panel_sha256'],
        comparisons=summaries, confirmation=confirmation,
        rank_stability_spearman=float(np.corrcoef(summaries['screen']['ranks'], summaries['confirm']['ranks'])[0, 1]))
    rows = []
    for index, row in enumerate(records):
        c = row['config']
        rows.append(dict(index=index, rank=summaries['all']['ranks'][index], algorithm=c['algorithm'],
            pretokenizer=c['pretokenizer'], vocab_size=row['vocabulary'], policy_probability=p[index],
            raw_bpb=float(raw[index].mean()), canonical_bpb=float(canonical[index].mean()),
            gap_bpb=float(gaps[index].mean()), screen_gap_bpb=float(gaps[index, :16].mean()),
            confirm_gap_bpb=float(gaps[index, 16:].mean()),
            screen_rank=summaries['screen']['ranks'][index], confirm_rank=summaries['confirm']['ranks'][index],
            replacement_draws=sum(batch['rejected_examples'] for batch in row['batches'])))
    return result, sorted(rows, key=lambda row: row['rank']), gaps


def plot(result, rows, gaps):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = ROOT / 'monitoring/web/Manrope.ttf'
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'Manrope', 'font.size':9, 'axes.spines.top':False,
                         'axes.spines.right':False, 'pdf.fonttype':42})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    p = np.array([row['policy_probability'] for row in sorted(rows, key=lambda row: row['index'])])
    summary = result['comparisons']['all']; dominant = summary['dominant_index']
    order = np.argsort(-gaps.mean(1), kind='stable'); ranks = np.arange(1, 1001)
    ax = axes[0, 0]
    ax.plot(ranks, gaps.mean(1)[order], color='#87919D', lw=1)
    ax.scatter(summary['dominant_rank'], gaps[dominant].mean(), color='#36967B', s=55, zorder=3)
    ax.annotate(f'Policy favorite #{dominant}: rank {summary["dominant_rank"]}',
                (summary['dominant_rank'], gaps[dominant].mean()), xytext=(8, 9), textcoords='offset points', fontsize=8)
    ax.set(title='Measured training regret across the complete pool', xlabel='Estimated hardness rank (1 = hardest)', ylabel='Mean regret, BPB')
    ax = axes[0, 1]
    ax.plot(ranks, np.cumsum(p[order]), color='#36967B', label='Saved policy')
    ax.plot(ranks, ranks / 1000, color='#87919D', ls='--', label='Uniform sampling')
    ax.set(title='Policy probability on the hardest tokenizers', xlabel='Number of hardest tokenizers included', ylabel='Cumulative probability', ylim=(0, 1.02))
    ax.legend(frameon=False)
    ax = axes[1, 0]
    configs = {row['index']:row for row in rows}
    for algorithm, color in [('bpe','#2F8FD0'), ('wordpiece','#36967B'), ('unigram','#8657BC')]:
        ids = [i for i in range(1000) if configs[i]['algorithm'] == algorithm]
        ax.scatter(gaps[ids, :16].mean(1), gaps[ids, 16:].mean(1), s=9, alpha=.5, color=color, label=algorithm)
    low = min(gaps[:, :16].mean(1).min(), gaps[:, 16:].mean(1).min())
    high = max(gaps[:, :16].mean(1).max(), gaps[:, 16:].mean(1).max())
    ax.plot([low,high],[low,high],color='#B7B7B7',ls=':',lw=.8)
    ax.scatter(gaps[dominant,:16].mean(),gaps[dominant,16:].mean(),s=90,marker='*',color='#242424',zorder=4)
    ax.set(title='Does hardness carry over to confirmation batches?', xlabel='Discovery regret (16 batches)', ylabel='Confirmation regret (16 batches)')
    ax.legend(frameon=False,fontsize=8)
    ax = axes[1, 1]
    winner = result['confirmation']['discovery_winner']
    values = np.vstack([gaps[:,16:].mean(0), p @ gaps[:,16:], gaps[dominant,16:], gaps[winner,16:]])
    rng = np.random.default_rng(20261115);draws=rng.integers(0,16,size=(10000,16))
    means = values.mean(1);bounds=np.quantile(values[:,draws].mean(2),[.025,.975],axis=1)
    ax.errorbar(np.arange(4),means,yerr=np.vstack([means-bounds[0],bounds[1]-means]),fmt='o',color='#2F8FD0',capsize=4)
    ax.set(xticks=np.arange(4),xticklabels=['Uniform','Saved policy','Policy favorite','Discovery winner'],
           title='Confirmation regret and paired-batch uncertainty',ylabel='Mean regret, BPB')
    ax.tick_params(axis='x',labelsize=8)
    for ax in axes.flat:
        ax.grid(axis='y', color='#E9EDF1', lw=.6);ax.set_axisbelow(True)
    fig.suptitle('Frozen regret step 6,500 · all 1,000 tokenizers · actual training reward',fontsize=13)
    PLOT.parent.mkdir(exist_ok=True)
    for suffix in ['png','pdf']:
        target=PLOT.with_suffix('.'+suffix);temporary=target.with_name(target.stem+'.tmp.'+suffix)
        fig.savefig(temporary,dpi=170,facecolor='white');temporary.replace(target)
        shutil.copy2(target,REPORT/target.name)
    plt.close(fig)


def main():
    manifest=json.loads((STATE/'panel-manifest.json').read_text());policy=json.loads((STATE/'policy.json').read_text())
    if sha256(STATE/'pool-index.json') != manifest['pool_index_sha256']:
        raise ValueError('Pool index metadata changed')
    index=json.loads((STATE/'pool-index.json').read_text())
    with gzip.open(STATE/'all-scores.json.gz','rt') as stream:records=json.load(stream)
    expected={r['config']['index']:r for r in index['tokenizers']}
    for record in records:
        row=expected[record['index']]
        if record['tokenizer_sha256']!=row['tokenizer_sha256'] or record['config']!=row['config']:
            raise ValueError('Native tokenizer identity mismatch')
    result,rows,gaps=analyze(records,policy,manifest)
    REPORT.mkdir(parents=True,exist_ok=True)
    atomic_json(REPORT/'analysis.json',result)
    with (REPORT/'tokenizer-ranking.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for name in ['panel-manifest.json','policy.json','run-plan.json','checkpoint-manifest.json','all-scores.json.gz','input-verification.json','pool-index.json']:
        if (STATE/name).exists():shutil.copy2(STATE/name,REPORT/name)
    plot(result,rows,gaps)
    a=result['comparisons']['all'];c=result['confirmation']
    lines=['# Frozen regret policy versus complete pool hardness','',
        'For checkpoint comparisons on held-out text, use the [4,992-token full-pool eval and H200 launch instructions](../heldout-pool-sweep/README.md). This report documents the earlier **32-batch training-reward diagnostic**.','',
        f'Frozen checkpoint **6,500**, all **1,000** native pool tokenizers, **32 × 128** training-style examples per tokenizer. Training continued independently.','',
        f'The saved policy favorite is **#{a["dominant_index"]}**, with **{100*a["dominant_probability"]:.2f}%** probability and estimated hardness rank **{a["dominant_rank"]}/1,000**. Its mean regret is **{a["dominant_gap"]:+.6f} BPB**.','',
        f'Uniform expected regret: **{a["uniform_mean_gap"]:+.6f} BPB**. Policy expected regret: **{a["policy_expected_gap"]:+.6f} BPB**. Estimated maximum: **{a["hardest_gap"]:+.6f} BPB**, tokenizer **#{a["hardest_index"]}**. Estimated oracle-minus-policy headroom: **{a["oracle_minus_policy"]:.6f} BPB**.','',
        f'Policy probability on estimated hardest 10 / 50: **{100*a["probability_on_hardest"]["10"]:.2f}% / {100*a["probability_on_hardest"]["50"]:.2f}%**.','',
        f'Discovery winner **#{c["discovery_winner"]}** minus policy favorite on the independent 16 confirmation batches: **{c["mean_gap_difference"]:+.6f} BPB**, paired-batch bootstrap 95% interval **[{c["paired_batch_bootstrap_95pct"][0]:+.6f}, {c["paired_batch_bootstrap_95pct"][1]:+.6f}]**. {c["interpretation"]}','',
        '## What was scored','',
        'The original corpus/Python state was replayed from full step 6,000 to 6,500 and all 500 actual paired-batch hashes matched. The next 32 recipe-sampled batches were frozen, covering all 23 corpus buckets. All actions used the same primary text draws and ordered replacement banks. Original rejected-example behavior and limits were preserved.','',
        'Each action used its native prefix, a variable canonical continuation, matched canonical control, batch 128, microbatch 32 and the original 128-total-position cap. Canonical targets/bytes match each transferred pair; cuts can differ between actions exactly as in training. Canonical controls were recomputed for each action. Hypernetwork dropout stayed active with fixed independent per-batch seeds; a new embedding table was generated for every batch. No model or policy updates occurred.','',
        'Regret is byte-weighted within each update, then averaged equally across updates, matching the policy objective. Lexical loss is excluded from the policy reward. No fixed-32-token or deterministic-dropout substitute was used.','',
        '## Interpretation limits','',
        'These are finite estimates of stochastic training-distribution rewards, not exact population ranks or a held-out generalization evaluation. The all-batch maximum is selected from 1,000 noisy estimates. The discovery winner is therefore also compared on independent confirmation batches. Bootstrap intervals describe variation across these fixed training-reward batches; sequential corpus windows can be correlated. Close ranks should not be treated as decisive.','',
        '![Policy versus measured hardness](regret_pool_ranking_6500.png)','',
        '[PDF](regret_pool_ranking_6500.pdf) · [Every tokenizer](tokenizer-ranking.csv) · [Analysis](analysis.json)','',
        '| Rank | Tokenizer | Algorithm / pretokenizer | Vocab | Policy probability | Mean regret |',
        '| ---: | ---: | --- | ---: | ---: | ---: |']
    for row in rows[:10]:
        lines.append(f'| {row["rank"]} | #{row["index"]} | {row["algorithm"]} / {row["pretokenizer"]} | {row["vocab_size"]:,} | {row["policy_probability"]:.5g} | {row["gap_bpb"]:+.6f} |')
    (REPORT/'README.md').write_text('\n'.join(lines)+'\n')
    atomic_json(REPORT/'manifest.json',dict(files={str(p.relative_to(REPORT)):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in REPORT.rglob('*') if p.is_file() and p.name!='manifest.json'}))
    print(json.dumps({k:v for k,v in a.items() if k!='ranks'}))


if __name__=='__main__':main()
