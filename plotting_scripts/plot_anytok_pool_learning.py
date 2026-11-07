"""Plot uniform-pool learning from saved logs; rerun as training progresses."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from plot_joint_hypernetwork_learning import style

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-step', type=int, default=None)
    args = parser.parse_args()
    plan = json.loads((ROOT/'plans/tokenizer-pool-1000-v1.json').read_text())
    training, evaluation = {}, {}
    for path in sorted((ROOT/'modal_results/anytok_pool_20k').glob('*-train.log')):
        for line in path.read_text(errors='replace').splitlines():
            try:
                item = json.loads(line[line.index('{'):])
            except (ValueError, TypeError):
                continue
            step = item.get('step', 0)
            if args.max_step is not None and step > args.max_step:
                continue
            if item.get('stage') == 'train':
                training[step] = item
            elif item.get('stage') in {'initial_eval', 'eval'}:
                results = item['results']
                total = sum(r['bytes'] for r in results)
                generated = sum(r['generated_bpb']*r['bytes'] for r in results)/total
                canonical = sum(r['canonical_bpb']*r['bytes'] for r in results)/total
                evaluation[step] = {'step':step,'generated_bpb':generated,
                                    'canonical_bpb':canonical,'gap_bpb':generated-canonical,
                                    'continuation_bytes':total,
                                    'examples':sum(r['examples'] for r in results)}
    rows = [evaluation[s] for s in sorted(evaluation)]
    if not rows or not training:
        raise ValueError('No training and evaluation records found')
    if len({(r['continuation_bytes'],r['examples']) for r in rows}) != 1:
        raise ValueError('Sanity-evaluation cohort changed within this run')
    output = ROOT/'plots/anytok_pool_learning'
    output.parent.mkdir(exist_ok=True)
    with output.with_suffix('.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    output.with_suffix('.json').write_text(json.dumps({
        'measurement':'Two fixed ZeTT sanity tokenizers, BOS+prefix+suffix <=128 positions; not the full transfer panel.',
        'cross_run_comparison':'Old and new sanity tokenizers differ. These curves support within-run comparisons only.',
        'eval':rows,'train_steps':len(training),
        'distinct_tokenizers':len({r['tokenizer_config_index'] for r in training.values()})},indent=2)+'\n')
    style()
    figure,axes = plt.subplots(1,2,figsize=(7.5,2.75),layout='constrained')
    ax=axes[0]
    ax.plot([r['step'] for r in rows],[r['gap_bpb'] for r in rows],color='#8657BC',marker='o',lw=1.8,ms=4)
    ax.set(title='Transfer gap is shrinking',xlabel='Optimizer step',ylabel='Transferred − canonical (BPB)')
    ax.set_ylim(bottom=0);ax.xaxis.set_major_locator(MaxNLocator(4,integer=True))
    ax.annotate(f"{rows[-1]['gap_bpb']:.3f}",(rows[-1]['step'],rows[-1]['gap_bpb']),
                xytext=(-8,10),textcoords='offset points',ha='right',color='#8657BC',fontsize=9)
    ax=axes[1]
    for algorithm,color in [('bpe','#2F8FD0'),('wordpiece','#2F8058'),('unigram','#8657BC')]:
        bins=defaultdict(list)
        for step,r in training.items():
            if plan['configs'][r['tokenizer_config_index']]['algorithm']==algorithm:
                bins[(step-1)//100].append(r)
        points=[(statistics.mean(r['step'] for r in values),
                 statistics.mean(r['loss_nats_per_token'] for r in values))
                for _,values in sorted(bins.items()) if len(values)>=5]
        if points:
            ax.plot(*zip(*points),color=color,lw=1.5,label={'bpe':'BPE','wordpiece':'WordPiece','unigram':'Unigram'}[algorithm])
    ax.set(title='All three algorithms learn',xlabel='Optimizer step',ylabel='Training LM loss (nats/token)')
    ax.xaxis.set_major_locator(MaxNLocator(4,integer=True))
    ax.legend(frameon=False,fontsize=8)
    for ax in axes:
        ax.grid(axis='y',alpha=.15,lw=.6)
    for suffix in ['.png','.pdf']:
        figure.savefig(output.with_suffix(suffix),dpi=180)
    print(output)


if __name__=='__main__':
    main()
