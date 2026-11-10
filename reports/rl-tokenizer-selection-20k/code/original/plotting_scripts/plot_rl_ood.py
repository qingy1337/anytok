"""Paired RL-minus-uniform BPB effects on the exact fixed tokenizer panels."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from plot_joint_hypernetwork_learning import style
from plot_uniform_trajectory import NAMES

ROOT=Path(__file__).resolve().parents[1]


def main():
    report=json.loads((ROOT/'modal_results/rl_ood_20k/comparison.json').read_text())
    rows=sorted(report['cases'],key=lambda r:['ood_weak','in_pool_weak','control'].index(r['group']))
    style();fig,axes=plt.subplots(1,2,figsize=(14,8),sharey=True)
    fig.subplots_adjust(left=.30,right=.97,top=.85,bottom=.15,wspace=.23)
    for ax,metric,title in zip(axes,['transferred','gap'],['Absolute transferred BPB','Transfer gap vs own trained canonical']):
        ax.axvline(0,color='#73808E',lw=1,ls='--')
        for i,row in enumerate(rows):
            score=row['scores'][metric];value=score['delta_rl_minus_uniform'];lo,hi=score['delta_ci95']
            color='#36967B' if value<0 else '#B57849'
            ax.errorbar(value,i,xerr=[[max(0,value-lo)],[max(0,hi-value)]],fmt='o',color=color,capsize=3,ms=5)
            if i and row['group']!=rows[i-1]['group']:ax.axhline(i-.5,color='#DADDE1',lw=.8)
        ax.set_title(title,pad=14);ax.grid(axis='x',alpha=.13);ax.set_xlabel('RL − uniform (BPB) · negative favors RL')
    axes[0].set_yticks(np.arange(len(rows)),[NAMES.get(r['candidate']['id'],r['key']) for r in rows])
    axes[0].invert_yaxis()
    fig.suptitle('RL vs uniform · 20k updates · fixed tokenizer weaknesses',fontsize=16,y=.96)
    fig.text(.30,.90,'Five OOD weaknesses · two in-pool weaknesses · five retained controls',fontsize=10,color='#73808E')
    fig.text(.30,.06,'Identical saved passages and tokenizers · 32 continuation tokens · ≤128 total positions\n95% paired source-document-cluster intervals · one training seed per method · selected panel',fontsize=9,color='#73808E')
    for suffix in ['png','pdf']:fig.savefig(ROOT/f'plots/rl_ood_20k.{suffix}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
