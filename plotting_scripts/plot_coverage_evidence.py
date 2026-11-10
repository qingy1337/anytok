"""Scientific plots for the fixed coverage, reward and broad-transfer audit."""
import json
from pathlib import Path
import tempfile
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from plot_joint_hypernetwork_learning import style,BLUE,PURPLE,GRAY

ROOT=Path(__file__).resolve().parents[1]
GREEN='#36967B';ORANGE='#B57849'
DOMAIN_NAMES={'web-en':'English','web-zh':'Chinese','web-ja':'Japanese','web-ar':'Arabic',
              'web-hi':'Hindi','code':'Code','math-en':'Math','medical-en':'Medical','legal-en':'Legal'}
DOMAIN_ORDER=list(DOMAIN_NAMES)


def save(fig,name):
    for ext in ['png','pdf']:
        target=ROOT/f'plots/{name}.{ext}'
        with tempfile.NamedTemporaryFile(dir=target.parent,suffix='.'+ext,delete=False) as stream:temporary=Path(stream.name)
        fig.savefig(temporary,dpi=180);temporary.replace(target)
    plt.close(fig)


def overview(report):
    support=report['support'];fig,axes=plt.subplots(2,2,figsize=(14.5,10.8))
    fig.subplots_adjust(left=.08,right=.97,top=.89,bottom=.11,hspace=.55,wspace=.36)
    ax=axes[0,0]
    ax.scatter(np.array(support['pool_chinese'])*100,np.array(support['pool_japanese'])*100,s=8,alpha=.20,color=GRAY,label='1,000 saved mixtures')
    bound=support['theoretical_max_chinese_or_japanese']*100
    ax.plot([0,bound],[0,bound],ls='--',lw=2,color=BLUE,label='All mixtures current generator can express')
    names=support['bucket_names']
    for c in support['specialty']:
        if c['label'] not in ['Chinese35','Japanese35']:continue
        x=sum(v for v,n in zip(c['bucket_fractions'],names) if n.endswith('-zh'))*100
        y=sum(v for v,n in zip(c['bucket_fractions'],names) if n.endswith('-ja'))*100
        ax.scatter([x],[y],s=45,color=ORANGE,zorder=3)
        offset,align=((12,-9),'left') if c['label']=='Japanese35' else ((-6,10),'right')
        ax.annotate(c['label'],(x,y),xytext=offset,textcoords='offset points',ha=align,fontsize=9)
    ax.set(xlim=(0,40),ylim=(0,40),xlabel='Chinese share of tokenizer-training bytes (%)',ylabel='Japanese share (%)',title='Language directions missing from the generator')
    ax.set_aspect('equal',adjustable='box');ax.legend(frameon=False,fontsize=7,loc='center left');ax.grid(alpha=.12)
    ax=axes[0,1];rows=support['specialty'];y=np.arange(len(rows))
    ax.barh(y-.17,[r['nearest_pool_mixture_tv'] for r in rows],height=.3,color=BLUE,label='Nearest saved pool mixture')
    ax.barh(y+.17,[r['scheme_minimum_mixture_tv'] for r in rows],height=.3,color=ORANGE,label='Best possible with current generator')
    ax.set_yticks(y,[r['label'] for r in rows]);ax.invert_yaxis();ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set(title='Finite-pool gaps versus fixed generator limits',xlabel='Minimum text-mixture mass to reassign (TV distance)')
    ax.legend(frameon=False,fontsize=7,loc='upper right');ax.grid(axis='x',alpha=.12)
    ax=axes[1,0];nonlinear=report['nonlinear_reward'];keys=['time_only','time_and_settings','time_and_batch_domains','time_settings_and_batch_domains']
    x=np.arange(4)
    for shift,protocol,color,label in [(-.18,'held_out_batches',BLUE,'Held-out batches'),(.18,'held_out_tokenizers',PURPLE,'Held-out tokenizer IDs')]:
        values=[nonlinear[protocol][k]['r2']*100 for k in keys]
        ax.bar(x+shift,values,width=.34,color=color,label=label)
        for px,v in zip(x+shift,values):ax.text(px,v+.35,f'{v:.1f}',ha='center',fontsize=7)
    ax.set_xticks(x,['Time','+ tokenizer\nsettings','+ batch\ndomains','+ both'])
    ax.set(ylabel='Held-out policy-advantage R² (%)',title='What predicts the policy reward fluctuations?')
    ax.set_ylim(0,max(nonlinear['held_out_batches'][k]['r2']*100 for k in keys)+5)
    ax.legend(frameon=False,fontsize=7);ax.grid(axis='y',alpha=.12)
    ax=axes[1,1];domains=report['screen']['domains']
    for i,d in enumerate(DOMAIN_ORDER):
        r=domains[d];v=r['delta_gap'];lo,hi=r['delta_gap_ci95']
        ax.errorbar(v,i,xerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='o',color=GREEN if v<0 else ORANGE,capsize=3)
    ax.set_yticks(np.arange(9),[DOMAIN_NAMES[d] for d in DOMAIN_ORDER]);ax.invert_yaxis()
    ax.axvline(0,color=GRAY,ls='--',lw=1);ax.grid(axis='x',alpha=.12)
    ax.set(title='Where RL changes transfer · 48 fixed tokenizers',xlabel='Mean RL − uniform transfer gap (BPB)')
    fig.suptitle('Tokenizer coverage and reward audit · evidence overview',fontsize=17,y=.97)
    fig.text(.08,.935,'Existing 20k runs · no new training · 432 paired tokenizer/domain evaluations',fontsize=11,color=GRAY)
    fig.text(.08,.045,'Mixture distances describe generator inputs, not a generalization bound. Reward prediction is descriptive.\nDomain intervals resample shared source articles; tokenizers are fixed and utility-selected. One training seed per method; exploratory comparisons.',fontsize=8,color=GRAY)
    save(fig,'coverage_evidence_overview')


def heatmap(report,result):
    metadata={c['id']:c for c in result['candidates']}
    def category(c):return 0 if c['kind']=='pool' else 2 if c['vocab_size']>128000 else 1
    candidates=sorted(result['candidates'],key=lambda c:(category(c),c['id']))
    names=[c['id'] for c in candidates];rows={(r['tokenizer'],r['domain']):r for r in report['screen']['cases']}
    arrays=[np.asarray([[rows[(n,d)][m] for d in DOMAIN_ORDER] for n in names]) for m in ['uniform_gap','rl_gap','delta_gap']]
    low=min(a.min() for a in arrays[:2]);high=max(a.max() for a in arrays[:2]);limit=max(abs(arrays[2].min()),abs(arrays[2].max()))
    fig,axes=plt.subplots(1,3,figsize=(15,15),sharey=True)
    fig.subplots_adjust(left=.20,right=.96,top=.92,bottom=.11,wspace=.30)
    for i,(ax,arr,title) in enumerate(zip(axes,arrays,['Uniform20k · transfer gap','RL20k · transfer gap','RL − uniform · gap change'])):
        image=ax.imshow(arr,aspect='auto',interpolation='nearest',cmap='RdBu_r' if i==2 else 'viridis',
                        vmin=-limit if i==2 else low,vmax=limit if i==2 else high)
        ax.set_xticks(np.arange(9),[DOMAIN_NAMES[d] for d in DOMAIN_ORDER],rotation=60,ha='right')
        ax.set_title(title,pad=14)
        for y in range(1,len(names)):
            if category(candidates[y])!=category(candidates[y-1]):ax.axhline(y-.5,color='white',lw=2)
        bar=fig.colorbar(image,ax=ax,fraction=.045,pad=.035);bar.set_label('BPB')
    axes[0].set_yticks(np.arange(len(names)),names,fontsize=7)
    fig.suptitle('All 432 paired diagnostics · identical saved screening contexts',fontsize=16,y=.975)
    fig.text(.20,.945,'Rows: 22 in-pool · 14 unseen within vocabulary range · 12 outside vocabulary range',fontsize=10,color=GRAY)
    fig.text(.20,.045,'Each cell: 48 saved documents · 32 scored continuation tokens · ≤128 total positions.\nBlue in the change panel favors RL; red favors uniform. Screening cases are exploratory, not independent confirmation.',fontsize=9,color=GRAY)
    save(fig,'coverage_evidence_heatmap')


def policy(report,plan):
    history=report['policy_history'];steps=np.r_[0,[h['step'] for h in history]]
    probs=np.asarray([np.full(1000,.001),*[h['probabilities'] for h in history]])
    configs=plan['configs'];fig,axes=plt.subplots(2,2,figsize=(14,9))
    fig.subplots_adjust(left=.08,right=.97,top=.88,bottom=.12,hspace=.55,wspace=.28)
    for ax,field,title in [(axes[0,0],'algorithm','Algorithm probability mass'),(axes[0,1],'pretokenizer','Pretokenizer probability mass')]:
        for label,color in zip(sorted({c[field] for c in configs}),[BLUE,GREEN,PURPLE]):
            mask=np.asarray([c[field]==label for c in configs]);values=probs[:,mask].sum(1)
            ax.plot(steps,values,color=color,lw=1.8,label=label)
            ax.axhline(mask.mean(),color=color,ls=':',lw=.8,alpha=.5)
        ax.set(title=title,xlabel='Training step',ylabel='Probability');ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(frameon=False);ax.grid(alpha=.12)
    ax=axes[1,0];share=np.asarray(report['support']['pool_chinese']);expected=probs@share
    actual=np.asarray([np.mean(share),*[np.dot(h['counts'],share)/h['step'] for h in history]])
    ax.plot(steps,expected,color=BLUE,label='Expected under policy');ax.plot(steps,actual,color=GREEN,label='Cumulative sampled')
    ax.axhline(share.mean(),color=GRAY,ls='--',label='Uniform pool')
    ax.set(title='Chinese share of selected tokenizer-training mixtures',xlabel='Training step',ylabel='Text fraction (Japanese matches to byte rounding)')
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(frameon=False);ax.grid(alpha=.12)
    ax=axes[1,1]
    for row in report['screen']['policy_subset']:
        c=configs[row['index']];color={'bpe':BLUE,'wordpiece':GREEN,'unigram':PURPLE}[c['algorithm']]
        ax.scatter(row['log_exposure_ratio'],row['mean_delta_gap'],color=color,s=25)
        if row['index'] in [616,822]:ax.annotate(str(row['index']),(row['log_exposure_ratio'],row['mean_delta_gap']),xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.axhline(0,color=GRAY,lw=.8);ax.axvline(0,color=GRAY,lw=.8)
    ax.set(title='More exposure versus transfer change · 22 selected members',xlabel='log(RL selections / uniform selections), smoothed',ylabel='Mean RL − uniform gap across 9 domains')
    ax.grid(alpha=.12)
    fig.suptitle('What the shared settings policy actually changed',fontsize=17,y=.97)
    fig.text(.08,.93,'Dotted lines show initial uniform group mass. Final policy and counts are read from cloud checkpoints.',fontsize=10,color=GRAY)
    fig.text(.08,.045,'Exposure is endogenous: difficult configurations can be selected more often. The scatter is descriptive, not a causal estimate of extra training.\nThe Chinese/Japanese text fractions describe tokenizer-training corpora, not the LM’s language mix.',fontsize=8,color=GRAY)
    save(fig,'coverage_evidence_policy')


def mechanism(report):
    if not report.get('component_swaps') or not report.get('feature_support_losses'):return
    fig,axes=plt.subplots(2,2,figsize=(14,10.5));fig.subplots_adjust(left=.15,right=.97,top=.89,bottom=.12,hspace=.55,wspace=.32)
    keys=['ood48-chinese35--web-zh','ood48-japanese35--web-ja','base64--medical-en','unigram-2gb--web-ja','qwen25coder--web-zh']
    labels=['Chinese35 / Chinese','Japanese35 / Japanese','Base64k / medical','Unigram150k / Japanese','QwenCoder152k / Chinese']
    lookup={r['case']:r for r in report['feature_audit']['rows']};ax=axes[0,0];y=np.arange(5)
    ax.barh(y-.17,[lookup[k]['occurrence_form_coverage'] for k in keys],height=.30,color=BLUE,label='Same surface-form encoding')
    ax.barh(y+.17,[lookup[k]['occurrence_form_and_prior_range_coverage'] for k in keys],height=.30,color=PURPLE,label='Also prior within observed range')
    ax.set_yticks(y,labels,fontsize=8);ax.invert_yaxis();ax.set_xlim(0,1.02);ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set(title='Prefix features available in just 22 training members',xlabel='Fraction of prefix-token occurrences in evaluated passages');ax.legend(frameon=False,fontsize=7,loc='upper left',bbox_to_anchor=(0,-.16));ax.grid(axis='x',alpha=.12)
    ax=axes[0,1];support={r['case']:r['groups'] for r in report['feature_support_losses']}
    primary=json.loads((ROOT/'modal_results/rl_ood_20k/comparison.json').read_text());full={r['key']:r for r in primary['cases']}
    x=np.arange(3)
    for shift,method,color in [(-.24,'uniform',GRAY),(-.08,'rl',BLUE),(.08,'uniform',GRAY),(.24,'rl',BLUE)]:
        known=shift>0;values=[];errors=[]
        for key in keys[:3]:
            if known:
                row=support[key]['all_forms_and_prior_range'];v=row[method+'_gap'];ci=row[method+'_gap_ci95']
            else:v=full[key]['scores']['gap'][method];ci=None
            values.append(v);errors.append(None if ci is None else [v-ci[0],ci[1]-v])
        ax.bar(x+shift,values,width=.15,color=color,hatch='///' if known else None,
               label=('Known features · ' if known else 'All passages · ')+method,alpha=.85)
        if known:ax.errorbar(x+shift,values,yerr=np.asarray(errors).T,fmt='none',color=color,capsize=2,lw=1)
    ax.set_xticks(x,['Chinese35','Japanese35','Base64k / medical']);ax.set(title='Residual gaps on passages with supported features',ylabel='Transfer gap (BPB)')
    ax.legend(frameon=False,fontsize=7);ax.grid(axis='y',alpha=.12)
    medical=next(r for r in report['component_swaps'] if r['case']=='base64--medical-en');ax=axes[1,0]
    combinations=['uniform_lm__uniform_hn','uniform_lm__rl_hn','rl_lm__uniform_hn','rl_lm__rl_hn']
    values=[medical['combinations'][k]['transferred'] for k in combinations]
    ax.bar(np.arange(4),values,color=[GRAY,PURPLE,ORANGE,GREEN]);ax.set_xticks(np.arange(4),['U LM\nU HN','U LM\nRL HN','RL LM\nU HN','RL LM\nRL HN'])
    for i,v in enumerate(values):ax.text(i,v+.006,f'{v:.4f}',ha='center',fontsize=8)
    ax.set(title='Medical case · actual swapped-model performance',ylabel='Absolute transferred BPB');ax.set_ylim(0,max(values)*1.13);ax.grid(axis='y',alpha=.12)
    ax=axes[1,1];effects=medical['effects']['transferred'];names=['hypernetwork_on_uniform_lm','lm_with_uniform_hypernetwork','interaction','overall']
    for i,name in enumerate(names):
        row=effects[name];v=row['delta'];lo,hi=row['ci95']
        ax.errorbar(v,i,xerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='o',color=GREEN if v<0 else ORANGE,capsize=3)
    ax.set_yticks(range(4),['Swap hypernetwork only','Swap LM only','Interaction','Actual RL − uniform']);ax.invert_yaxis();ax.axvline(0,color=GRAY,ls='--',lw=1)
    ax.set(title='Medical case · factorial decomposition',xlabel='Change in transferred BPB');ax.grid(axis='x',alpha=.12)
    fig.suptitle('Lexical availability and component-swap checks',fontsize=17,y=.97)
    fig.text(.10,.935,'Frozen original panels · same 20k models · no additional training',fontsize=11,color=GRAY)
    fig.text(.10,.045,'Availability is a lower bound from 22 training vocabularies, not proof of gradient or joint-context exposure.\nHybrid models are interventions on co-adapted components; their interaction prevents a simple additive attribution. Intervals use paired source articles.',fontsize=8,color=GRAY)
    save(fig,'coverage_evidence_mechanism')


def main():
    root=ROOT/'modal_results/coverage_evidence';report=json.loads((root/'analysis.json').read_text());result=json.loads((root/'result.json').read_text())
    style();overview(report);heatmap(report,result);policy(report,json.loads((ROOT/'plans/tokenizer-pool-1000-v1.json').read_text()));mechanism(report)


if __name__=='__main__':main()
