"""Fit only observed 32k–43k timings; keep 128k measurements out of fitting."""
import json
from pathlib import Path

import numpy as np

from pool_generalization import save_json, file_sha

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/vocab-scaling'
METHODS=['original','gpu_new','staged_new']
LABELS={'original':'Original, GPU only','gpu_new':'New, GPU only','staged_new':'New, CPU staging'}
COLORS={'original':'#315A84','gpu_new':'#36967B','staged_new':'#8657BC'}


def seconds(result,hn=False):
    return sum(p['seconds'] for p in result['phases'] if
               (p['phase'] in ['hypernetwork forward','lexical loss','hypernetwork backward'] if hn
                else not p['phase'].startswith('load ')))


def fit(x,y):
    coefficients=np.linalg.lstsq(np.column_stack([np.ones(len(x)),x]),y,rcond=None)[0]
    residual=np.asarray(y)-(coefficients[0]+coefficients[1]*np.asarray(x))
    denominator=float(np.square(np.asarray(y)-np.mean(y)).sum())
    return coefficients,1-float(np.square(residual).sum())/denominator if denominator else float('nan')


def analyze(rows,draw_count=2000):
    fitted={};by_method={m:[r for r in rows if r['method']==m and r['fit']] for m in METHODS}
    complete_cells={}
    for size in sorted({r['vocabulary'] for r in rows if r['fit']}):
        repeat_sets=[{r['repeat'] for r in by_method[m] if r['vocabulary']==size} for m in METHODS]
        common=set.intersection(*repeat_sets)
        if common:complete_cells[size]=sorted(common)
    by_method={m:[r for r in records if r['repeat'] in complete_cells.get(r['vocabulary'],[])]
               for m,records in by_method.items()}
    for method,records in by_method.items():
        if len({r['vocabulary'] for r in records})<3:continue
        x=np.array([r['vocabulary']/1000 for r in records]);y=np.array([seconds(r['result']) for r in records])
        co,r2=fit(x,y);hn_co,_=fit(x,[seconds(r['result'],True) for r in records])
        loo=[]
        for size in sorted(set(x)):
            mask=x!=size;c,_=fit(x[mask],y[mask]);loo.append(float(c[0]+128*c[1]))
        fitted[method]=dict(intercept_seconds=float(co[0]),seconds_per_1k=float(co[1]),prediction_128_seconds=float(co[0]+128*co[1]),
            r_squared=r2,measurements=len(records),sizes=len(set(x)),leave_one_size_out_128_range=[min(loo),max(loo)],
            hypernetwork_intercept_seconds=float(hn_co[0]),hypernetwork_seconds_per_1k=float(hn_co[1]),
            hypernetwork_prediction_128_seconds=float(hn_co[0]+128*hn_co[1]))
    rng=np.random.default_rng(43128);draws={m:[] for m in METHODS}
    target=[seconds(r['result']) for r in rows if r['method']=='staged_new' and not r['fit']]
    target_draws=[]
    lookup={(r['vocabulary'],r['repeat'],r['method']):seconds(r['result']) for r in rows if r['fit']}
    # Resample vocabulary groups, then matched repetition IDs. Every method
    # sees the same bootstrap cells, preserving pairing for slowdown ratios.
    if len(complete_cells)>=3:
        sizes=list(complete_cells)
        for _ in range(draw_count):
            chosen=rng.choice(sizes,size=len(sizes),replace=True)
            if len(set(chosen))<2:continue
            cells=[]
            for size in chosen:
                reps=complete_cells[size]
                cells.extend((int(size),int(r)) for r in rng.choice(reps,size=len(reps),replace=True))
            x=[s/1000 for s,_ in cells]
            for method in METHODS:
                c,_=fit(x,[lookup[(s,r,method)] for s,r in cells]);draws[method].append(c)
            if target:target_draws.append(float(np.mean(rng.choice(target,size=len(target),replace=True))))
    for method in fitted:
        if draws[method]:
            values=np.array(draws[method]);pred=values[:,0]+128*values[:,1]
            fitted[method]['bootstrap_95_128_seconds']=np.percentile(pred,[2.5,97.5]).tolist()
    ratios={}
    if 'original' in fitted and draws['original']:
        original=np.array(draws['original']);denom=original[:,0]+128*original[:,1]
        for method in ['gpu_new','staged_new']:
            values=np.array(draws[method]);pred=values[:,0]+128*values[:,1]
            if method in fitted:
                valid=denom>0
                ratios[method+'_predicted']=dict(value=fitted[method]['prediction_128_seconds']/fitted['original']['prediction_128_seconds'],
                    bootstrap_95=np.percentile(pred[valid]/denom[valid],[2.5,97.5]).tolist(),invalid_original_predictions=int((~valid).sum()))
        if target:
            valid=denom>0
            ratios['staged_new_measured']=dict(value=float(np.mean(target))/fitted['original']['prediction_128_seconds'],
                bootstrap_95=np.percentile(np.array(target_draws)[valid]/denom[valid],[2.5,97.5]).tolist(),invalid_original_predictions=int((~valid).sum()))
    return dict(fits=fitted,slowdown_128=ratios,measured_staged_128_seconds=target,
                fit_range=[32000,43000],target_vocab=128000,measurements=len(rows),complete=len(rows)==57,
                workload_by_vocabulary={str(r['vocabulary']):dict(scored_tokens=r['result']['inputs'].get('scored_tokens'),scored_bytes=r['result']['inputs'].get('scored_bytes')) for r in rows if 'inputs' in r['result']},
                uncertainty='Paired bootstrap over observed vocabulary groups and timing repeats, conditional on an affine model. Does not cover unknown extrapolation bias or another GPU.'),draws


def render(rows):
    REPORT.mkdir(parents=True,exist_ok=True)
    analysis,draws=analyze(rows);save_json(REPORT/'analysis.json',analysis)
    with (REPORT/'timings.csv').open('w') as f:
        f.write('vocabulary,method,repeat,fit,total_seconds,hypernetwork_seconds,peak_gpu_gib,tokenizer_index\n')
        for r in rows:f.write(f"{r['vocabulary']},{r['method']},{r['repeat']},{r['fit']},{seconds(r['result'])},{seconds(r['result'],True)},{r['result']['peak_allocated_bytes']/2**30},{r['tokenizer_index']}\n")
    plot(rows,analysis,draws)
    lines=['# Vocabulary size and training-step slowdown','',
        'This compares the original high-memory implementation, the new GPU-only implementation, and the complete lower-memory recipe with selected CPU staging. All measurements use the same local RTX PRO 6000, original step-2,000 checkpoint, original inputs and complete native tokenizer vocabularies. Production training is unchanged.','',
        '**Only native32k–43k measurements enter the fit.** The original128k step does not fit this GPU. Its timing is extrapolated, not measured. The staged128k measurements are held out of fitting.','',
        f"Status: {len(rows)} /57 timed steps; {'complete' if analysis['complete'] else 'interim results'}. Each has a separate full-workload warmup followed by an exact checkpoint/RNG/corpus reload. Allocation stack tracing is disabled for all methods. CUDA peak counters remain enabled.",'',
        '![Scaling curves](vocab_scaling.png)','','[PDF](vocab_scaling.pdf) · [Timing data](timings.csv) · [Fit data](analysis.json) · [Full trial records](results.json)','',
        '## Fit and128k projection','','The affine model is `seconds = intercept + slope × vocabulary_in_thousands`. At fixed source-piece length, hidden widths and 16 global latents, the hypernetwork’s dense work grows linearly with vocabulary. This motivates an affine fit; it does not guarantee that GPU kernels, memory traffic or CPU transfers will stay in the same regime at 128k.','',
        '| Method | Intercept (s) | Seconds /1k entries | Predicted128k (s) | Bootstrap95% interval (s) | R² |',
        '|---|---:|---:|---:|---:|---:|']
    for method,record in analysis['fits'].items():
        ci=record.get('bootstrap_95_128_seconds');interval=f'{ci[0]:.2f}–{ci[1]:.2f}' if ci else 'Pending paired cells'
        lines.append(f"| {LABELS[method]} | {record['intercept_seconds']:.3f} | {record['seconds_per_1k']:.4f} | {record['prediction_128_seconds']:.2f} | {interval} | {record['r_squared']:.3f} |")
    lines+=['','## Slowdown at 128k','','All ratios below divide by the **extrapolated original128k time**. A ratio of 1 means equal speed; 2 means twice as many seconds per step.','',
        '| Comparison | Time ratio | Bootstrap95% interval |','|---|---:|---:|']
    ratio_labels={'gpu_new_predicted':'GPU-only rewrite: both times projected','staged_new_predicted':'CPU staging: both times projected','staged_new_measured':'CPU staging measured / original projected'}
    for name,r in analysis['slowdown_128'].items():lines.append(f"| {ratio_labels[name]} | {r['value']:.2f}× | {r['bootstrap_95'][0]:.2f}–{r['bootstrap_95'][1]:.2f}× |")
    if analysis['measured_staged_128_seconds']:
        values=analysis['measured_staged_128_seconds'];lines+=['',f"Measured staged128k steps: {', '.join(f'{x:.2f}s' for x in values)}; mean {np.mean(values):.2f}s."]
    lines+=['','## Scope and uncertainty','',
        'The pool begins at 32k. The observed range is only 11k entries wide, whereas128k is 85k beyond its upper edge. The bootstrap intervals capture variation among the observed native vocabularies and repetitions under the affine assumption. **They are not a guarantee of 128k accuracy.** Leave-one-vocabulary-out prediction ranges are retained in analysis.json. No estimate is an H200/B300 throughput forecast.','',
        'Each timed step includes canonical scoring, complete hypernetwork and LM forward/backward, lexical loss, gradient clipping, the populated Adam update and any CPU transfers. Loading, example/tokenizer preparation, warmup, checkpoint reload and result logging are excluded. Kernels are warm; the allocator cache is cleared before checkpoint reload. This measures one controlled update after reload, not long-run steady-state throughput. New CPU staging is always enabled in its own curve, even at small vocabularies; it is not mixed with the GPU-only curve.','',
        'Six actual BPE/no-regex tokenizers are used:32k (#494),34k (#728),36k (#14),37k (#138),40k (#550),43k (#817). These native vocabularies are not truncated or padded. The128k staged holdout is native tokenizer #989. Different vocabulary points have different tokenizer contents. The original paired-boundary construction can slightly change the scored suffix between vocabulary points; hypernetwork-only fits are also retained to separate that from the LM work. All methods and repetitions at a given point must match example proofs, scored bytes/tokens, both losses, lexical loss and final RNG exactly.','',
        'Every run retains the original batch128, microbatch32 and128-position cap, full vocabulary-wide attention, canonical output head and lexical weight0.5. The existing losslessness study and full 128k gradient checks are linked in [the prior report](../chunked-memory/README.md). This is a performance study, not a substitute for quality evaluation.','',
        'The target vocabulary affects the hypernetwork and transferred input embeddings; the LM output head remains canonical. The exact sampling order, source hashes, all phases, peaks and warmup/reload checks are saved in design.json and results.json. Three repeats per point use shuffled vocabulary order and rotated method order. No new paid cloud job was launched.','',
        '## Reproduce','','```bash','.venv-memory/bin/python scripts/run_vocab_scaling.py --resume','.venv-memory/bin/python scripts/plot_vocab_scaling.py','```','']
    content='\n'.join(lines)
    for a,b in [('native32','native 32'),('original128','original 128'),('staged128','staged 128'),('at128','at 128'),('at32','at 32'),('by11','by 11'),('only11','only 11'),('is 85','is 85'),('and 16','and 16'),('and128','and 128'),('batch128','batch 128'),('microbatch32','microbatch 32'),('weight0.5','weight 0.5'),('/57','/ 57'),('/1k','/ 1k'),('Predicted128','Predicted 128'),('Bootstrap95','Bootstrap 95'),('ratio of1','ratio of 1'),(';2','; 2')]:content=content.replace(a,b)
    (REPORT/'README.md').write_text(content)
    save_json(REPORT/'manifest.json',dict(files={str(p.relative_to(REPORT)):dict(bytes=p.stat().st_size,sha256=file_sha(p)) for p in REPORT.rglob('*') if p.is_file() and p.name!='manifest.json'}))


def plot(rows,analysis,draws):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(str(ROOT/'monitoring/web/Manrope.ttf'))
    plt.rcParams.update({'font.family':'Manrope','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),constrained_layout=True)
    grid=np.unique(np.r_[np.linspace(32,43,80),np.linspace(43,128,160)])
    for method in METHODS:
        records=[r for r in rows if r['method']==method and r['fit']]
        x=np.array([r['vocabulary']/1000 for r in records]);y=np.array([seconds(r['result']) for r in records])
        if not len(x):continue
        panels=[axes[0,1]] if method=='staged_new' else [axes[0,0],axes[0,1],axes[1,0]]
        for ax in panels:
            ax.scatter(x,y,color=COLORS[method],s=20,alpha=.65,label=LABELS[method])
            if method in analysis['fits']:
                r=analysis['fits'][method];curve=r['intercept_seconds']+r['seconds_per_1k']*grid
                ax.plot(grid[grid<=43],curve[grid<=43],color=COLORS[method])
                if ax is not axes[1,0]:
                    ax.plot(grid[grid>=43],curve[grid>=43],color=COLORS[method],ls='--')
                    if draws[method]:
                        d=np.array(draws[method]);lower,upper=np.percentile(d[:,0,None]+d[:,1,None]*grid,[2.5,97.5],axis=0)
                        ax.fill_between(grid,lower,upper,color=COLORS[method],alpha=.10)
    for ax in axes[0]:
        ax.axvline(43,color='#89919C',ls=':',lw=1);ax.set(xlabel='Vocabulary entries · thousands',ylabel='Seconds per model/optimizer step',xlim=(30,132));ax.legend(frameon=False,fontsize=8)
    axes[0,0].set_title('GPU-only methods · measured to 43k, extrapolated to 128k')
    axes[0,1].set_title('CPU staging · prediction and held-out 128k measurement')
    target=analysis['measured_staged_128_seconds']
    if target:
        axes[0,1].scatter([128]*len(target),target,marker='*',s=90,color=COLORS['staged_new'],edgecolor='white',zorder=5,label='Measured 128k')
        axes[0,1].legend(frameon=False,fontsize=8)
    axes[1,0].set(title='Observed GPU-only timing range',xlabel='Vocabulary entries · thousands',ylabel='Seconds per model/optimizer step',xlim=(31,44));axes[1,0].legend(frameon=False,fontsize=8)
    ax=axes[1,1];keys=['gpu_new_predicted','staged_new_predicted','staged_new_measured'];names=['GPU-only\npredicted','CPU staging\npredicted','CPU staging\nmeasured']
    for i,(key,name) in enumerate(zip(keys,names)):
        if key not in analysis['slowdown_128']:continue
        r=analysis['slowdown_128'][key];low,high=r['bootstrap_95'];ax.bar(i,r['value'],color='#36967B' if i==0 else '#8657BC',alpha=.75 if i==1 else 1)
        ax.errorbar(i,r['value'],yerr=[[max(0,r['value']-low)],[max(0,high-r['value'])]],color='#59616D',capsize=4,fmt='none')
        ax.text(i,max(r['value'],high)+.07,f"{r['value']:.2f}×",ha='center',fontsize=9)
    ax.set_xticks(range(3),names);ax.axhline(1,color='#89919C',ls=':');ax.set(title='128k slowdown · versus extrapolated original',ylabel='Time ratio',ylim=(0,None))
    if not analysis['fits']:axes[0,0].text(.5,.5,'Waiting for three vocabulary sizes per method',transform=axes[0,0].transAxes,ha='center')
    for ax in axes.flat:ax.grid(axis='y',color='#E9EDF1',lw=.6);ax.set_axisbelow(True)
    fig.suptitle(f"Vocabulary scaling · {len(rows)}/57 timed steps · fit 32k–43k only",fontsize=13)
    for folder in [REPORT,ROOT/'plots']:
        for ext in ['png','pdf']:
            destination=folder/f'vocab_scaling.{ext}';temp=destination.with_name(destination.stem+f'.tmp.{ext}')
            fig.savefig(temp,dpi=170,facecolor='white');temp.replace(destination)
    plt.close(fig)


if __name__=='__main__':render(json.loads((REPORT/'results.json').read_text()))
