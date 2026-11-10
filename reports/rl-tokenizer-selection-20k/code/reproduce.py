"""Verify saved measurements and regenerate this report without cloud access."""
import argparse
from collections import Counter
from contextlib import contextmanager
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('SOURCE_DATE_EPOCH','0')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT/'code/original'
sys.path[:0]=[str(ORIGINAL/'scripts'),str(ORIGINAL/'plotting_scripts')]


def read(name):return json.loads((ROOT/name).read_text())


def csv_rows(name):
    with (ROOT/name).open(newline='') as stream:return list(csv.DictReader(stream))


def save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def save_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def close(a,b):
    np.testing.assert_allclose(a,b,rtol=0,atol=1e-10)


def verify_manifest():
    manifest=read('manifest.json')
    for name,item in manifest['files'].items():
        path=ROOT/name
        if not path.is_file() or path.stat().st_size!=item['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Report checksum mismatch: '+name)


@contextmanager
def original_layout():
    """Adapt exported metadata to the original scripts' historical paths."""
    mapping={
        'data/pool/plan.json':'plans/tokenizer-pool-1000-v1.json',
        'data/evaluation/broad-screen.json':'modal_results/coverage_evidence/result.json',
        'data/analysis/coverage.json':'modal_results/coverage_evidence/analysis.json',
        'data/evaluation/original-panel.json':'modal_results/rl_ood_20k/comparison.json',
        'data/evaluation/panel-spec.json':'modal_results/uniform_trajectories/trajectory-spec.json',
        'data/policy/uniform-counts.json':'modal_results/uniform_trajectories/exposures.json',
    }
    with tempfile.TemporaryDirectory(prefix='anytok-report-') as folder:
        root=Path(folder);(root/'plots').mkdir()
        for source,target in mapping.items():
            destination=root/target;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/source,destination)
        yield root


def audit():
    from rl_ood_eval import compare
    from analyze_coverage_evidence import domain_effect,swap_effects,generator_support,replay_buckets
    spec=read('data/evaluation/panel-spec.json')
    calculated=compare(spec,read('data/evaluation/uniform-per-document.json'),read('data/evaluation/rl-per-document.json'))
    saved=read('data/evaluation/original-panel.json')
    for a,b in zip(calculated['cases'],saved['cases']):
        if a['key']!=b['key']:raise ValueError('Case order changed')
        for metric in ['canonical','transferred','gap']:
            for field in ['uniform','rl','delta_rl_minus_uniform','delta_ci95']:close(a['scores'][metric][field],b['scores'][metric][field])
    screen=read('data/evaluation/broad-screen.json');analysis=read('data/analysis/coverage.json')
    plan=read('data/pool/plan.json')
    support=generator_support(plan,read('data/pool/ood-48k-plan.json'))
    for key in ['theoretical_max_chinese_or_japanese','observed_max_chinese','observed_max_japanese',
                'chinese35_japanese35_settings_domain_difference','pool_chinese','pool_japanese']:
        close(support[key],analysis['support'][key])
    for calculated_support,saved_support in zip(support['specialty'],analysis['support']['specialty']):
        assert calculated_support['label']==saved_support['label']
        for key in ['scheme_minimum_mixture_tv','nearest_pool_mixture_tv']:close(calculated_support[key],saved_support[key])
    _,rng_digest=replay_buckets(plan['recipe'],20000)
    for method in ['rl','uniform']:
        assert rng_digest==read(f'data/provenance/{method}-rng.json')['python_rng_digest']
    if len(screen['rows'])!=432 or len({(r['tokenizer'],r['domain']) for r in screen['rows']})!=432:raise ValueError('Incomplete broad screen')
    if len(screen['candidates'])!=48:raise ValueError('Expected48 screen tokenizers')
    for row in screen['rows']:
        if len(row['document_ids'])!=48 or row['uniform_reproduction_max_error']>1e-5:raise ValueError('Changed screening panel')
        for score in row['scores'].values():
            for field,metric in [('canonical_nats','canonical_bpb'),('transferred_nats','bpb')]:
                values=np.asarray(score[field]);assert values.shape==(48,) and np.isfinite(values).all() and (values>=0).all()
                close(values.sum()/sum(row['byte_counts'])/math.log(2),score['score'][metric])
            assert score['score']['continuation_tokens']==48*32
            assert score['score']['continuation_bytes']==sum(row['byte_counts'])
    for domain,saved_domain in analysis['screen']['domains'].items():
        calculated_domain=domain_effect([r for r in screen['rows'] if r['domain']==domain])
        for key in ['delta_bpb','delta_gap','delta_bpb_ci95','delta_gap_ci95']:close(calculated_domain[key],saved_domain[key])
    swaps=swap_effects(read('data/analysis/component-swaps.json'),spec)
    for a,b in zip(swaps,analysis['component_swaps']):
        assert a['case']==b['case']
        for metric in a['effects']:
            for effect in a['effects'][metric]:
                for field in ['delta','ci95']:close(a['effects'][metric][effect][field],b['effects'][metric][effect][field])
    history=read('data/policy/history.json')
    if [r['step'] for r in history]!=list(range(500,20001,500)):raise ValueError('Incomplete policy history')
    for row in history:
        assert len(row['probabilities'])==1000 and len(row['counts'])==1000
        close(sum(row['probabilities']),1.);assert sum(row['counts'])==row['step']
    sanity=csv_rows('data/training/sanity.csv');initial={}
    for row in sanity:
        close(float(row['bpb'])-float(row['canonical_bpb']),float(row['gap_bpb']))
        assert int(row['examples'])==92
        if int(row['step'])==0:initial[(row['method'],row['tokenizer'])]=row
    for tokenizer in ['T1','T2']:
        for field in ['bpb','canonical_bpb','continuation_bytes']:
            close(float(initial[('rl',tokenizer)][field]),float(initial[('uniform',tokenizer)][field]))
    print('Verified 12 paired cases and intervals, 432 score aggregates, 9 joint-domain intervals, 12 component swaps, 40 policy snapshots, generator support and both corpus RNG replays.')


def exploration():
    configs=read('data/pool/plan.json')['configs'];history=read('data/policy/history.json')
    trains={m:csv_rows(f'data/training/{m}.csv') for m in ['rl','uniform']}
    counts={};cumulative={}
    for method,rows in trains.items():
        assert [int(r['step']) for r in rows]==list(range(1,20001))
        counter=Counter();seen=[]
        for row in rows:counter[int(row['tokenizer_config_index'])]+=1;seen.append(len(counter))
        counts[method]=[counter[i] for i in range(1000)];cumulative[method]=seen
    if counts['rl']!=history[-1]['counts']:raise ValueError('Training actions differ from final checkpoint counts')
    saved_uniform=read('data/policy/uniform-counts.json')['counts']
    assert counts['uniform']==[saved_uniform[str(i)] for i in range(1000)]
    rows=[]
    for step in [500,1000,2000,5000,10000,15000,20000]:
        snapshot=next(h for h in history if h['step']==step)
        rows.append({'step':step,'rl_distinct':cumulative['rl'][step-1],
                     'uniform_distinct':cumulative['uniform'][step-1],
                     'uniform_expected_distinct':1000*(1-.999**step),
                     'effective_count':math.exp(-sum(p*math.log(p) for p in snapshot['probabilities'] if p)),
                     'max_probability':max(snapshot['probabilities'])})
    under=[i for i,n in enumerate(counts['rl']) if n<5]
    result={'milestones':rows,'final':{m:{'distinct':len(set(int(r['tokenizer_config_index']) for r in trains[m])),
              'updates_fewer_than_5':sum(n<5 for n in counts[m]),'updates_fewer_than_10':sum(n<10 for n in counts[m]),
              'minimum_updates':min(counts[m]),'maximum_updates':max(counts[m])} for m in trains},
              'rl_fewer_than_5_by_pretokenizer':dict(Counter(configs[i]['pretokenizer'] for i in under)),
              'limits':'Selection count means128-example training batches. Finite-pool exposure is not continuous-generator coverage. Uniform expectation is analytical; actual uniform counts are also provided. Associations do not prove exploration caused evaluation differences.'}
    per_tokenizer=[{'index':i,'algorithm':c['algorithm'],'pretokenizer':c['pretokenizer'],'vocab_size':c['vocab_size'],
                    'rl_updates':counts['rl'][i],'uniform_updates':counts['uniform'][i],
                    'final_rl_probability':history[-1]['probabilities'][i]} for i,c in enumerate(configs)]
    return result,per_tokenizer,trains,cumulative


def style():
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    # Manrope is already distributed with the parent repository's monitor.
    font=ROOT.parents[1]/'monitoring/web/Manrope.ttf'
    if font.is_file():font_manager.fontManager.addfont(font)
    family='Manrope' if font.is_file() else 'DejaVu Sans'
    plt.rcParams.update({'font.family':family,'font.size':8.5,'axes.labelsize':9,'axes.titlesize':10,
        'axes.edgecolor':'#555B65','text.color':'#333333','axes.labelcolor':'#333333',
        'figure.facecolor':'white','savefig.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'ps.fonttype':42})


def save_figure(fig,path):
    import matplotlib.pyplot as plt
    for ext in ['png','pdf']:fig.savefig(path.with_suffix('.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)


def training_plots(output,trains,cumulative,per_tokenizer):
    import matplotlib.pyplot as plt
    colors={'rl':'#8657BC','uniform':'#2F8FD0'};style()
    sanity=csv_rows('data/training/sanity.csv')
    fig,axes=plt.subplots(2,2,figsize=(12,8.4));fig.subplots_adjust(hspace=.4,wspace=.28,top=.85,bottom=.12)
    for i,tokenizer in enumerate(['T1','T2']):
        for method in ['uniform','rl']:
            rows=sorted([r for r in sanity if r['method']==method and r['tokenizer']==tokenizer and int(r['step'])>0],key=lambda r:int(r['step']))
            x=[int(r['step'])/1000 for r in rows]
            label='RL' if method=='rl' else 'Uniform'
            axes[0,i].plot(x,[float(r['bpb']) for r in rows],color=colors[method],label=label+' transferred')
            axes[0,i].plot(x,[float(r['canonical_bpb']) for r in rows],color=colors[method],ls='--',alpha=.7,label=label+' canonical')
            axes[1,i].plot(x,[float(r['gap_bpb']) for r in rows],color=colors[method],label=label)
        axes[0,i].set(title=tokenizer+' · absolute continuation BPB',ylabel='BPB');axes[0,i].legend(frameon=False,fontsize=7)
        axes[1,i].set(title=tokenizer+' · transfer gap',ylabel='Transferred − own canonical BPB');axes[1,i].legend(frameon=False,fontsize=7)
    for ax in axes.flat:ax.set_xlabel('Training updates (thousands)');ax.grid(alpha=.12)
    fig.suptitle('RL and uniform · fixed sanity evaluations during training',fontsize=16,y=.97)
    fig.text(.125,.905,'Two fixed tokenizers · 92 examples each · ≤128 total positions · variable continuation length',fontsize=10,color='#73808E')
    fig.text(.125,.035,'Curves start at 500 updates to show later differences; identical step 0 scores are included in sanity.csv.\nThese sanity curves are separate from the final 32-token OOD evaluations. Neither run establishes a consistent sanity-score advantage.',fontsize=8,color='#73808E')
    save_figure(fig,output/'training_sanity')
    fig,axes=plt.subplots(2,2,figsize=(12,8.4));fig.subplots_adjust(hspace=.45,wspace=.28,top=.86,bottom=.12)
    x=np.arange(1,20001);take=np.r_[np.arange(0,20000,50),19999]
    ax=axes[0,0]
    for method in ['uniform','rl']:ax.plot(x[take]/1000,np.asarray(cumulative[method])[take],color=colors[method],label='RL' if method=='rl' else 'Uniform')
    ax.plot(x[take]/1000,1000*(1-.999**x[take]),color='#73808E',ls=':',label='Uniform expectation')
    ax.set(title='How many different tokenizers had been sampled?',ylabel='Distinct tokenizers',xlabel='Training updates (thousands)');ax.legend(frameon=False,fontsize=7)
    ax=axes[0,1];rows=trains['rl']
    ax.plot(x[take]/1000,[float(rows[i]['policy_effective_tokenizers']) for i in take],color=colors['rl'])
    ax.axhline(1000,color=colors['uniform'],ls='--');ax.set(title='How concentrated was the RL policy?',ylabel='Effective tokenizer count · exp(entropy)',xlabel='Training updates (thousands)',ylim=(0,1050))
    distributions=read('data/policy/distributions.json');ax=axes[1,0]
    uniform_regex=distributions[0]['groups']['pretokenizer']['regex']['uniform']
    ax.plot([0]+[r['step']/1000 for r in distributions],[uniform_regex]+[r['groups']['pretokenizer']['regex']['policy'] for r in distributions],color=colors['rl'],label='RL')
    ax.axhline(uniform_regex,color=colors['uniform'],ls='--',label='Uniform');ax.set(title='Early suppression of regex pretokenization',ylabel='Probability mass',xlabel='Training updates (thousands)');ax.legend(frameon=False)
    ax=axes[1,1]
    for method in ['uniform','rl']:ax.hist([r[method+'_updates'] for r in per_tokenizer],bins=np.arange(0,66,5),color=colors[method],alpha=.5,label='RL' if method=='rl' else 'Uniform')
    ax.set(title='Training exposure across all 1,000 tokenizers',xlabel='Number of 128-example updates',ylabel='Number of tokenizers');ax.legend(frameon=False)
    for ax in axes.flat:ax.grid(alpha=.12)
    fig.suptitle('Exploration audit · early concentration persisted in exposure counts',fontsize=15,y=.97)
    fig.text(.125,.035,'RL: 157 tokenizers received fewer than 5 updates; uniform: 0. All 157 under-sampled RL tokenizers use regex pretokenization.\nExploration was uneven; these observations do not establish that it caused the final OOD results.',fontsize=8,color='#73808E')
    save_figure(fig,output/'exploration')


def final_tables():
    saved=read('data/evaluation/original-panel.json');rows=[]
    for case in saved['cases']:
        row={'case':case['key'],'group':case['group'],'documents':case['documents'],
             'continuation_tokens':case['continuation_tokens'],'continuation_bytes':case['continuation_bytes']}
        for metric in ['canonical','transferred','gap']:
            score=case['scores'][metric]
            for method in ['uniform','rl']:row[method+'_'+metric+'_bpb']=score[method]
            row['delta_'+metric+'_bpb']=score['delta_rl_minus_uniform']
            row['delta_'+metric+'_ci95_low'],row['delta_'+metric+'_ci95_high']=score['delta_ci95']
        rows.append(row)
    return rows


def write_manifest():
    files={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
           for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='manifest.json'
           and '__pycache__' not in p.parts and 'reproduced-plots' not in p.parts and p.suffix!='.pyc'}
    save_json(ROOT/'manifest.json',{'version':1,'files':files,
        'scope':'Saved numerical/configuration metadata, source snapshots, report and figures; no checkpoints or raw text corpora.',
        'source_commit':read('data/provenance/sources.json')['source_commit']})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'reproduced-plots')
    parser.add_argument('--write-derived',action='store_true',help='Maintainer: refresh exported tables, plots and manifest')
    args=parser.parse_args()
    if not args.write_derived:verify_manifest()
    audit();summary,per_tokenizer,trains,cumulative=exploration()
    if not args.write_derived:
        if summary!=read('data/analysis/exploration.json'):raise ValueError('Exploration summary differs')
    else:
        save_json(ROOT/'data/analysis/exploration.json',summary)
        save_csv(ROOT/'data/policy/exposures.csv',per_tokenizer)
        save_csv(ROOT/'data/evaluation/original-panel.csv',final_tables())
    if args.check_only:return
    output=ROOT/'plots' if args.write_derived else args.output
    output.mkdir(parents=True,exist_ok=True)
    theme=importlib.import_module('plot_joint_hypernetwork_learning');theme.style=style
    with original_layout() as root:
        for name in ['plot_rl_ood','plot_coverage_evidence']:
            module=importlib.import_module(name);module.ROOT=root;module.main()
        for source in (root/'plots').iterdir():shutil.copyfile(source,output/source.name)
    training_plots(output,trains,cumulative,per_tokenizer)
    if args.write_derived:write_manifest()
    print('Exploration verified. Saved7 PNG/PDF figure sets to '+str(output))


if __name__=='__main__':main()
