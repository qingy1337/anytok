"""Audit generator support, policy exposures, reward signal and paired broad scores."""
import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import random

import numpy as np
from scipy.optimize import linprog
from scipy.stats import spearmanr
from pool_generalization import digest,save_json

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'modal_results/coverage_evidence'
DOMAINS=['web','wikipedia','code','papers','math','qa','books','medical','legal']


def bucket_domain(name):
    return 'web' if name.startswith('web-') else 'wikipedia' if name.startswith('wiki-') else name.split('-')[0]


def generator_support(plan, specialty):
    buckets=plan['recipe']['buckets'];names=[b['name'] for b in buckets]
    vectors=np.asarray([[c['bucket_bytes'][n]/sum(c['bucket_bytes'].values()) for n in names] for c in plan['configs']])
    basis=np.zeros((9,len(names)))
    for i,d in enumerate(DOMAINS):
        total=sum(b['train_target_bytes'] for b in buckets if bucket_domain(b['name'])==d)
        for j,b in enumerate(buckets):
            if bucket_domain(b['name'])==d:basis[i,j]=b['train_target_bytes']/total
    language={lang:vectors[:,[i for i,n in enumerate(names) if n.endswith('-'+lang)]].sum(1) for lang in ['zh','ja']}
    samples=[{'label':'Base',**specialty['baseline']},*specialty['mixtures']]
    rows=[]
    for sample in samples:
        quota=sample['bucket_bytes'];target=np.asarray([quota[n]/sum(quota.values()) for n in names]);k=len(names)
        objective=np.r_[np.zeros(9),np.full(k,.5)]
        problem=linprog(objective,A_ub=np.r_[np.c_[basis.T,-np.eye(k)],np.c_[-basis.T,-np.eye(k)]],
                        b_ub=np.r_[target,-target],A_eq=np.r_[np.ones(9),np.zeros(k)][None,:],b_eq=[1.],bounds=(0,None),method='highs')
        if not problem.success:raise ValueError('Generator support LP failed')
        distance=.5*np.abs(vectors-target).sum(1)
        rows.append({'label':sample['label'],'scheme_minimum_mixture_tv':float(problem.fun),
                     'nearest_pool_mixture_tv':float(distance.min()),'nearest_pool_index':int(distance.argmin()),
                     'bucket_fractions':target.tolist(),'domain_fractions':(target@np.array([[bucket_domain(n)==d for d in DOMAINS] for n in names])).tolist()})
    language_idx=[names.index('web-zh'),names.index('wiki-zh')]
    theory_max=float(basis[:,language_idx].sum(1).max())
    aliases={r['label']:r['domain_fractions'] for r in rows}
    return {'bucket_names':names,'mixture_basis_rank':int(np.linalg.matrix_rank(basis)),
            'chinese35_japanese35_settings_domain_difference':float(np.abs(np.asarray(aliases['Chinese35'])-aliases['Japanese35']).max()) if 'Chinese35' in aliases and 'Japanese35' in aliases else None,
            'theoretical_max_chinese_or_japanese':theory_max,
            'observed_max_chinese':float(language['zh'].max()),'observed_max_japanese':float(language['ja'].max()),
            'max_chinese_japanese_difference':float(np.abs(language['zh']-language['ja']).max()),
            'pool_chinese':language['zh'].tolist(),'pool_japanese':language['ja'].tolist(),
            'vocab_range':[min(c['vocab_size'] for c in plan['configs']),max(c['vocab_size'] for c in plan['configs'])],
            'pool_training_bytes':plan['total_bytes_per_tokenizer'],'specialty_training_bytes':specialty['total_train_bytes_per_mixture'],
            'specialty':rows,
            'limits':'TV is distance between tokenizer-training byte mixtures, not a bound on hypernetwork generalization. Scheme LP allows all9 domain weights but keeps current fixed within-domain ratios; pool distance ignores other settings.'}


def load_training():
    rows={};mismatch=[]
    for path in sorted((ROOT/'modal_results/anytok_pool_settings_rl_20k').glob('*-train.log')):
        for line in path.read_text(errors='replace').splitlines():
            try:r=json.loads(line[line.index('{'):])
            except (ValueError,TypeError):continue
            if r.get('stage')!='train':continue
            if r['step'] in rows and rows[r['step']]['tokenizer_config_index']!=r['tokenizer_config_index']:mismatch.append(r['step'])
            rows[r['step']]=r
    if mismatch or sorted(rows)!=list(range(1,20001)):raise ValueError('Incomplete or conflicting training trajectory')
    return [rows[s] for s in sorted(rows)]


def replay_buckets(recipe,steps):
    rng=random.Random(42);weights=[b['train_target_bytes'] for b in recipe['buckets']]
    population=list(range(len(weights)))
    for _ in range(16):rng.choices(population,weights=weights,k=128)
    batches=np.asarray([np.bincount(rng.choices(population,weights=weights,k=128),minlength=len(weights)) for _ in range(steps)])
    return batches,digest(rng.getstate())


def cv_predictability(y,features,folds):
    pred=np.zeros(len(y))
    for fold in range(5):
        train=folds!=fold;test=~train
        mean=features[train].mean(0);scale=features[train].std(0);scale[scale<1e-10]=1
        x=np.c_[np.ones(train.sum()),(features[train]-mean)/scale]
        xt=np.c_[np.ones(test.sum()),(features[test]-mean)/scale]
        beta=np.linalg.solve(x.T@x+np.eye(x.shape[1])*1e-4,x.T@y[train])
        pred[test]=xt@beta
    residual=y-pred
    return {'r2':float(1-(residual@residual)/np.square(y-y.mean()).sum()),
            'rmse_bpb':float(np.sqrt(np.mean(residual**2)))}


def reward_audit(plan,rows):
    path=ROOT/'runtime/anytok_pool_settings_rl_20k/hf_hypernet/settings_policy.py'
    module_spec=importlib.util.spec_from_file_location('settings_feature_encoding',path)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    idx=np.asarray([r['tokenizer_config_index'] for r in rows]);encoded=module.settings_features(plan['configs'])[idx]
    context,rng_digest=replay_buckets(plan['recipe'],len(rows))
    time=np.eye(40)[np.arange(len(rows))//500];phase=np.arange(len(rows))//5000
    settings=np.c_[encoded,*[encoded*(phase==i)[:,None] for i in range(1,4)]]
    rng=np.random.default_rng(20261110);folds=np.empty(len(rows),dtype=int)
    for start in range(0,len(rows),500):folds[start:start+500]=rng.permutation(np.arange(500)%5)
    variants={'time_only':time,'time_and_settings':np.c_[time,settings],
              'time_and_batch_domains':np.c_[time,context],'time_settings_and_batch_domains':np.c_[time,settings,context]}
    metrics={}
    for outcome in ['policy_advantage_bpb','policy_reward_bpb']:
        y=np.asarray([r[outcome] for r in rows]);metrics[outcome]={key:cv_predictability(y,x,folds) for key,x in variants.items()}
        metrics[outcome]['std_bpb']=float(y.std())
    expected=np.array([b['train_target_bytes'] for b in plan['recipe']['buckets']],dtype=float);expected/=expected.sum()
    counts=Counter(idx.tolist());uniform=json.loads((ROOT/'modal_results/uniform_trajectories/exposures.json').read_text())['counts']
    uniform_rows={}
    for path in sorted((ROOT/'modal_results/anytok_pool_20k').glob('*-train.log')):
        for line in path.read_text(errors='replace').splitlines():
            try:r=json.loads(line[line.index('{'):])
            except (ValueError,TypeError):continue
            if r.get('stage')=='train':uniform_rows[r['step']]=r
    if sorted(uniform_rows)!=list(range(1,20001)):raise ValueError('Uniform action logs incomplete')
    uniform_idx=np.asarray([uniform_rows[i]['tokenizer_config_index'] for i in range(1,20001)])
    uniform_observed=Counter(uniform_idx.tolist())
    if any(uniform_observed[i]!=uniform[str(i)] for i in range(1000)):raise ValueError('Uniform logged actions differ from saved counts')
    names=[b['name'] for b in plan['recipe']['buckets']]
    domains={'Chinese':[i for i,n in enumerate(names) if n.endswith('-zh')],
             'Japanese':[i for i,n in enumerate(names) if n.endswith('-ja')],
             'Medical':[names.index('medical-en')]}
    joint={}
    timing={}
    for i in [616,822,416,635]:
        joint[str(i)]={name:{'rl_examples':int(context[idx==i][:,columns].sum()),
                            'uniform_examples':int(context[uniform_idx==i][:,columns].sum())}
                       for name,columns in domains.items()}
        timing[str(i)]={method:{'batches_by_5k':[int(sum(idx_array[j]==i for j in range(start,start+5000))) for start in range(0,20000,5000)],
                               'hn_lr_sum':float(sum(r['hn_lr'] for r in trajectory if r['tokenizer_config_index']==i)),
                               'lm_lr_sum':float(sum(r['lm_lr'] for r in trajectory if r['tokenizer_config_index']==i))}
                         for method,trajectory,idx_array in [('rl',rows,idx),('uniform',[uniform_rows[j] for j in range(1,20001)],uniform_idx)]}
    last=rows[-1]
    return {'metrics':metrics,'python_rng_digest':rng_digest,
            'batch_domains':[{'bucket':b['name'],'expected_examples_per_batch':float(p*128),
                             'observed_mean':float(context[:,i].mean())} for i,(b,p) in enumerate(zip(plan['recipe']['buckets'],expected))],
            'selected_examples':{str(i):{'rl':counts[i],'uniform':uniform[str(i)]} for i in [616,822,416,635]},
            'counts':[counts[i] for i in range(1000)],'final_policy':{k:v for k,v in last.items() if k.startswith('policy_')},
            'joint_tokenizer_domain_examples':joint,
            'exposure_timing_proxy':timing,
            'total_rejected_examples':sum(r['rejected_examples'] for r in rows),
            'method':'5-fold held-out-batch prediction, stratified within500-update blocks. Ridge1e-4, time bins and settings-by5k-phase interactions. Bucket counts replay seed42 after16 initial queue draws; verify against saved final checkpoint RNG before interpretation. Associations are descriptive, not causal or out-of-tokenizer validation.'}


def nonlinear_reward(plan,rows):
    """Check whether the linear audit misses useful nonlinear settings signal."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    path=ROOT/'runtime/anytok_pool_settings_rl_20k/hf_hypernet/settings_policy.py'
    ms=importlib.util.spec_from_file_location('feature_encoding',path);module=importlib.util.module_from_spec(ms);ms.loader.exec_module(module)
    indices=np.asarray([r['tokenizer_config_index'] for r in rows]);settings=module.settings_features(plan['configs'])[indices]
    context,_=replay_buckets(plan['recipe'],len(rows));step=np.arange(len(rows))[:,None]/20000
    y=np.asarray([r['policy_advantage_bpb'] for r in rows])
    rng=np.random.default_rng(20261110);folds=np.empty(len(rows),dtype=int)
    for start in range(0,len(rows),500):folds[start:start+500]=rng.permutation(np.arange(500)%5)
    per_id=rng.permutation(np.arange(1000)%5);id_folds=per_id[indices]
    result={}
    variants={'time_only':step,'time_and_settings':np.c_[step,settings],
              'time_and_batch_domains':np.c_[step,context],'time_settings_and_batch_domains':np.c_[step,settings,context]}
    for protocol,split in [('held_out_batches',folds),('held_out_tokenizers',id_folds)]:
        result[protocol]={}
        for name,x in variants.items():
            pred=np.zeros(len(y))
            for fold in range(5):
                train=split!=fold;test=~train
                model=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,min_samples_leaf=50,
                    learning_rate=.07,l2_regularization=5,random_state=20261110,early_stopping=False)
                model.fit(x[train],y[train]);pred[test]=model.predict(x[test])
            result[protocol][name]={'r2':float(1-np.square(y-pred).sum()/np.square(y-y.mean()).sum()),
                                    'rmse_bpb':float(np.sqrt(np.mean(np.square(y-pred))))}
            print('NONLINEAR',protocol,name,result[protocol][name],flush=True)
    result['method']='Histogram gradient boosting:100 trees,15 leaves,min leaf50,L2=5,5-fold CV. Descriptive prediction of observed policy advantage; unmodeled interactions or text difficulty remain. Held-out-tokenizer folds keep each pool ID entirely outside its training folds.'
    return result


def clean_case_effects():
    from analyze_uniform_trajectory import curve_bootstrap
    spec=json.loads((ROOT/'modal_results/uniform_trajectories/trajectory-spec.json').read_text())
    diagnostic=json.loads((ROOT/'modal_results/uniform_trajectories/feature-diagnostics.json').read_text())
    uniform=json.loads((ROOT/'modal_results/uniform_trajectories/checkpoints/step-020000.json').read_text())
    rl=json.loads((ROOT/'modal_results/rl_ood_20k/evaluate-result.json').read_text())
    if diagnostic['spec_sha256']!=spec['spec_sha256']:raise ValueError('Feature diagnostics use a different panel')
    rows=[]
    for case in spec['cases']:
        key=case['key'];indices=diagnostic['cases'][key]['clean_indices'];a=uniform['cases'][key];b=rl['cases'][key]
        curves=curve_bootstrap([[r['canonical_nats'][i] for i in indices] for r in [a,b]],
            [[r['transferred_nats'][i] for i in indices] for r in [a,b]],
            [case['byte_counts'][i] for i in indices],[case['cluster_ids'][i] for i in indices])
        point,boot=curves['gap'];rows.append({'case':key,'clean_passages':len(indices),
            'total_passages':len(case['document_ids']),'uniform_gap':float(point[0]),'rl_gap':float(point[1]),
            'delta_gap':float(point[1]-point[0]),'delta_gap_ci95':np.quantile(boot[1]-boot[0],[.025,.975]).tolist()})
    return rows


def feature_support_losses(audit):
    from analyze_uniform_trajectory import curve_bootstrap
    spec=json.loads((ROOT/'modal_results/uniform_trajectories/trajectory-spec.json').read_text())
    uniform=json.loads((ROOT/'modal_results/uniform_trajectories/checkpoints/step-020000.json').read_text())
    rl=json.loads((ROOT/'modal_results/rl_ood_20k/evaluate-result.json').read_text());lookup={c['key']:c for c in spec['cases']}
    result=[]
    for row in audit['rows']:
        case=lookup[row['case']];a=uniform['cases'][row['case']];b=rl['cases'][row['case']]
        if 'all_forms_present_indices' not in row:continue
        groups={'all_forms_present':row['all_forms_present_indices'],
                'all_forms_and_prior_range':row['all_forms_and_prior_range_indices'],
                'at_least_one_new_form':sorted(set(range(len(case['document_ids'])))-set(row['all_forms_present_indices']))}
        summaries={}
        for label,indices in groups.items():
            if len(indices)<2:continue
            curves=curve_bootstrap([[r['canonical_nats'][i] for i in indices] for r in [a,b]],
                [[r['transferred_nats'][i] for i in indices] for r in [a,b]],
                [case['byte_counts'][i] for i in indices],[case['cluster_ids'][i] for i in indices])
            point,boot=curves['gap']
            summaries[label]={'passages':len(indices),'uniform_gap':float(point[0]),'rl_gap':float(point[1]),
                'uniform_gap_ci95':np.quantile(boot[0],[.025,.975]).tolist(),'rl_gap_ci95':np.quantile(boot[1],[.025,.975]).tolist(),
                'delta_gap':float(point[1]-point[0]),'delta_gap_ci95':np.quantile(boot[1]-boot[0],[.025,.975]).tolist()}
        result.append({'case':row['case'],'groups':summaries})
    return result


def swap_effects(data,spec=None):
    from analyze_uniform_trajectory import curve_bootstrap
    if spec is None:spec=json.loads((ROOT/'modal_results/uniform_trajectories/trajectory-spec.json').read_text())
    if data['spec_sha256']!=spec['spec_sha256'] or set(data['cases'])!={c['key'] for c in spec['cases']}:raise ValueError('Swap inputs differ')
    labels=['uniform_lm__uniform_hn','uniform_lm__rl_hn','rl_lm__uniform_hn','rl_lm__rl_hn']
    coefficients={'hypernetwork_on_uniform_lm':[-1,1,0,0],'lm_with_uniform_hypernetwork':[-1,0,1,0],
                  'interaction':[1,-1,-1,1],'overall':[-1,0,0,1],'hypernetwork_on_rl_lm':[0,0,-1,1]}
    result=[]
    for case in spec['cases']:
        row=data['cases'][case['key']]
        if row['context_sha256']!=case['context_sha256']:raise ValueError('Swap contexts differ')
        scores=[row['scores'][name] for name in labels]
        curves=curve_bootstrap([r['canonical_nats'] for r in scores],[r['transferred_nats'] for r in scores],case['byte_counts'],case['cluster_ids'])
        effects={}
        for metric,(points,boot) in curves.items():
            effects[metric]={}
            for name,vector in coefficients.items():
                v=np.asarray(vector);effects[metric][name]={'delta':float(v@points),
                    'ci95':np.quantile(v@boot,[.025,.975]).tolist()}
        result.append({'case':case['key'],'effects':effects,
                       'combinations':{label:{metric:float(curves[metric][0][i]) for metric in curves} for i,label in enumerate(labels)}})
    return result


def paired_row(row,replicates=5000):
    a=row['scores']['uniform'];b=row['scores']['rl'];counts=np.asarray(row['byte_counts'],dtype=float)
    delta=np.asarray(b['transferred_nats'])-np.asarray(a['transferred_nats'])
    gap=delta-(np.asarray(b['canonical_nats'])-np.asarray(a['canonical_nats']))
    rng=np.random.default_rng(20261110);draws=rng.integers(len(counts),size=(replicates,len(counts)))
    denominator=counts[draws].sum(1)*math.log(2)
    return {'tokenizer':row['tokenizer'],'domain':row['domain'],
            'uniform_gap':a['score']['gap_bpb'],'rl_gap':b['score']['gap_bpb'],
            'delta_bpb':float(delta.sum()/(counts.sum()*math.log(2))),
            'delta_gap':float(gap.sum()/(counts.sum()*math.log(2))),
            'delta_bpb_ci95':np.quantile(delta[draws].sum(1)/denominator,[.025,.975]).tolist(),
            'delta_gap_ci95':np.quantile(gap[draws].sum(1)/denominator,[.025,.975]).tolist(),
            'feature_stats':row['feature_stats'],'utility':row['utility']}


def domain_effect(rows,replicates=5000):
    """Equal-case mean, resampling source articles jointly across tokenizers."""
    labels=list(dict.fromkeys(i for r in rows for i in r['document_ids']));lookup={v:i for i,v in enumerate(labels)}
    counts=np.zeros((len(rows),len(labels)));delta=counts.copy();gap=counts.copy()
    for i,row in enumerate(rows):
        a,b=row['scores']['uniform'],row['scores']['rl']
        for j,label in enumerate(row['document_ids']):
            k=lookup[label];counts[i,k]+=row['byte_counts'][j]
            delta[i,k]+=b['transferred_nats'][j]-a['transferred_nats'][j]
            gap[i,k]+=b['transferred_nats'][j]-b['canonical_nats'][j]-a['transferred_nats'][j]+a['canonical_nats'][j]
    rng=np.random.default_rng(20261110);weights=rng.multinomial(len(labels),np.full(len(labels),1/len(labels)),size=replicates)
    denom=counts@weights.T*math.log(2)
    if (denom<=0).any():raise ValueError('Empty cluster-resampled case')
    return {'cases':len(rows),'source_documents':len(labels),
            'delta_bpb':float(np.mean(delta.sum(1)/(counts.sum(1)*math.log(2)))),
            'delta_gap':float(np.mean(gap.sum(1)/(counts.sum(1)*math.log(2)))),
            'delta_bpb_ci95':np.quantile(np.mean((delta@weights.T)/denom,axis=0),[.025,.975]).tolist(),
            'delta_gap_ci95':np.quantile(np.mean((gap@weights.T)/denom,axis=0),[.025,.975]).tolist()}


def analyze_screen(result,history):
    rows=result['rows']
    if len(rows)!=432 or len({(r['tokenizer'],r['domain']) for r in rows})!=432:raise ValueError('Broad screen incomplete')
    for r in rows:
        if (len(r['document_ids'])!=48 or len(r['byte_counts'])!=48 or r['uniform_reproduction_max_error']>1e-5):raise ValueError('Changed screen panel')
        for score in r['scores'].values():
            if score['score']['continuation_tokens']!=1536 or score['score']['continuation_bytes']!=sum(r['byte_counts']):raise ValueError('Scored counts differ')
            for field,key in [('canonical_nats','canonical_bpb'),('transferred_nats','bpb')]:
                a=np.asarray(score[field]);expected=a.sum()/(sum(r['byte_counts'])*math.log(2))
                if a.shape!=(48,) or not np.isfinite(a).all() or (a<0).any() or abs(expected-score['score'][key])>1e-10:raise ValueError('Invalid paired losses')
    cases=[paired_row(r) for r in rows]
    domains={d:domain_effect([r for r in rows if r['domain']==d]) for d in sorted({r['domain'] for r in rows})}
    by_id={c['id']:c for c in result['candidates']};pool=[c for c in result['candidates'] if c['kind']=='pool']
    def stratum(name):
        c=by_id[name]
        return 'in_pool' if c['kind']=='pool' else 'outside_vocab_range' if not 32000<=c['vocab_size']<=128000 else 'unseen_in_range'
    strata={s:{d:domain_effect([r for r in rows if r['domain']==d and stratum(r['tokenizer'])==s]) for d in domains}
            for s in ['in_pool','unseen_in_range','outside_vocab_range']}
    variance={}
    names=list(by_id)
    for method in ['uniform','rl']:
        variance[method]={}
        lookup={(r['tokenizer'],r['domain']):r for r in rows}
        for metric in ['bpb','canonical_bpb','gap_bpb']:
            matrix=np.asarray([[lookup[(n,d)]['scores'][method]['score'][metric] for d in domains] for n in names])
            grand=matrix.mean();a=matrix.mean(1)-grand;b=matrix.mean(0)-grand
            total=np.square(matrix-grand).sum()
            variance[method][metric]={'domain_fraction':float(len(names)*np.square(b).sum()/total),
                'tokenizer_fraction':float(len(domains)*np.square(a).sum()/total),
                'interaction_fraction':float(np.square(matrix-grand-a[:,None]-b[None,:]).sum()/total)}
    final=history[-1];policies=[]
    uniform_counts=json.loads((ROOT/'modal_results/uniform_trajectories/exposures.json').read_text())['counts']
    for c in pool:
        index=int(c['id'].split('-')[1]);same=[r for r in rows if r['tokenizer']==c['id']]
        policies.append({'tokenizer':c['id'],'index':index,'probability':final['probabilities'][index],
            'exposures':final['counts'][index],
            'uniform_exposures':uniform_counts[str(index)],
            'log_exposure_ratio':float(np.log((final['counts'][index]+.5)/(uniform_counts[str(index)]+.5))),
            'mean_delta_gap':float(np.mean([v['delta_gap'] for v in cases if v['tokenizer']==c['id']])),
            'mean_rl_raw_bpb':float(np.mean([r['scores']['rl']['score']['bpb'] for r in same])),
            'mean_rl_gap':float(np.mean([r['scores']['rl']['score']['gap_bpb'] for r in same]))})
    correlations={field:float(spearmanr([r['probability'] for r in policies],[r[field] for r in policies]).statistic)
                  for field in ['mean_rl_raw_bpb','mean_rl_gap']}
    correlations['exposure_ratio_vs_delta_gap']=float(spearmanr([r['log_exposure_ratio'] for r in policies],[r['mean_delta_gap'] for r in policies]).statistic)
    correlations['leave_algorithm_out']={algorithm:float(spearmanr(
        [r['log_exposure_ratio'] for r in policies if by_id[r['tokenizer']]['config']['algorithm']!=algorithm],
        [r['mean_delta_gap'] for r in policies if by_id[r['tokenizer']]['config']['algorithm']!=algorithm]).statistic)
        for algorithm in ['bpe','wordpiece','unigram']}
    return {'cases':cases,'domains':domains,'policy_subset':policies,'policy_rank_correlations':correlations,
            'strata':strata,'two_way_variance':variance,
            'maximum_uniform_reproduction_error':max(r['uniform_reproduction_max_error'] for r in rows),
            'limits':'Exploratory48-tokenizer utility-selected screen, not population coverage. Per-case intervals unadjusted for432 comparisons. Domain effects are equal-case means with shared source-document bootstrap, one training seed per method. Policy correlations use only22 compression-selected pool members and nine domains, not the actual full training reward.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--local-only',action='store_true');parser.add_argument('--nonlinear',action='store_true');args=parser.parse_args()
    STATE.mkdir(exist_ok=True,parents=True)
    plan=json.loads((ROOT/'plans/tokenizer-pool-1000-v1.json').read_text())
    specialty=json.loads((ROOT/'results/specialty_48k_sweep_plan.json').read_text())
    report={'support':generator_support(plan,specialty),'reward':reward_audit(plan,load_training())}
    report['clean_case_effects']=clean_case_effects()
    if args.nonlinear:save_json(STATE/'nonlinear-reward.json',nonlinear_reward(plan,load_training()))
    if (STATE/'nonlinear-reward.json').exists():report['nonlinear_reward']=json.loads((STATE/'nonlinear-reward.json').read_text())
    if (STATE/'rng-audit.json').exists():
        audit=json.loads((STATE/'rng-audit.json').read_text())
        if audit['python_rng_digest']!=report['reward']['python_rng_digest']:raise ValueError('Corpus RNG replay differs')
        report['reward']['rng_replay_verified']=True
    if (STATE/'uniform-rng-audit.json').exists():
        audit=json.loads((STATE/'uniform-rng-audit.json').read_text())
        if audit['python_rng_digest']!=report['reward']['python_rng_digest']:raise ValueError('Uniform corpus RNG differs')
        report['reward']['uniform_rng_replay_verified']=True
    if (STATE/'feature-audit.json').exists():
        report['feature_audit']=json.loads((STATE/'feature-audit.json').read_text())
        report['feature_support_losses']=feature_support_losses(report['feature_audit'])
    if (STATE/'component-swaps.json').exists():report['component_swaps']=swap_effects(json.loads((STATE/'component-swaps.json').read_text()))
    if not args.local_only:
        result=json.loads((STATE/'result.json').read_text());history=result['policy_history']
        if history[-1]['counts']!=report['reward']['counts']:raise ValueError('Logged exposures differ from policy checkpoint')
        report['screen']=analyze_screen(result,history)
        report['policy_history']=history
        audit=json.loads((STATE/'rng-audit.json').read_text())
        if audit['python_rng_digest']!=report['reward']['python_rng_digest']:raise ValueError('Corpus RNG replay does not match actual checkpoint')
        report['reward']['rng_replay_verified']=True
    save_json(STATE/('local-analysis.json' if args.local_only else 'analysis.json'),report)
    print(json.dumps({'reward':report['reward']['metrics'],'support':[{k:r[k] for k in ['label','scheme_minimum_mixture_tv','nearest_pool_mixture_tv']} for r in report['support']['specialty']],
                      'domains':report.get('screen',{}).get('domains')},indent=2))


if __name__=='__main__':main()
