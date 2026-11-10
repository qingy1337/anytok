"""Bounded H200 diagnostics on all 432 existing tokenizer/domain screen pairs."""
from datetime import datetime, timezone
from decimal import Decimal
import io
import json
import os
from pathlib import Path
import sys
import time

import modal
from pool_generalization import digest, file_sha, save_json

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'modal_results/coverage_evidence'
NAME='anytok-coverage-evidence'
PROFILE='gildmember'
app=modal.App(NAME)
training=modal.Volume.from_name('anytok-pool-20k',version=2)
frozen=modal.Volume.from_name('anytok-uniform-trajectories',version=2)
outputs=modal.Volume.from_name('anytok-rl-ood-20k',version=2)
SOURCE=ROOT/'runtime/anytok_pool_20k'
gpu_image=(modal.Image.from_registry('ubuntu:24.04',add_python='3.12')
    .apt_install('libpython3.12')
    .uv_pip_install('torch==2.15.0','transformers==5.15.1','tokenizers==0.22.2',
                    'numpy==2.3.5','safetensors==0.8.0','accelerate==1.14.0')
    .env({'TOKENIZERS_PARALLELISM':'false'})
    .add_local_dir(SOURCE/'hf_hypernet','/opt/zett/hf_hypernet')
    .add_local_file(SOURCE/'rust_utils/target/release/librust_utils.so','/opt/zett/rust_utils/target/release/librust_utils.so')
    .add_local_python_source('uniform_trajectory','uniform_weakness','pool_ood_eval','pool_generalization'))


def now():return datetime.now(timezone.utc).isoformat()


@app.function(image=gpu_image,volumes={'/train':training.with_mount_options(read_only=True),
    '/out':frozen.with_mount_options(read_only=True),'/rl':outputs},timeout=900)
def audit_features(spec):
    """Lower-bound lexical coverage using all22 evaluated in-pool vocabularies."""
    import numpy as np
    from uniform_trajectory import load_candidate
    root=Path('/rl')/spec['prefix']
    screen=json.loads(Path('/out/screen-report.json').read_text())
    candidates=[c for c in screen['candidates'] if c['kind']=='pool']
    def emit(stage,completed):
        row={'at_utc':now(),'stage':stage,'completed':completed,'total':34,'profile':PROFILE}
        save_json(root/'feature-status.json',row);outputs.commit();print(json.dumps(row),flush=True)
    arrays=[];prior_arrays=[]
    for i,candidate in enumerate(candidates):
        folder=Path('/out/features')/candidate['id'];manifest=json.loads((folder/'manifest.json').read_text())
        for name,key in [('forms.npy','forms_sha256'),('priors.npy','priors_sha256')]:
            if file_sha(folder/name)!=manifest[key]:raise ValueError('Training vocabulary features changed')
        arrays.append(np.load(folder/'forms.npy'));prior_arrays.append(np.load(folder/'priors.npy'))
        emit('reading reference training vocabularies',i+1)
    forms=np.concatenate(arrays);priors=np.concatenate(prior_arrays)
    keys=np.ascontiguousarray(forms,dtype=np.int32).view(np.dtype((np.void,forms.shape[1]*4))).ravel()
    unique,inverse=np.unique(keys,return_inverse=True)
    low=np.full(len(unique),np.inf);high=np.full(len(unique),-np.inf)
    np.minimum.at(low,inverse,priors);np.maximum.at(high,inverse,priors)
    del arrays,prior_arrays,inverse,keys,forms,priors
    frozen_spec=json.loads(Path('/out/trajectory-spec.json').read_text());rows=[]
    for i,case in enumerate(frozen_spec['cases']):
        candidate=case['candidate'];folder=Path('/out/features')/candidate['id']
        forms=np.ascontiguousarray(np.load(folder/'forms.npy'),dtype=np.int32);priors=np.load(folder/'priors.npy')
        query=forms.view(np.dtype((np.void,forms.shape[1]*4))).ravel();indices=np.searchsorted(unique,query)
        indices=np.minimum(indices,len(unique)-1);matched=unique[indices]==query
        within=matched&(priors>=low[indices]-1e-6)&(priors<=high[indices]+1e-6)
        tokenizer,_,_,_,_=load_candidate(candidate['path'],candidate['kind'])
        panel=json.loads((Path('/out/cohorts')/(case['key']+'.json')).read_text())
        if digest(panel)!=case['context_sha256']:raise ValueError('Feature audit panel changed')
        ids=[tokenizer.encode(r['prefix'],add_special_tokens=False).ids for r in panel]
        flat=np.asarray([j for sequence in ids for j in sequence],dtype=int)
        unique_ids=np.unique(flat)
        rows.append({'case':case['key'],'prefix_token_occurrences':len(flat),'unique_prefix_token_ids':len(unique_ids),
            'occurrence_form_coverage':float(matched[flat].mean()),'unique_form_coverage':float(matched[unique_ids].mean()),
            'occurrence_form_and_prior_range_coverage':float(within[flat].mean()),
            'passages_with_all_forms_present':sum(bool(matched[sequence].all()) for sequence in ids),
            'all_forms_present_indices':[j for j,sequence in enumerate(ids) if matched[sequence].all()],
            'all_forms_and_prior_range_indices':[j for j,sequence in enumerate(ids) if within[sequence].all()],
            'total_passages':len(panel)})
        emit('checking fixed confirmation prefixes',len(candidates)+i+1)
    result={'schema_version':2,'reference_pool_members':[c['id'] for c in candidates],'reference_unique_forms':len(unique),
        'rows':rows,'limits':'Lower-bound form availability in vocabularies of22 compression-selected training members, not all1000 or proof of useful gradient exposure. Form-only matches omit priors; prior-range matches require the same form and interpolation within observed min/max priors. These metrics do not measure co-occurrence/context coverage.'}
    save_json(root/'feature-audit-v2.json',result);outputs.commit();return result


def scoring_models(spec,emit):
    sys.path.insert(0,'/opt/zett')
    import torch
    from transformers import AutoModelForCausalLM
    from hf_hypernet.modeling_hypernet import ZettHypernet
    from hf_hypernet.smollm_setup import separate_input_embeddings_and_freeze_output
    from hf_hypernet.tokenizer_inputs import enable_byte_fallbacks
    from transformers import AutoTokenizer
    canonical=AutoTokenizer.from_pretrained('/train/input/model',local_files_only=True)
    inventory=json.loads(Path('/out/inventory.json').read_text())
    expected=next(r['manifest'] for r in inventory['checkpoints'] if r['manifest']['step']==20000)
    models={}
    for name,cp,manifest in [('uniform',Path('/train/run/step-020000'),expected),
                             ('rl',Path('/rl')/spec['base_run_id']/'checkpoint',spec['rl_manifest'])]:
        emit('loading '+name+'20k')
        actual=json.loads((cp/'manifest.json').read_text())
        for filename in ['lm.pt','hypernet/config.json','hypernet/model.safetensors']:
            if actual['files'][filename]!=manifest['files'][filename] or file_sha(cp/filename)!=manifest['files'][filename]['sha256']:
                raise ValueError('Model weights changed')
        model=AutoModelForCausalLM.from_pretrained('/train/input/model',local_files_only=True,dtype=torch.float32,low_cpu_mem_usage=True).to('cuda')
        separate_input_embeddings_and_freeze_output(model);model.eval();reference=model.get_input_embeddings().weight.detach().clone()
        model.load_state_dict(torch.load(cp/'lm.pt',map_location='cpu',weights_only=True));table=model.get_input_embeddings().weight.detach()
        if not torch.equal(table,reference) or not torch.equal(model.get_output_embeddings().weight,reference):raise ValueError('Canonical tables changed')
        del reference
        hypernet=ZettHypernet.from_pretrained(cp/'hypernet',local_files_only=True).to('cuda').eval()
        models[name]=(model,table,hypernet,enable_byte_fallbacks(hypernet,canonical))
    return canonical,models


@app.function(image=gpu_image,gpu='H200',volumes={'/train':training.with_mount_options(read_only=True),
    '/out':frozen.with_mount_options(read_only=True),'/rl':outputs},timeout=900)
def swap_components(spec):
    sys.path.insert(0,'/opt/zett')
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from uniform_trajectory import grid_inputs
    from pool_ood_eval import score_documents
    from uniform_weakness import compare_scores
    root=Path('/rl')/spec['prefix'];rows={}
    def emit(stage):
        row={'at_utc':now(),'stage':stage,'completed':len(rows),'total':12,'profile':PROFILE}
        save_json(root/'swap-status.json',row);outputs.commit();print(json.dumps(row),flush=True)
    canonical,models=scoring_models(spec,emit)
    frozen_spec=json.loads(Path('/out/trajectory-spec.json').read_text());prepared=grid_inputs(canonical,frozen_spec)
    uniform=json.loads(Path('/out/checkpoints/step-020000.json').read_text())
    rl=json.loads((Path('/rl')/spec['base_run_id']/'result.json').read_text())
    cache={name:{} for name in models}
    for case in frozen_spec['cases']:
        key=case['key'];data=prepared[key];emit('scoring component swaps: '+key)
        forms=torch.from_numpy(data['forms'].astype(np.int64)).to('cuda');priors=torch.from_numpy(data['priors']).to('cuda')
        generated={}
        for h,(_,table,hypernet,fallbacks) in models.items():
            if fallbacks!=data['fallbacks']:raise ValueError('Fallback mapping changed')
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):generated[h],output,_=hypernet(forms,priors,table,None,deterministic=True)
            if output is not None or not torch.isfinite(generated[h]).all():raise ValueError('Invalid swapped embeddings')
        scores={};signature=digest(data['canonical_examples'])
        for l,(model,table,_,_) in models.items():
            if signature not in cache[l]:cache[l][signature]=score_documents(model,data['canonical_examples'],table,table)
            for h in models:
                nats=score_documents(model,data['target_examples'],generated[h],table)
                score=compare_scores(nats,cache[l][signature],case['byte_counts'],replicates=1000)
                scores[l+'_lm__'+h+'_hn']={'canonical_nats':cache[l][signature],'transferred_nats':nats,'score':score}
                if l==h:
                    reference=(uniform if l=='uniform' else rl)['cases'][key]['score']
                    if max(abs(score[f]-reference[f]) for f in ['bpb','canonical_bpb','gap_bpb'])>1e-5:raise ValueError('Own-model swap control failed reproduction')
        rows[key]={'scores':scores,'context_sha256':case['context_sha256']}
        save_json(root/'component-swaps.json',{'cases':rows,'spec_sha256':frozen_spec['spec_sha256']});outputs.commit()
        del generated,forms,priors;torch.cuda.empty_cache()
    emit('complete');return {'cases':rows,'spec_sha256':frozen_spec['spec_sha256']}


@app.function(image=gpu_image,gpu='H200',volumes={
    '/train':training.with_mount_options(read_only=True),'/out':frozen.with_mount_options(read_only=True),
    '/rl':outputs},timeout=1800)
def evaluate(spec):
    sys.path.insert(0,'/opt/zett')
    import gzip
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from hf_hypernet.modeling_hypernet import ZettHypernet
    from hf_hypernet.smollm_setup import separate_input_embeddings_and_freeze_output
    from hf_hypernet.tokenizer_inputs import enable_byte_fallbacks
    from uniform_trajectory import fixed_document,load_candidate,feature_aliases
    from pool_ood_eval import examples_for,score_documents
    from uniform_weakness import compare_scores
    root=Path('/rl')/spec['prefix'];root.mkdir(parents=True,exist_ok=True)
    rows=[];started=time.monotonic()
    def emit(stage,**fields):
        row={'at_utc':now(),'stage':stage,'completed':len(rows),'total':432,'profile':PROFILE,**fields}
        save_json(root/'status.json',row)
        with (root/'events.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        outputs.commit();print(json.dumps(row),flush=True)
    try:
        emit('validating frozen screen and policy history')
        screen=json.loads(Path('/out/screen-report.json').read_text())
        if digest(screen)!=spec['screen_digest'] or len(screen['screen'])!=432 or len(screen['candidates'])!=48:
            raise ValueError('Saved screen differs')
        source_client=modal.Client.from_credentials(os.environ['SOURCE_MODAL_TOKEN_ID'],os.environ['SOURCE_MODAL_TOKEN_SECRET'])
        source=modal.Volume.from_name('anytok-pool-rl-20k',version=2,client=source_client)
        policy=[]
        for step in range(500,20001,500):
            prefix=f'settings-run/step-{step:06d}'
            manifest=json.loads(b''.join(source.read_file(prefix+'/manifest.json')))
            raw=b''.join(source.read_file(prefix+'/policy.pt'))
            import hashlib
            if hashlib.sha256(raw).hexdigest()!=manifest['files']['policy.pt']['sha256']:raise ValueError('Policy checkpoint mismatch')
            state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
            if state['updates']!=step or int(state['counts'].sum())!=step:raise ValueError('Policy counters differ')
            policy.append({'step':step,'probabilities':state['logits'].softmax(0).tolist(),'counts':state['counts'].tolist()})
        save_json(root/'policy-history.json',policy);outputs.commit()
        state=torch.load(io.BytesIO(b''.join(source.read_file('settings-run/step-020000/state.pt'))),map_location='cpu',weights_only=True)
        save_json(root/'rng-audit.json',{'step':state['step'],'python_rng_digest':digest(state['python_rng'])})
        outputs.commit();del state
        canonical=AutoTokenizer.from_pretrained('/train/input/model',local_files_only=True)
        uniform_manifest=json.loads(Path('/train/run/step-020000/manifest.json').read_text())
        inventory=json.loads(Path('/out/inventory.json').read_text())
        expected=next(r['manifest'] for r in inventory['checkpoints'] if r['manifest']['step']==20000)
        models={}
        for name,cp,manifest in [('uniform',Path('/train/run/step-020000'),uniform_manifest),
                                 ('rl',Path('/rl')/spec['base_run_id']/'checkpoint',spec['rl_manifest'])]:
            emit('loading '+name+'20k')
            for filename in ['lm.pt','hypernet/config.json','hypernet/model.safetensors']:
                target=expected if name=='uniform' else spec['rl_manifest']
                if manifest['files'][filename]!=target['files'][filename] or file_sha(cp/filename)!=target['files'][filename]['sha256']:
                    raise ValueError('Model weights changed')
            model=AutoModelForCausalLM.from_pretrained('/train/input/model',local_files_only=True,
                dtype=torch.float32,low_cpu_mem_usage=True).to('cuda')
            separate_input_embeddings_and_freeze_output(model);model.eval()
            reference=model.get_input_embeddings().weight.detach().clone()
            model.load_state_dict(torch.load(cp/'lm.pt',map_location='cpu',weights_only=True))
            table=model.get_input_embeddings().weight.detach()
            if not torch.equal(table,reference) or not torch.equal(model.get_output_embeddings().weight,reference):raise ValueError('Canonical tables changed')
            del reference
            hypernet=ZettHypernet.from_pretrained(cp/'hypernet',local_files_only=True).to('cuda').eval()
            models[name]=(model,table,hypernet,enable_byte_fallbacks(hypernet,canonical))
        raw_sources={};canonical_cache={name:{} for name in models}
        prior=[]
        if (root/'results.json').is_file():
            prior=json.loads((root/'results.json').read_text())['rows']
            if any(r['screen_digest']!=spec['screen_digest'] for r in prior):raise ValueError('Partial screen changed')
            rows.extend(prior)
        complete={(r['tokenizer'],r['domain']) for r in prior}
        for candidate in screen['candidates']:
            if time.monotonic()-started>1500:raise RuntimeError('Bounded diagnostic budget reached; partial results preserved')
            name=candidate['id']
            needed=[r for r in screen['screen'] if r['tokenizer']==name and (name,r['domain']) not in complete]
            if not needed:continue
            if file_sha(candidate['path'])!=candidate['sha256']:raise ValueError('Native tokenizer changed')
            tokenizer,_,_,_,forbidden=load_candidate(candidate['path'],candidate['kind'])
            folder=Path('/out/features')/name;manifest=json.loads((folder/'manifest.json').read_text())
            for filename,field in [('forms.npy','forms_sha256'),('priors.npy','priors_sha256'),('truncated.npy','truncated_sha256')]:
                if file_sha(folder/filename)!=manifest[field]:raise ValueError('Feature arrays changed')
            forms=np.load(folder/'forms.npy');priors=np.load(folder/'priors.npy');truncated=np.load(folder/'truncated.npy')
            aliases=feature_aliases(forms,priors,forbidden,[0],return_mask=True)['mask']
            forms_gpu=torch.from_numpy(forms.astype(np.int64)).to('cuda');priors_gpu=torch.from_numpy(priors).to('cuda')
            embeddings={}
            for method,(_,table,hypernet,fallbacks) in models.items():
                if fallbacks!=manifest['fallbacks']:raise ValueError('Fallback mapping differs')
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    generated,output,_=hypernet(forms_gpu,priors_gpu,table,None,deterministic=True)
                if output is not None or not torch.isfinite(generated).all():raise ValueError('Invalid embedding output')
                embeddings[method]=generated
            for old in needed:
                domain=old['domain'];emit('scoring paired screen',tokenizer=name,domain=domain)
                if domain not in raw_sources:
                    with gzip.open('/train/input/eval/'+domain+'.jsonl.gz','rt') as stream:
                        raw_sources[domain]={r['id']:r for r in map(json.loads,stream)}
                panel=[fixed_document(raw_sources[domain][i],canonical,tokenizer,forbidden) for i in old['screen_ids']]
                context=digest([(r['id'],r['prefix'],r['canonical_continuation_ids'],r['continuation_bytes']) for r in panel])
                if context!=old['context_sha256']:raise ValueError('Original screening contexts differ')
                target=examples_for(panel,tokenizer);control=examples_for(panel);counts=[r['continuation_bytes'] for r in panel]
                scores={}
                for method,(model,table,_,_) in models.items():
                    if context not in canonical_cache[method]:canonical_cache[method][context]=score_documents(model,control,table,table)
                    nats=score_documents(model,target,embeddings[method],table)
                    source_nats=canonical_cache[method][context]
                    scores[method]={'canonical_nats':source_nats,'transferred_nats':nats,'score':compare_scores(nats,source_nats,counts,replicates=1000)}
                reproduction=max(abs(scores['uniform']['score'][k]-old['score'][k]) for k in ['bpb','canonical_bpb','gap_bpb'])
                if reproduction>1e-5:raise ValueError(f'Uniform screen did not reproduce: {name}/{domain}/{reproduction}')
                ids=np.asarray([i for prefix,_,_ in target for i in prefix],dtype=int)
                rows.append({'tokenizer':name,'domain':domain,'screen_digest':spec['screen_digest'],
                    'context_sha256':context,'document_ids':old['screen_ids'],'byte_counts':counts,
                    'scores':scores,'uniform_reproduction_max_error':reproduction,'utility':old['utility'],
                    'prefix_tokens':{'canonical':sum(len(r[0]) for r in control),'target':len(ids)},
                    'feature_stats':{'truncated_prefix_fraction':float(truncated[ids].mean()),
                                     'aliased_prefix_fraction':float(aliases[ids].mean())}})
                complete.add((name,domain))
                if len(rows)==18:emit('paired uniform reproduction smoke passed')
            save_json(root/'results.json',{'spec_digest':digest(spec),'rows':rows,'candidates':screen['candidates']});outputs.commit()
            del forms_gpu,priors_gpu,embeddings,generated;torch.cuda.empty_cache()
        if len(rows)!=432:raise ValueError('Incomplete broad screen')
        save_json(root/'complete.json',{'sha256':file_sha(root/'results.json'),'pairs':432,'spec_digest':digest(spec)})
        emit('complete',elapsed_seconds=time.monotonic()-started)
        return {'rows':rows,'candidates':screen['candidates'],'spec_digest':digest(spec),'policy_history':policy}
    except BaseException as error:
        emit('failed',error=str(error));raise


@app.local_entrypoint()
def main(action:str='plan'):
    if os.environ.get('MODAL_PROFILE')!=PROFILE:raise ValueError('Use gildmember with the existing frozen assets')
    STATE.mkdir(parents=True,exist_ok=True)
    if action=='plan':
        print(json.dumps({'pairs':432,'models':['uniform20k','RL20k'],'gpu':'H200','timeout_seconds':1800,
                          'new_training':False,'new_tokenizer_training':False,'cpu':None,'memory':None}));return
    if action not in ['run','features','swaps']:raise ValueError('Unknown action')
    from modal_credits import billing_row
    from modal_tokenizer_pool_transfer import source_client
    balance=billing_row(PROFILE,'this month',Decimal(30))
    minimum=2.1 if action=='features' else 3.6 if action=='swaps' else 5.5
    if balance['status']!='ok' or float(balance['estimated_remaining_usd'])<minimum:raise ValueError('Need verified credit and safety margin')
    base=json.loads((ROOT/'modal_results/rl_ood_20k/evaluation-spec.json').read_text())
    spec={'screen_digest':digest(json.loads((ROOT/'modal_results/uniform_trajectories/screen-report.json').read_text())),
          'base_run_id':base['run_id'],'rl_manifest':base['checkpoint_manifest'],'version':1}
    spec['prefix']=base['run_id']+'/coverage-evidence-'+digest(spec)[:12]
    save_json(STATE/'spec.json',spec)
    if action=='swaps':
        pointer=STATE/'swaps-call.json'
        if pointer.is_file():call=modal.FunctionCall.from_id(json.loads(pointer.read_text())['call_id'])
        else:
            call=swap_components.spawn(spec);save_json(pointer,{'call_id':call.object_id,'app_id':app.app_id,'at_utc':now()})
        while True:
            try:result=call.get(timeout=10);break
            except (TimeoutError,modal.exception.TimeoutError):pass
            except (ConnectionError,modal.exception.ConnectionError):time.sleep(3)
            try:save_json(STATE/'swap-status.json',json.loads(b''.join(outputs.read_file(spec['prefix']+'/swap-status.json'))))
            except (FileNotFoundError,modal.exception.NotFoundError,modal.exception.ConnectionError):pass
        save_json(STATE/'component-swaps.json',result)
        save_json(STATE/'swap-status.json',{'at_utc':now(),'stage':'complete','completed':12,'total':12,'profile':PROFILE})
        return
    if action=='features':
        pointer=STATE/'feature-call.json'
        if pointer.is_file():call=modal.FunctionCall.from_id(json.loads(pointer.read_text())['call_id'])
        else:
            call=audit_features.spawn(spec);save_json(pointer,{'call_id':call.object_id,'app_id':app.app_id,'at_utc':now()})
        while True:
            try:result=call.get(timeout=10);break
            except (TimeoutError,modal.exception.TimeoutError):pass
            except (ConnectionError,modal.exception.ConnectionError):time.sleep(3)
            try:save_json(STATE/'feature-status.json',json.loads(b''.join(outputs.read_file(spec['prefix']+'/feature-status.json'))))
            except (FileNotFoundError,modal.exception.NotFoundError,modal.exception.ConnectionError):pass
        save_json(STATE/'feature-audit.json',result)
        save_json(STATE/'feature-status.json',{'at_utc':now(),'stage':'complete','completed':34,'total':34,'profile':PROFILE})
        return
    pointer=STATE/'call.json'
    if pointer.is_file():call=modal.FunctionCall.from_id(json.loads(pointer.read_text())['call_id'])
    else:
        _,secret=source_client('neural-net-1411');call=evaluate.with_options(secrets=[secret]).spawn(spec)
        save_json(pointer,{'call_id':call.object_id,'app_id':app.app_id,'at_utc':now()})
    while True:
        try:result=call.get(timeout=10);break
        except (TimeoutError,modal.exception.TimeoutError):pass
        except (ConnectionError,modal.exception.ConnectionError):time.sleep(3)
        except Exception as error:
            save_json(STATE/'status.json',{'at_utc':now(),'stage':'failed','completed':0,'total':432,'profile':PROFILE,'error':str(error)})
            raise
        try:
            status=json.loads(b''.join(outputs.read_file(spec['prefix']+'/status.json')));save_json(STATE/'status.json',status)
        except (FileNotFoundError,modal.exception.NotFoundError,modal.exception.ConnectionError):pass
    save_json(STATE/'result.json',result)
    save_json(STATE/'status.json',{'at_utc':now(),'stage':'complete','completed':432,'total':432,'profile':PROFILE})
