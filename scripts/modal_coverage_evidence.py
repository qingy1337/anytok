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
from modal_eval_rl_ood import gpu_image, training, frozen, outputs
from pool_generalization import digest, file_sha, save_json

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'modal_results/coverage_evidence'
NAME='anytok-coverage-evidence'
PROFILE='gildmember'
app=modal.App(NAME)


def now():return datetime.now(timezone.utc).isoformat()


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
    if action!='run':raise ValueError('Unknown action')
    from modal_credits import billing_row
    from modal_tokenizer_pool_transfer import source_client
    balance=billing_row(PROFILE,'this month',Decimal(30))
    if balance['status']!='ok' or float(balance['estimated_remaining_usd'])<5.5:raise ValueError('Need verified credit and safety margin')
    base=json.loads((ROOT/'modal_results/rl_ood_20k/evaluation-spec.json').read_text())
    spec={'screen_digest':digest(json.loads((ROOT/'modal_results/uniform_trajectories/screen-report.json').read_text())),
          'base_run_id':base['run_id'],'rl_manifest':base['checkpoint_manifest'],'version':1}
    spec['prefix']=base['run_id']+'/coverage-evidence-'+digest(spec)[:12]
    save_json(STATE/'spec.json',spec)
    pointer=STATE/'call.json'
    if pointer.is_file():call=modal.FunctionCall.from_id(json.loads(pointer.read_text())['call_id'])
    else:
        _,secret=source_client('neural-net-1411');call=evaluate.with_options(secrets=[secret]).spawn(spec)
        save_json(pointer,{'call_id':call.object_id,'app_id':app.app_id,'at_utc':now()})
    while True:
        try:result=call.get(timeout=10);break
        except (TimeoutError,modal.exception.TimeoutError):pass
        except (ConnectionError,modal.exception.ConnectionError):time.sleep(3)
        try:
            status=json.loads(b''.join(outputs.read_file(spec['prefix']+'/status.json')));save_json(STATE/'status.json',status)
        except (FileNotFoundError,modal.exception.NotFoundError,modal.exception.ConnectionError):pass
    save_json(STATE/'result.json',result)
    save_json(STATE/'status.json',{'at_utc':now(),'stage':'complete','completed':432,'total':432,'profile':PROFILE})
