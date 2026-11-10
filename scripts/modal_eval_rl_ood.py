"""Plan (default), then score RL20k on the unchanged uniform trajectory panel.

Use gildmember: its persistent uniform data and frozen contexts are reused
read-only. Only the three inference weight files move cloud-to-cloud.
"""
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import sys
import time
import uuid

import modal
from pool_generalization import digest, file_sha, save_json

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT/'modal_results/rl_ood_20k'
PROFILE = 'gildmember'
NAME = 'anytok-rl-ood-20k'
app = modal.App(NAME)
training = modal.Volume.from_name('anytok-pool-20k', version=2)
frozen = modal.Volume.from_name('anytok-uniform-trajectories', version=2)
outputs = modal.Volume.from_name(NAME, create_if_missing=True, version=2)
WEIGHTS = ['lm.pt', 'hypernet/config.json', 'hypernet/model.safetensors']
SOURCE = ROOT/'runtime/anytok_pool_20k'
cpu_image = modal.Image.debian_slim(python_version='3.12').add_local_python_source('pool_generalization')
gpu_image = (modal.Image.from_registry('ubuntu:24.04', add_python='3.12')
    .apt_install('libpython3.12')
    .uv_pip_install('torch==2.15.0','transformers==5.15.1','tokenizers==0.22.2',
                    'numpy==2.3.5','safetensors==0.8.0','accelerate==1.14.0')
    .env({'TOKENIZERS_PARALLELISM':'false'})
    .add_local_dir(SOURCE/'hf_hypernet','/opt/zett/hf_hypernet')
    .add_local_file(SOURCE/'rust_utils/target/release/librust_utils.so',
                    '/opt/zett/rust_utils/target/release/librust_utils.so')
    .add_local_python_source('uniform_trajectory','uniform_weakness','pool_ood_eval','pool_generalization'))


def now():return datetime.now(timezone.utc).isoformat()


def local_status(stage, completed=0, **fields):
    row={'at_utc':now(),'stage':stage,'completed':completed,'total':12,'profile':PROFILE,**fields}
    save_json(STATE/'status.json',row)
    with (STATE/'events.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


@app.function(image=cpu_image, volumes={'/rl':outputs}, timeout=1800)
def prepare(spec):
    import hashlib
    source_client=modal.Client.from_credentials(os.environ['SOURCE_MODAL_TOKEN_ID'],os.environ['SOURCE_MODAL_TOKEN_SECRET'])
    source=modal.Volume.from_name('anytok-pool-rl-20k',version=2,client=source_client)
    original=json.loads(b''.join(source.read_file('settings-run/step-020000/manifest.json')))
    if original!=spec['checkpoint_manifest']:raise ValueError('Final RL manifest changed')
    root=Path('/rl')/spec['run_id'];root.mkdir(parents=True,exist_ok=True)
    for name in WEIGHTS:
        expected=original['files'][name];target=root/'checkpoint'/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.is_file() and target.stat().st_size==expected['bytes'] and file_sha(target)==expected['sha256']:continue
        for attempt in range(4):
            partial=target.with_name(target.name+'.'+uuid.uuid4().hex+'.partial');h=hashlib.sha256()
            with partial.open('wb') as stream:
                for block in source.read_file('settings-run/step-020000/'+name):stream.write(block);h.update(block)
            actual={'bytes':partial.stat().st_size,'sha256':h.hexdigest()}
            if actual==expected:
                partial.replace(target);outputs.commit();break
            print(json.dumps({'stage':'copy_retry','file':name,'attempt':attempt+1,'actual':actual,'expected':expected}),flush=True)
            if attempt==3:raise ValueError('Final RL checkpoint copy checksum mismatch')
            time.sleep(2**attempt)
    save_json(root/'checkpoint/manifest.json',original);save_json(root/'evaluation-spec.json',spec);outputs.commit()
    return {'files':len(WEIGHTS),'bytes':sum(original['files'][n]['bytes'] for n in WEIGHTS),'hashes_verified':True}


@app.function(image=gpu_image,gpu='H200',volumes={
    '/train':training.with_mount_options(read_only=True),'/out':frozen.with_mount_options(read_only=True),'/rl':outputs},timeout=1800)
def evaluate(spec):
    sys.path.insert(0,'/opt/zett')
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from hf_hypernet.modeling_hypernet import ZettHypernet
    from hf_hypernet.smollm_setup import separate_input_embeddings_and_freeze_output
    from hf_hypernet.tokenizer_inputs import enable_byte_fallbacks
    from uniform_trajectory import grid_inputs
    from pool_ood_eval import score_documents
    from uniform_weakness import compare_scores
    root=Path('/rl')/spec['run_id']
    def emit(stage,completed=0,**fields):
        row={'at_utc':now(),'stage':stage,'completed':completed,'total':12,'profile':PROFILE,**fields}
        save_json(root/'status.json',row)
        with (root/'events.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        outputs.commit();print(json.dumps(row),flush=True)
    try:
        if (root/'complete.json').is_file():
            old=json.loads((root/'result.json').read_text());marker=json.loads((root/'complete.json').read_text())
            if marker['sha256']!=file_sha(root/'result.json') or old['evaluation_digest']!=digest(spec):raise ValueError('Completed evaluation changed')
            return old
        emit('validating unchanged tokenizer files, heldout contexts and checkpoint hashes')
        frozen_spec=json.loads(Path('/out/trajectory-spec.json').read_text())
        if (digest({k:v for k,v in frozen_spec.items() if k!='spec_sha256'})!=frozen_spec['spec_sha256']
                or frozen_spec['spec_sha256']!=spec['frozen_spec_sha256']):raise ValueError('Frozen contexts changed')
        baseline=Path('/out/checkpoints/step-020000.json')
        if file_sha(baseline)!=spec['uniform_result_sha256']:raise ValueError('Uniform reference changed')
        cp=root/'checkpoint'
        for name in WEIGHTS:
            expected=spec['checkpoint_manifest']['files'][name]
            if (cp/name).stat().st_size!=expected['bytes'] or file_sha(cp/name)!=expected['sha256']:raise ValueError('RL weights changed')
        canonical=AutoTokenizer.from_pretrained('/train/input/model',local_files_only=True)
        prepared=grid_inputs(canonical,frozen_spec)
        emit('loading RL20k LM and hypernetwork')
        model=AutoModelForCausalLM.from_pretrained('/train/input/model',local_files_only=True,dtype=torch.float32,low_cpu_mem_usage=True).to('cuda')
        separate_input_embeddings_and_freeze_output(model);model.eval()
        reference=model.get_input_embeddings().weight.detach().clone()
        model.load_state_dict(torch.load(cp/'lm.pt',map_location='cpu',weights_only=True))
        table=model.get_input_embeddings().weight.detach()
        if not torch.equal(table,reference) or not torch.equal(model.get_output_embeddings().weight,reference):raise ValueError('Canonical embedding tables changed')
        del reference
        hypernet=ZettHypernet.from_pretrained(cp/'hypernet',local_files_only=True).to('cuda').eval()
        fallbacks=enable_byte_fallbacks(hypernet,canonical);rows={};canonical_cache={}
        for case in frozen_spec['cases']:
            key=case['key'];data=prepared[key];emit('scoring fixed panel',len(rows),case=key)
            if fallbacks!=data['fallbacks']:raise ValueError('Byte fallback mapping changed')
            signature=digest(data['canonical_examples'])
            if signature not in canonical_cache:canonical_cache[signature]=score_documents(model,data['canonical_examples'],table,table)
            forms=torch.from_numpy(data['forms'].astype(np.int64)).to('cuda');priors=torch.from_numpy(data['priors']).to('cuda')
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                generated,output,_=hypernet(forms,priors,table,None,deterministic=True)
            if output is not None or not torch.isfinite(generated).all():raise ValueError('Invalid transferred embeddings')
            nats=score_documents(model,data['target_examples'],generated,table);source=canonical_cache[signature]
            rows[key]={'canonical_nats':source,'transferred_nats':nats,
                       'score':compare_scores(nats,source,case['byte_counts'],replicates=1000),
                       'context_sha256':case['context_sha256'],
                       'input_audit':{'document_ids':case['document_ids'],'byte_counts':case['byte_counts'],
                           'canonical_examples_sha256':signature,'target_examples_sha256':digest(data['target_examples']),
                           'suffix_tokens':32,'position_cap':128,'tokenizer_sha256':case['candidate']['sha256']}}
            result={'step':20000,'spec_sha256':frozen_spec['spec_sha256'],'evaluation_digest':digest(spec),
                    'cases':rows,'at_utc':now(),'platform':'Modal H200','profile':PROFILE,'method':'shared_settings_rl'}
            save_json(root/'result.json',result);outputs.commit()
            del forms,priors,generated;torch.cuda.empty_cache()
        save_json(root/'complete.json',{'sha256':file_sha(root/'result.json'),'cases':len(rows),'evaluation_digest':digest(spec)})
        emit('complete',len(rows));return result
    except BaseException as error:
        emit('failed',len(locals().get('rows',{})),error=str(error));raise


def await_result(call, run_id=None):
    while True:
        try:return call.get(timeout=10)
        except (TimeoutError,modal.exception.TimeoutError):pass
        except (ConnectionError,modal.exception.ConnectionError):time.sleep(3)
        if run_id:
            try:
                row=json.loads(b''.join(outputs.read_file(run_id+'/status.json')))
                save_json(STATE/'status.json',row)
            except (FileNotFoundError,modal.exception.NotFoundError,modal.exception.ConnectionError):pass


@app.local_entrypoint()
def main(action:str='plan'):
    if os.environ.get('MODAL_PROFILE')!=PROFILE:raise ValueError('Use gildmember for the existing immutable evaluation inputs')
    STATE.mkdir(parents=True,exist_ok=True)
    if action=='plan':
        print(json.dumps({'profile':PROFILE,'gpu':'H200','cases':12,'source':'neural-net-1411/settings-run/step-020000',
                          'frozen_panel':'anytok-uniform-trajectories/trajectory-spec.json','position_cap':128,'suffix_tokens':32,
                          'cpu_request':None,'memory_request':None}));return
    if action!='run':raise ValueError('Unknown action')
    from modal_credits import billing_row
    from modal_tokenizer_pool_transfer import source_client
    balance=billing_row(PROFILE,'this month',Decimal(30))
    if balance['status']!='ok' or float(balance['estimated_remaining_usd'])<3:raise ValueError('Insufficient verified evaluation credit')
    spec_path=STATE/'evaluation-spec.json'
    if spec_path.is_file():spec=json.loads(spec_path.read_text())
    else:
        client,_=source_client('neural-net-1411');source=modal.Volume.from_name('anytok-pool-rl-20k',version=2,client=client)
        manifest=json.loads(b''.join(source.read_file('settings-run/step-020000/manifest.json')))
        if manifest['step']!=20000 or manifest['kind']!='full_restart_checkpoint':raise ValueError('Final checkpoint is not committed')
        frozen_spec=json.loads((ROOT/'modal_results/uniform_trajectories/trajectory-spec.json').read_text())
        spec={'step':20000,'source_profile':'neural-net-1411','checkpoint_manifest':manifest,
              'frozen_spec_sha256':frozen_spec['spec_sha256'],
              'uniform_result_sha256':file_sha(ROOT/'modal_results/uniform_trajectories/checkpoints/step-020000.json')}
        spec['run_id']='shared-settings-20k-'+digest(spec)[:12];save_json(spec_path,spec)
    for phase in ['prepare','evaluate']:
        result_path=STATE/(phase+'-result.json');pointer=STATE/(phase+'-call.json')
        if result_path.is_file():continue
        if pointer.is_file():call=modal.FunctionCall.from_id(json.loads(pointer.read_text())['call_id'])
        else:
            if phase=='prepare':
                _,secret=source_client('neural-net-1411');call=prepare.with_options(secrets=[secret]).spawn(spec)
            else:call=evaluate.spawn(spec)
            save_json(pointer,{'call_id':call.object_id,'app_id':app.app_id,'phase':phase,'run_id':spec['run_id'],'at_utc':now()})
        local_status('copying RL inference weights cloud-to-cloud' if phase=='prepare' else 'starting H200 evaluation',call_id=call.object_id)
        try:
            result=await_result(call,spec['run_id'] if phase=='evaluate' else None)
        except BaseException as error:
            local_status('failed',error=str(error),phase=phase);raise
        save_json(result_path,result)
    local_status('complete',12)
