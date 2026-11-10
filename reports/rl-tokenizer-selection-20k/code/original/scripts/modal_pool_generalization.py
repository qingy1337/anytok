"""Prepare disjoint text, generate matched fresh tokenizers, and stage eval assets."""
from pathlib import Path
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import uuid

import modal
from pool_generalization import load_plan, file_sha, copy_disjoint_documents, save_json

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'plans/pool-generalization-v1.json'
STATE=ROOT/'modal_results/pool_generalization'
NAME='anytok-pool-generalization'
CORPUS_NAME='anytok-pool-generalization-corpus'
WHEEL=ROOT/'build/pool-v2/tokenizers-0.22.2-cp39-abi3-manylinux_2_34_x86_64.whl'
app=modal.App(NAME)
volume=modal.Volume.from_name(NAME,create_if_missing=True,version=2)
corpus=modal.Volume.from_name(CORPUS_NAME,create_if_missing=True,version=2)
original=modal.Volume.from_name('anytok-corpus',create_if_missing=True)
image=modal.Image.debian_slim(python_version='3.11').add_local_python_source('pool_generalization')
training_image=(modal.Image.debian_slim(python_version='3.11')
    .add_local_file(str(WHEEL),'/tmp/'+WHEEL.name,copy=True)
    .run_commands(f'pip install --no-deps /tmp/{WHEEL.name}')
    .env({'RAYON_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'true'})
    .add_local_python_source('tokenizer_pool','pool_generalization'))


@app.function(image=image,volumes={'/corpus':original.with_mount_options(read_only=True),'/eval':volume},
              max_containers=100,timeout=3600)
def prepare_bucket(bucket, skip, required, plan_hash):
    relative=str(Path(bucket['source_path']).relative_to('/corpus')/'train.jsonl.gz')
    target=Path('/eval/corpus')/relative;meta=target.with_suffix('.manifest.json')
    if meta.exists():
        saved=json.loads(meta.read_text())
        if saved['plan_sha256']!=plan_hash or file_sha(target)!=saved['sha256']:
            raise ValueError('Existing disjoint corpus differs from this experiment')
        return saved
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix('.partial')
    with gzip.open(Path(bucket['source_path'])/'train.jsonl.gz','rt',encoding='utf-8') as src:
        with gzip.open(temporary,'wt',encoding='utf-8',compresslevel=1) as dst:
            stats=copy_disjoint_documents(src,dst,skip,required)
    temporary.replace(target)
    saved={'bucket':bucket['name'],'source_path':'corpus/'+relative,'relative_path':relative,
           'bytes':target.stat().st_size,'sha256':file_sha(target),'plan_sha256':plan_hash,
           'skip_threshold':skip,'required_bytes':required,**stats}
    save_json(meta,saved);volume.commit()
    print(json.dumps({'stage':'disjoint_bucket_ready',**saved}),flush=True)
    return saved


@app.function(image=image,volumes={'/corpus':corpus,'/eval':volume},max_containers=100,timeout=3600)
def copy_asset(item, source_volume, destination_kind):
    from grpclib.exceptions import StreamTerminatedError
    import time
    client=modal.Client.from_credentials(os.environ['SOURCE_MODAL_TOKEN_ID'],os.environ['SOURCE_MODAL_TOKEN_SECRET'])
    source=modal.Volume.from_name(source_volume,version=2,client=client)
    relative=Path(item['destination'])
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid asset destination')
    root=Path('/corpus' if destination_kind=='corpus' else '/eval')
    destination=root/relative
    if destination.is_file() and destination.stat().st_size==item['bytes'] and file_sha(destination)==item['sha256']:
        return {'path':str(relative),'reused':True}
    destination.parent.mkdir(parents=True,exist_ok=True)
    tmp=destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.partial')
    for attempt in range(4):
        try:
            h=hashlib.sha256()
            with tmp.open('wb') as dst:
                for chunk in source.read_file(item['source']):dst.write(chunk);h.update(chunk)
            break
        except (StreamTerminatedError,OSError,TimeoutError,modal.exception.ConnectionError):
            if attempt==3:raise
            time.sleep(2**attempt)
    if tmp.stat().st_size!=item['bytes'] or h.hexdigest()!=item['sha256']:
        raise ValueError('Copied asset failed its checksum')
    tmp.replace(destination)
    (corpus if destination_kind=='corpus' else volume).commit()
    return {'path':str(relative),'reused':False}


@app.function(image=training_image,volumes={'/corpus':corpus.with_mount_options(read_only=True),'/eval':volume},
              max_containers=100,single_use_containers=True,timeout=6*3600)
def train_fresh(pair, recipe, plan_hash, provenance):
    import tokenizer_pool
    destination=Path('/eval/fresh')/f"{pair['pair']:04d}"
    saved=destination/'manifest.json'
    if saved.exists():
        result=json.loads(saved.read_text())
        if (result['config']!=pair['fresh'] or result.get('experiment_plan_sha256')!=plan_hash
                or file_sha(destination/'tokenizer.json')!=result['tokenizer_sha256']):
            raise ValueError('Existing fresh tokenizer has different provenance')
        return result
    with tempfile.TemporaryDirectory() as folder:
        folder=Path(folder)
        save_json(folder/'config.json',pair['fresh']);save_json(folder/'recipe.json',recipe)
        subprocess.run([sys.executable,tokenizer_pool.__file__,'--train-config',str(folder/'config.json'),
                        '--recipe',str(folder/'recipe.json'),'--destination',str(destination)],check=True)
    result=json.loads(saved.read_text())
    if result['tokenizer_sha256']==pair['seen_sha256']:
        raise ValueError('Fresh tokenizer duplicates its seen partner')
    if not 0<=200000000-result['total_used_bytes']<=3*len(recipe['buckets']):
        raise ValueError('Fresh tokenizer used a different corpus size')
    result.update(experiment_plan_sha256=plan_hash,paired_seen_index=pair['seen']['index'],provenance=provenance)
    save_json(saved,result);volume.commit()
    return result


def source(profile,name):
    from modal_tokenizer_pool_transfer import source_client
    client,secret=source_client(profile)
    return modal.Volume.from_name(name,version=2,client=client),secret


def metadata(src,path,destination):
    raw=b''.join(src.read_file(path))
    return raw,{'source':path,'destination':destination,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def copy_items(secret,name,items,kind='eval'):
    return list(copy_asset.with_options(secrets=[secret]).starmap([(i,name,kind) for i in items],order_outputs=False))


@app.local_entrypoint()
def main(action: str='prepare'):
    profile=os.environ.get('MODAL_PROFILE')
    if profile not in ('group-cron','trenton','remote-worker'):raise ValueError('Explicit allowed evaluation profile required')
    plan=load_plan(PLAN);STATE.mkdir(parents=True,exist_ok=True)
    with volume.batch_upload(force=True) as batch:batch.put_file(PLAN,'plan.json')
    if action=='prepare':
        if profile!='group-cron':raise ValueError('Source preparation belongs to group-cron')
        jobs=[(b,plan['skip_bytes'][b['name']],plan['required_bytes'][b['name']],plan['plan_sha256']) for b in plan['recipe']['buckets']]
        files=list(prepare_bucket.starmap(jobs,order_outputs=False))
        raw=json.dumps({'plan_sha256':plan['plan_sha256'],'files':files},indent=2).encode()
        with volume.batch_upload(force=True) as batch:batch.put_file(io.BytesIO(raw),'corpus-manifest.json')
        print(json.dumps({'stage':'corpus_ready','files':len(files)}),flush=True)
    elif action=='import-corpus':
        src,secret=source('group-cron',NAME)
        manifest=json.loads(b''.join(src.read_file('corpus-manifest.json')))
        if manifest['plan_sha256']!=plan['plan_sha256']:raise ValueError('Corpus plan differs')
        items=[{'source':f['source_path'],'destination':f['relative_path'],'bytes':f['bytes'],'sha256':f['sha256']} for f in manifest['files']]
        copy_items(secret,NAME,items,'corpus')
        with volume.batch_upload(force=True) as batch:batch.put_file(io.BytesIO(json.dumps(manifest).encode()),'corpus-manifest.json')
        print(json.dumps({'stage':'corpus_imported'}),flush=True)
    elif action=='assets':
        src,secret=source('group-cron','anytok-tokenizer-pool');items=[]
        base='pools/'+plan['source_pool_sha256'][:16]
        for pair in plan['pairs']:
            i=pair['seen']['index'];raw,m=metadata(src,f'{base}/tokenizers/{i:04d}/manifest.json',f"seen/{pair['pair']:04d}/manifest.json")
            manifest=json.loads(raw)
            if manifest['config']!=pair['seen'] or manifest['tokenizer_sha256']!=pair['seen_sha256']:
                raise ValueError('Seen tokenizer differs from the frozen selection')
            items.extend([m,{'source':f'{base}/tokenizers/{i:04d}/tokenizer.json','destination':f"seen/{pair['pair']:04d}/tokenizer.json",
                             'bytes':manifest['tokenizer_bytes'],'sha256':manifest['tokenizer_sha256']}])
        copy_items(secret,'anytok-tokenizer-pool',items)
        src,secret=source('csproject','anytok-pool-20k')
        inp=json.loads(b''.join(src.read_file('input/manifest.json')))
        items=[{'source':'input/model/'+name,'destination':'model/'+name,**inp['files']['model/'+name]}
               for name in ['config.json','model.safetensors','tokenizer.json']]
        copy_items(secret,'anytok-pool-20k',items)
        for step,profile in [(2000,'csproject'),(6000,'data-pipelines'),(12000,'deployment')]:
            src,secret=source(profile,'anytok-pool-20k');base=f'run/step-{step:06d}';dest=f'checkpoints/step-{step:06d}'
            raw,item=metadata(src,base+'/manifest.json',dest+'/manifest.json');m=json.loads(raw)
            if m['step']!=step or m['kind']!='full_restart_checkpoint':raise ValueError('Wrong frozen checkpoint')
            names=['lm.pt','hypernet/config.json','hypernet/model.safetensors']
            items=[{'source':base+'/'+n,'destination':dest+'/'+n,**m['files'][n]} for n in names]
            copy_items(secret,'anytok-pool-20k',items)
            copy_items(secret,'anytok-pool-20k',[item])
        print(json.dumps({'stage':'assets_ready'}),flush=True)
    elif action=='train':
        from modal_credits import billing_row
        from decimal import Decimal
        budget=billing_row(profile,'this month',Decimal(30))
        if budget['status']!='ok' or float(budget['estimated_remaining_usd'])<plan['cpu_cost_forecast_usd']+2:
            raise RuntimeError('Fresh tokenizer forecast exceeds available credit margin')
        provenance={'wheel_sha256':file_sha(WHEEL),'worker_sha256':file_sha(ROOT/'scripts/tokenizer_pool.py')}
        failures=[]
        jobs=[(p,plan['recipe'],plan['plan_sha256'],provenance) for p in plan['pairs']]
        for result in train_fresh.starmap(jobs,order_outputs=False,return_exceptions=True):
            if isinstance(result,BaseException):failures.append(str(result));continue
            i=result['config']['index'];save_json(STATE/f'fresh-{i:04d}.json',result)
            print(json.dumps({'stage':'fresh_complete','pair':i,'seconds':result['train_seconds']}),flush=True)
        if failures:raise RuntimeError(f'Fresh tokenizer failures are not replaced: {failures}')
        print(json.dumps({'stage':'fresh_pool_complete','count':18}),flush=True)
    else:raise ValueError('Unknown action')
