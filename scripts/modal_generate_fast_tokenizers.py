"""Generate selected tokenizer configurations with the original pool trainer.

Default action only prints a plan. Example (no generation):
    MODAL_PROFILE=gildmember modal run scripts/modal_generate_fast_tokenizers.py --indices 413

Generate explicitly with --action generate. Inputs use the original 200 MB
quotas, on gildmember/anytok-pool-20k. Outputs stay in anytok-fast-tokenizers.
The experimental fast trainer is withdrawn. Use the original pool-v2 wheel
and preserve its serialized tokenizer without runtime-setting rewrites.
No explicit CPU, memory or GPU reservations are requested.
"""
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import modal
from pool_generalization import file_sha, save_json
from tokenizer_pool import digest

ROOT = Path(__file__).resolve().parents[1]
WHEEL = ROOT/'build/pool-v2/tokenizers-0.22.2-cp39-abi3-manylinux_2_34_x86_64.whl'
STATE = ROOT/'modal_results/fast_tokenizer_generation'
app = modal.App('anytok-fast-tokenizer-generation')
source = modal.Volume.from_name('anytok-pool-20k',version=2)
outputs = modal.Volume.from_name('anytok-fast-tokenizers',version=2)
image = (modal.Image.debian_slim(python_version='3.12')
         .add_local_file(WHEEL,'/tmp/reference.whl',copy=True)
         .run_commands("python -c \"import zipfile; zipfile.ZipFile('/tmp/reference.whl').extractall('/opt/reference')\"")
         .env({'PYTHONPATH':'/opt/reference','RAYON_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'true'})
         .add_local_python_source('tokenizer_pool','tokenizer_speed','pool_generalization'))


@app.function(image=image,volumes={'/train':source.with_mount_options(read_only=True),'/out':outputs},
              max_containers=100,single_use_containers=True,timeout=21600)
def generate_one(config, runtime):
    """Reusable worker: a complete generator config in, small artifact metadata out."""
    import tokenizer_speed
    if sum(config['bucket_bytes'].values()) != 200_000_000:
        raise ValueError('Retain the original 200 MB training quotas')
    if (file_sha('/train/input/corpus/manifest.json') != runtime['corpus_manifest_sha256']
            or file_sha('/train/input/recipe.json') != runtime['recipe_sha256']):
        raise ValueError('Persistent input provenance changed')
    identity = digest(config)
    folder = Path('/out/generated')/runtime['id']/identity
    folder.mkdir(parents=True,exist_ok=True)
    marker = folder/'complete.json'
    if marker.exists():
        row = json.loads(marker.read_text())
        if row['runtime'] != runtime or row['manifest']['config'] != config or file_sha(folder/'tokenizer.json') != row['manifest']['tokenizer_sha256']:
            raise ValueError('Saved output has different provenance or checksum')
        return row
    recipe = json.loads(Path('/train/input/recipe.json').read_text())
    for bucket in recipe['buckets']: bucket['source_path']='/train/input/corpus/'+bucket['name']
    config_path,recipe_path = folder/'config.json',folder/'recipe.json'
    save_json(config_path,config);save_json(recipe_path,recipe)
    print(json.dumps({'stage':'training','config_index':config['index'],'artifact_dir':str(folder)}),flush=True)
    with (folder/'process.log').open('w') as stream:
        result = subprocess.run([sys.executable,tokenizer_speed.__file__,'--config',str(config_path),
                                 '--recipe',str(recipe_path),'--destination',str(folder),
                                 '--heldout-root','/train/input/eval'],stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        failure={'returncode':result.returncode,'tail':(folder/'process.log').read_text(errors='replace')[-4000:]}
        save_json(folder/'failure.json',failure);outputs.commit()
        raise RuntimeError(f'Configuration {config["index"]} failed: {failure}')
    row=json.loads((folder/'benchmark.json').read_text())
    row.update({'runtime':runtime,'artifact_dir':str(folder),'config_sha256':identity,
                'volume_path':f'anytok-fast-tokenizers/generated/{runtime["id"]}/{identity}/tokenizer.json'})
    save_json(marker,row);outputs.commit()
    return row


@app.function(image=image,timeout=21600)
def generate_selection(configs,runtime):
    return list(generate_one.starmap([(config,runtime) for config in configs],order_outputs=False))


@app.local_entrypoint()
def main(indices: str = '',action: str = 'plan',plan_path: str = 'plans/tokenizer-pool-1000-v1.json'):
    if os.environ.get('MODAL_PROFILE') != 'gildmember': raise ValueError('Explicit gildmember profile required')
    if action not in ['plan','generate']:raise ValueError('Expected plan or generate')
    path=Path(plan_path)
    plan=json.loads(path.read_text())
    selected=list(dict.fromkeys(int(i) for i in indices.split(',') if i.strip()))
    by_index={c['index']:c for c in plan['configs']}
    if not selected or not set(selected)<=by_index.keys():raise ValueError('Specify saved configuration indices')
    configs=[by_index[i] for i in selected]
    if any(sum(c['bucket_bytes'].values())!=200_000_000 for c in configs):
        raise ValueError('Retain the original 200 MB per tokenizer')
    runtime={'trainer':'original-pool-v2','wheel_sha256':file_sha(WHEEL),'worker_sha256':file_sha(ROOT/'scripts/tokenizer_speed.py'),
             'trainer_sha256':file_sha(ROOT/'scripts/tokenizer_pool.py'),'launcher_sha256':file_sha(__file__),
             'source_volume':'anytok-pool-20k','rayon_threads':4}
    runtime['corpus_manifest_sha256']=hashlib.sha256(b''.join(source.read_file('input/corpus/manifest.json'))).hexdigest()
    runtime['recipe_sha256']=hashlib.sha256(b''.join(source.read_file('input/recipe.json'))).hexdigest()
    runtime['id']=digest(runtime)[:16]
    selection=digest({'runtime':runtime,'configs':configs})[:16]
    if action=='plan':
        print(json.dumps({'selection':selection,'indices':selected,'tokenizers':len(configs),
                          'training_bytes_each':200_000_000,'runtime':runtime,
                          'output_volume':'gildmember/anytok-fast-tokenizers',
                          'output_prefix':'generated/'+runtime['id']},indent=2));return
    STATE.mkdir(parents=True,exist_ok=True)
    pointer=STATE/f'{selection}-call.json'
    if pointer.exists():
        saved=json.loads(pointer.read_text());call=modal.FunctionCall.from_id(saved['call_id'])
    else:
        from modal_credits import billing_row
        balance=billing_row('gildmember','this month',Decimal(30))
        if balance['status']!='ok' or float(balance['estimated_remaining_usd'])<6:
            raise RuntimeError('Verified $6 credit margin required for a new launch')
        prefix='generated/'+runtime['id']
        selection_plan=STATE/f'{selection}-plan.json'
        save_json(selection_plan,{'configs':configs,'runtime':runtime,'selection':selection})
        with outputs.batch_upload(force=True) as batch:
            batch.put_file(selection_plan,prefix+'/selections/'+selection+'/plan.json')
            batch.put_file(WHEEL,prefix+'/runtime.whl')
            for name in ['tokenizer_pool.py','tokenizer_speed.py','modal_generate_fast_tokenizers.py']:
                batch.put_file(ROOT/'scripts'/name,prefix+'/source/'+name)
        call=generate_selection.spawn(configs,runtime)
        save_json(pointer,{'call_id':call.object_id,'selection':selection,'runtime':runtime,'indices':selected})
    while True:
        try:rows=call.get(timeout=10);break
        except (TimeoutError,modal.exception.TimeoutError):pass
        completed=0
        for config in configs:
            try:
                list(outputs.iterdir(f'generated/{runtime["id"]}/{digest(config)}/complete.json'))
                completed+=1
            except (FileNotFoundError,modal.exception.NotFoundError):pass
        record={'at_utc':datetime.now(timezone.utc).isoformat(),'stage':'generating',
                'completed':completed,'total':len(configs),'call_id':call.object_id}
        save_json(STATE/'status.json',record)
        with (STATE/'events.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    save_json(STATE/f'{selection}-results.json',rows)
    record={'at_utc':datetime.now(timezone.utc).isoformat(),'stage':'complete','completed':len(rows),'total':len(configs)}
    save_json(STATE/'status.json',record)
    with (STATE/'events.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
    print(json.dumps({'selection':selection,'profile':'gildmember','artifacts':[r['volume_path'] for r in rows]}),flush=True)
