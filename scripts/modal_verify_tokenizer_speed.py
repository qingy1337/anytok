"""Reload benchmark artifacts in cloud storage, including preemption replays.

    MODAL_PROFILE=gildmember modal run scripts/modal_verify_tokenizer_speed.py --run-id RUN

Default is a read-only plan. Use --action verify after generation; --scope fast
checks the nine optimized artifacts, while the default all checks both variants.
No tokenizer files or corpora return to the caller. No CPU/memory reservations.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

import modal
from pool_generalization import file_sha, save_json

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'modal_results/tokenizer_speed_v8'
WHEEL=ROOT/'build/fast-v8/tokenizers-0.22.2-cp39-abi3-manylinux_2_34_x86_64.whl'
app=modal.App('anytok-tokenizer-speed-verify')
source=modal.Volume.from_name('anytok-pool-20k',version=2)
outputs=modal.Volume.from_name('anytok-fast-tokenizers',version=2)
image=(modal.Image.debian_slim(python_version='3.12')
       .add_local_file(WHEEL,'/tmp/fast.whl',copy=True)
       .run_commands("python -c \"import zipfile; zipfile.ZipFile('/tmp/fast.whl').extractall('/opt/fast')\"")
       .env({'PYTHONPATH':'/opt/fast','RAYON_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'true'})
       .add_local_python_source('tokenizer_pool','tokenizer_speed','pool_generalization'))


def now():return datetime.now(timezone.utc).isoformat()


@app.function(image=image,volumes={'/train':source.with_mount_options(read_only=True),'/out':outputs},
              max_containers=1,timeout=1800)
def verify(run_id,scope,wheel_sha):
    import tokenizer_speed
    from tokenizer_speed import lossless_audit
    from tokenizers import Tokenizer
    root=Path('/out')/run_id
    plan_path=root/'plan.json';plan=json.loads(plan_path.read_text())
    if file_sha('/tmp/fast.whl')!=wheel_sha:raise ValueError('Verification wheel changed')
    variants=['fast-200mb'] if scope=='fast' else ['baseline-200mb','fast-200mb']
    rows=[];matched={}
    for case in plan['cases']:
        for variant in variants:
            marker=root/case['key']/variant/'complete.json'
            saved=json.loads(marker.read_text())
            artifact=Path(saved['artifact_dir']).resolve()
            if not artifact.is_relative_to(root.resolve()):raise ValueError('Artifact path outside benchmark')
            path=artifact/'tokenizer.json';sha=file_sha(path)
            if (sha!=saved['manifest']['tokenizer_sha256'] or saved['manifest']['config']!=case['config']
                    or saved['provenance']['fast_wheel_sha256']!=wheel_sha
                    or saved['provenance']['plan_sha256']!=file_sha(plan_path)
                    or saved['provenance']['worker_sha256']!=file_sha(tokenizer_speed.__file__)):
                raise ValueError('Current artifact differs from benchmark provenance')
            if Tokenizer.from_file(str(path)).get_vocab_size()!=case['config']['vocab_size']:
                raise ValueError('Actual vocabulary differs from requested size')
            audit=lossless_audit(path,'/train/input/eval')
            if (audit['fixture_sha256']!=saved['lossless_audit']['fixture_sha256']
                    or audit['fixture_tokens']!=saved['lossless_audit']['fixture_tokens']):
                raise ValueError('Reloaded tokenization differs from recorded audit')
            rows.append({'case':case['key'],'variant':variant,'tokenizer_sha256':sha,
                         'artifact_dir':str(artifact),'audit':audit,'container_id':saved['container_id']})
            matched.setdefault(case['key'],[]).append(saved['container_id'])
            save_json(root/'verification/progress.json',{'completed':len(rows),'total':len(plan['cases'])*len(variants),'at_utc':now(),'scope':scope})
            outputs.commit()
    if scope=='all' and any(not all(ids) or len(set(ids))!=1 for ids in matched.values()):
        raise ValueError('Timing variants came from different containers')
    result={'run_id':run_id,'scope':scope,'at_utc':now(),'models':rows,'wheel_sha256':wheel_sha}
    save_json(root/f'verification/current-models-{scope}.json',result);outputs.commit()
    return result


@app.local_entrypoint()
def main(run_id: str,action: str='plan',scope: str='all'):
    if os.environ.get('MODAL_PROFILE')!='gildmember':raise ValueError('Explicit gildmember required')
    if not re.fullmatch('[a-f0-9]{16}',run_id):raise ValueError('Expected full benchmark ID')
    if action not in ['plan','verify'] or scope not in ['all','fast']:raise ValueError('Invalid action/scope')
    if action=='plan':
        print(json.dumps({'run_id':run_id,'scope':scope,'models':18 if scope=='all' else 9,
                          'read_only_models':True,'output_volume':'gildmember/anytok-fast-tokenizers'}));return
    STATE.mkdir(parents=True,exist_ok=True)
    def emit(record):
        row={'at_utc':now(),**record}
        save_json(STATE/'verification-status.json',row)
        with (STATE/'verification-events.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
    pointer=STATE/f'{run_id}-verify-{scope}-call.json';wheel_sha=file_sha(WHEEL)
    if pointer.exists():
        saved=json.loads(pointer.read_text())
        if saved['wheel_sha256']!=wheel_sha:raise ValueError('Saved verification used another wheel')
        call=modal.FunctionCall.from_id(saved['call_id'])
    else:
        call=verify.spawn(run_id,scope,wheel_sha)
        save_json(pointer,{'call_id':call.object_id,'wheel_sha256':wheel_sha})
    while True:
        try:result=call.get(timeout=10);break
        except (TimeoutError,modal.exception.TimeoutError):pass
        try:progress=json.loads(b''.join(outputs.read_file(f'{run_id}/verification/progress.json')))
        except (FileNotFoundError,modal.exception.NotFoundError):progress={}
        if progress.get('scope')!=scope:progress={'completed':0,'total':18 if scope=='all' else 9}
        emit({'stage':'reloading current artifacts',**progress})
    save_json(STATE/f'{run_id}-verified-{scope}.json',result)
    emit({'stage':'complete','completed':len(result['models']),'total':18 if scope=='all' else 9})
    print(json.dumps({'run_id':run_id,'scope':scope,'verified_models':len(result['models'])}),flush=True)
