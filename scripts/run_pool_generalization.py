"""Run the preregistered matched seen/fresh test without interrupting training."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

import modal
from pool_generalization import make_plan,save_json,digest,file_sha,load_plan,STEPS
from modal_tokenizer_pool_transfer import source_client

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'modal_results/pool_generalization'
PLAN=ROOT/'plans/pool-generalization-v1.json'
CPU='scripts/modal_pool_generalization.py'
GPU='scripts/modal_eval_pool_generalization.py'


def emit(stage,completed=0,**fields):
    value={'at_utc':datetime.now(timezone.utc).isoformat(),'stage':stage,'completed':completed,
           'total':21,**fields}
    save_json(STATE/'status.json',value)
    print(json.dumps(value),flush=True)


def design():
    pool=json.loads((ROOT/'plans/tokenizer-pool-1000-v1.json').read_text())
    manifests={i:json.loads((ROOT/f'modal_results/tokenizer_pool/{i:04d}.json').read_text()) for i in range(1000)}
    rows={}
    for path in (ROOT/'modal_results/anytok_pool_20k').glob('*-train.log'):
        for line in path.read_text(errors='replace').splitlines():
            try:r=json.loads(line[line.index('{'):])
            except (ValueError,TypeError):continue
            if r.get('stage')=='train' and r['step']<=12000:rows[r['step']]=r
    if set(rows)!=set(range(1,12001)):raise ValueError('Complete exposure history through 12k required')
    exposures={s:Counter(r['tokenizer_config_index'] for k,r in rows.items() if k<=s) for s in STEPS}
    folder=ROOT/'zett/output_joint_smollm2_32k_zett_retrain_15k/fineweb_edu_length_control/positions-0128'
    panel={n:file_sha(folder/n) for n in ['examples.jsonl.gz','corpus.jsonl.gz']}
    plan=make_plan(pool,manifests,exposures,panel)
    wheel=file_sha(ROOT/'build/pool-v2/tokenizers-0.22.2-cp39-abi3-manylinux_2_34_x86_64.whl')
    if any(manifests[p['seen']['index']]['provenance']['runtime_wheel_sha256']!=wheel for p in plan['pairs']):
        raise ValueError('Selected seen tokenizers used a different runtime wheel')
    plan['runtime_wheel_sha256']=wheel
    plan['cpu_cost_forecast_usd']=sum(manifests[p['seen']['index']]['estimated_process_cost_usd'] for p in plan['pairs'])*1.5+.1
    plan.pop('plan_sha256');plan['plan_sha256']=digest(plan)
    if PLAN.exists() and json.loads(PLAN.read_text())!=plan:raise ValueError('Existing experiment plan differs')
    save_json(PLAN,plan)
    return plan


def call(profile,script,args,label,stage):
    with (STATE/f'{label}.log').open('a') as stream:
        process=subprocess.Popen(['modal','run','--detach',script,*args],cwd=ROOT,
            env={**os.environ,'MODAL_PROFILE':profile},stdout=stream,stderr=subprocess.STDOUT)
        count=-1
        while process.poll() is None:
            fresh=len(list(STATE.glob('fresh-*.json')))
            if label=='train' and fresh!=count:
                count=fresh;emit(stage,fresh,profile=profile)
            time.sleep(5)
        if process.returncode:raise RuntimeError(f'{label} failed; inspect {STATE/f"{label}.log"}')


def main():
    STATE.mkdir(parents=True,exist_ok=True)
    with (STATE/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=design()
        emit('preparing disjoint tokenizer-training documents')
        call('group-cron',CPU,['--action','prepare'],'prepare','preparing corpus')
        profile=None
        for candidate in ['trenton','remote-worker']:
            try:
                client,_=source_client(candidate)
                with modal.App('anytok-generalization-capacity').run(client=client):pass
                profile=candidate;break
            except Exception as exc:print(f'{candidate} unavailable: {exc}',flush=True)
        if profile is None:raise RuntimeError('No verified evaluation workspace available')
        save_json(STATE/'workspace.json',{'profile':profile,'volume':'anytok-pool-generalization'})
        call(profile,CPU,['--action','import-corpus'],'import-corpus','installing corpus')
        emit('generating 18 fresh tokenizers',profile=profile)
        with ThreadPoolExecutor(max_workers=2) as executor:
            assets=executor.submit(call,profile,CPU,['--action','assets'],'assets','copying frozen checkpoints')
            call(profile,CPU,['--action','train'],'train','generating fresh tokenizers')
            assets.result()
        emit('evaluating 2k, 6k and 12k on H200',18,profile=profile)
        call(profile,GPU,[],'evaluate','evaluating')
        emit('complete',21,profile=profile,volume='anytok-pool-generalization',report='report.json')


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if STATE.exists():emit('failed',len(list(STATE.glob('fresh-*.json'))),error=str(exc))
        raise
