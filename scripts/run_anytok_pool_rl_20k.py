"""Plan (default) or run 20k adversarial pool updates on bounded Modal profiles.

    python scripts/run_anytok_pool_rl_20k.py --run

Existing immutable inputs and tokenizer files transfer cloud-to-cloud. Only
metadata returns locally. Training starts fresh; recovery resumes a full model,
optimizer, policy, corpus and RNG checkpoint. No explicit CPU/memory requests.
"""
import argparse
from datetime import datetime, timezone
from decimal import Decimal
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import modal
from anytok_pool_rl_config import APP_NAME, FINAL_STEP, POLICY, PROFILES, VOLUME_NAME, STATE_DIR, RUN_PREFIX
from modal_credits import billing_row
from modal_tokenizer_pool_transfer import source_client
from pool_generalization import save_json

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/STATE_DIR
LAUNCHER='scripts/modal_anytok_pool_rl_20k.py'


def emit(stage, **fields):
    old=json.loads((STATE/'status.json').read_text()) if (STATE/'status.json').exists() else {}
    row={**old,'at_utc':datetime.now(timezone.utc).isoformat(),'stage':stage,**fields}
    save_json(STATE/'status.json',row)
    with (STATE/'events.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def volume_for(profile):
    client,_=source_client(profile)
    return modal.Volume.from_name(VOLUME_NAME,version=2,client=client)


def latest(profile):
    try: entries=list(volume_for(profile).iterdir(RUN_PREFIX,recursive=True))
    except (FileNotFoundError,modal.exception.NotFoundError):return 0
    paths={e.path for e in entries}
    return max([int(Path(e.path).parent.name[5:]) for e in entries
                if e.path.endswith('/state.pt') and Path(e.path).parent.name.startswith('step-')
                and e.path==RUN_PREFIX+'/'+Path(e.path).parent.name+'/state.pt'
                and e.path.removesuffix('state.pt')+'manifest.json' in paths] or [0])


def active_apps(profile):
    result=subprocess.run(['modal','app','list','--json'],env={**os.environ,'MODAL_PROFILE':profile},
                          capture_output=True,text=True,check=True,timeout=30)
    return [r['app_id'] for r in json.loads(result.stdout) if r.get('description')==APP_NAME
            and r.get('state')!='stopped' and int(r.get('tasks') or 0)>0]


def call(profile, action, extra=(), resume_call=None):
    log=STATE/f'{profile}-{action}.log'
    command=['modal','run','--detach',LAUNCHER,'--action',action,*map(str,extra)]
    if resume_call:command+=['--resume-call',resume_call]
    with log.open('a') as stream:
        process=subprocess.Popen(command,cwd=ROOT,env={**os.environ,'MODAL_PROFILE':profile,
                                  'PYTHONUNBUFFERED':'1'},stdout=stream,stderr=subprocess.STDOUT)
        last=0
        while process.poll() is None:
            if time.monotonic()-last>=15:
                last=time.monotonic()
                records=[]
                for line in log.read_text(errors='replace').splitlines()[-400:]:
                    try:row=json.loads(line[line.index('{'):])
                    except (ValueError,TypeError):continue
                    if row.get('stage')=='train':records.append(row)
                if records and action=='train':
                    row=records[-1];seconds=statistics.median(r['seconds'] for r in records[-100:])
                    emit('training',profile=profile,step=row['step'],seconds_per_step=seconds,
                         train_bpb=row['train_bpb'],policy_entropy=row['policy_entropy'],
                         policy_effective_tokenizers=row['policy_effective_tokenizers'],
                         policy_max_probability=row['policy_max_probability'],
                         eta_seconds=(FINAL_STEP-row['step'])*seconds if len(records)>=20 else None,
                         workers=[{'id':'B300','profile':profile,'platform':'Modal · B300',
                                   'task':f"RL step {row['step']:,} / 20,000",'state':'Training'}])
            time.sleep(3)
    if process.returncode:raise RuntimeError(f'{profile} {action} failed; inspect {log}')
    return json.loads((STATE/f'{profile}-{action}-result.json').read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if not args.run:
        print(json.dumps({'profiles':PROFILES,'policy':POLICY,'steps':FINAL_STEP,'gpu':'B300',
                          'initialization':'same original LM and identity-warmed hypernetwork as uniform',
                          'volume':VOLUME_NAME,'max_segment_seconds':10800,'budget_estimate_per_hour_usd':8.25},indent=2));return
    STATE.mkdir(parents=True,exist_ok=True)
    with (STATE/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run_locked()


def run_locked():
    saved=json.loads((STATE/'status.json').read_text()) if (STATE/'status.json').exists() else {}
    if saved.get('stage')=='complete':print('Already complete.');return
    previous=saved.get('resume_source');step=saved.get('resume_step',0)
    recovered_smoke=None
    if previous and previous not in PROFILES:raise ValueError('Saved source is not enabled')
    if saved.get('profile'):
        profile=saved['profile']
        live=active_apps(profile)
        if live:
            action = {'preparing_inputs':'prepare','smoke_training_and_resume':'smoke',
                      'transferring_restart':'transfer'}.get(saved.get('stage'),'train')
            pointer_path=STATE/f'{profile}-{action}-call.json'
            if not pointer_path.exists():raise RuntimeError(f'Reconcile active app before recovery: {live}')
            pointer=json.loads(pointer_path.read_text())
            result=call(profile,action,resume_call=pointer['call_id'])
            if action=='train': previous=profile;step=result['step']
            if action=='smoke' and result['step']==3 and result['policy_updates']==3:
                saved['smoke_verified']=True
            if action=='smoke': recovered_smoke=result
        else:
            current=latest(profile)
            if current>=step and current>0:previous=profile;step=current
    emit('preparing',state='running',step=step,final_step=FINAL_STEP,profiles=list(PROFILES),
         resume_source=previous,resume_step=step,workers=[],eta_seconds=None,error=None,
         baseline_profile='gildmember',baseline_volume='anytok-pool-20k',policy=POLICY,
         code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    used=list(saved.get('used_profiles',[]))
    if previous and previous not in used:used.append(previous)
    for profile in PROFILES:
        if step>=FINAL_STEP:break
        balance=billing_row(profile,'this month',Decimal(30))
        if balance['status']!='ok' or float(balance['estimated_remaining_usd'])<10:continue
        if active_apps(profile):raise RuntimeError('Another RL app is active; refusing duplicate training')
        emit('preparing_inputs',profile=profile,workers=[])
        try:call(profile,'prepare')
        except RuntimeError:
            text=(STATE/f'{profile}-prepare.log').read_text(errors='replace').lower()
            if any(s in text for s in ['credits exhausted','has no credits','insufficient credits','spending limit','workspace is disabled']):
                emit('profile_unavailable',profile=profile);continue
            raise
        if previous and profile!=previous:
            emit('transferring_restart',profile=profile,resume_source=previous,resume_step=step)
            call(profile,'transfer',['--source-profile',previous,'--step',step])
            if latest(profile)!=step:raise RuntimeError('Transferred restart step mismatch')
            # Destination hash verification completed before source compaction.
            call(previous,'compact',['--step',step])
        elif previous and latest(profile)!=step:raise RuntimeError('Resume source step changed')
        if not saved.get('smoke_verified') and not step:
            emit('smoke_training_and_resume',profile=profile)
            result_path=STATE/f'{profile}-smoke-result.json'
            stored=json.loads(result_path.read_text()) if result_path.exists() else None
            if stored and (stored.get('step') not in [2,3] or stored.get('policy_updates')!=stored['step']):
                raise RuntimeError('Invalid saved smoke result')
            first=recovered_smoke or stored or call(profile,'smoke',['--step',2])
            second=call(profile,'smoke',['--step',3])
            if first['step'] not in [2,3] or second['step']!=3 or second['policy_updates']!=3:
                raise RuntimeError('Training/resume smoke failed')
            saved['smoke_verified']=True
            emit('smoke_verified',smoke_verified=True,profile=profile)
        balance=billing_row(profile,'this month',Decimal(30))
        if balance['status']!='ok':raise RuntimeError('Billing unavailable before training')
        seconds=min(10800,int((float(balance['estimated_remaining_usd'])-2.)/8.25*3600))
        if seconds<900:continue
        emit('starting_training',profile=profile,step=step,max_seconds=seconds,
             resume_source=previous,resume_step=step,workers=[])
        result=call(profile,'train',['--max-seconds',seconds])
        if result['step']<=step:raise RuntimeError('Segment made no progress')
        previous=profile;step=result['step']
        if profile not in used:used.append(profile)
        emit('checkpoint_committed',profile=profile,step=step,resume_source=profile,resume_step=step,
             used_profiles=used,workers=[])
    if step<FINAL_STEP:
        emit('budget_stop',state='budget_stop',step=step,resume_source=previous,resume_step=step,workers=[])
        return
    emit('collecting_checkpoints',profile=previous,step=step,workers=[])
    # Collect only grid checkpoints physically owned by each segment's profile.
    # Transferred full restarts can appear twice; matching manifests are required.
    selected={}
    for profile in used:
        source=volume_for(profile)
        for entry in list(source.iterdir(RUN_PREFIX,recursive=True)):
            if not entry.path.endswith('/manifest.json'):continue
            name=Path(entry.path).parent.name
            if not name.startswith('step-'):continue
            if entry.path != RUN_PREFIX+'/'+name+'/manifest.json':continue
            number=int(name[5:])
            if number%500 or not 500<=number<=FINAL_STEP:continue
            row=json.loads(b''.join(source.read_file(entry.path)))
            existing=selected.get(number)
            if existing:
                if any(existing['manifest']['files'][n]!=row['files'][n] for n in ['lm.pt','hypernet/model.safetensors','policy.pt']):
                    raise RuntimeError('Conflicting grid checkpoint weights')
            selected[number]={'profile':profile,'manifest':row}
    if set(selected)!=set(range(500,FINAL_STEP+1,500)):raise RuntimeError('Incomplete final grid')
    for number,item in sorted(selected.items()):
        if item['profile']!=previous:
            call(previous,'transfer',['--source-profile',item['profile'],'--step',number,'--grid-checkpoint'])
    call(previous,'audit-final')
    emit('complete',state='complete',profile=previous,step=step,workers=[],eta_seconds=0,
         artifact_profile=previous,artifact_volume=VOLUME_NAME,artifact_path=RUN_PREFIX,grid_collected=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        if STATE.exists():emit('failed',state='failed',error=str(error),workers=[])
        raise
