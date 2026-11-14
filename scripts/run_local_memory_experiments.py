"""Run and report the user-authorized local memory experiment matrix."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from pool_generalization import save_json
from report_local_memory import export

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT/'modal_results/local_memory'
REPORT = ROOT/'reports/local-memory'
PYTHON = ROOT/'.venv-memory/bin/python'
CASES = [('baseline-43k','baseline',652,True), ('repeat-43k','baseline',652,False),
         ('offload-43k','offload',652,False), ('fine-43k','fine',652,False),
         ('fine-offload-43k','fine_offload',652,False),
         ('baseline-128k','baseline',989,False), ('offload-128k','offload',989,False),
         ('fine-128k','fine',989,False), ('fine-offload-128k','fine_offload',989,False),
         ('split-norm-43k','split_norm_offload',652,False),('split-norm-128k','split_norm_offload',989,False),
         ('fine-offload-43k-expand','fine_offload',652,False),('fine-offload-128k-expand','fine_offload',989,False),
         ('split-norm-43k-expand','split_norm_offload',652,False),('split-norm-128k-expand','split_norm_offload',989,False),
         ('split-disk-43k-expand','split_norm_disk',652,False),('split-disk-128k-expand','split_norm_disk',989,False),
         ('split-norm-fixed-43k-expand','split_norm_offload',652,False),('split-norm-fixed-128k-expand','split_norm_offload',989,False),
         ('split-disk-fixed-43k-expand','split_norm_disk',652,False),('split-disk-fixed-128k-expand','split_norm_disk',989,False),
         ('split-disk-fixed-43k-host80-expand','split_norm_disk',652,False),('split-disk-fixed-128k-host80-expand','split_norm_disk',989,False),
         ('optimizer-only-43k','baseline',652,False),
         ('split-disk-fixed-43k-host80-optimizer-expand','split_norm_disk',652,False),
         ('split-disk-fixed-128k-host80-optimizer-expand','split_norm_disk',989,False)]


def status(stage, completed, **fields):
    row=dict(at_utc=datetime.now(timezone.utc).isoformat(),stage=stage,completed=completed,total=len(CASES),
        platform='Local · RTX PRO 6000 96 GB',workers=[] if stage in ['complete','failed'] else
        [dict(id='memory-gpu',profile='local',platform='Local · RTX PRO 6000 96 GB',state=stage)],**fields)
    save_json(STATE/'status.json',row)
    with (STATE/'events.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    STATE.mkdir(parents=True,exist_ok=True)
    while not (STATE/'artifacts/download_receipt.json').is_file():
        result=subprocess.run(['tmux','has-session','-t','anytok-local-memory-download'],capture_output=True)
        if result.returncode:raise RuntimeError('Artifact download ended without a complete verified receipt')
        time.sleep(10)
    rows=[];reference=STATE/'reference-43k'
    for position,(name,variant,index,write) in enumerate(CASES):
        folder=STATE/'trials'/name;folder.mkdir(parents=True,exist_ok=True)
        existing=json.loads((folder/'result.json').read_text()) if args.resume and (folder/'result.json').exists() else None
        if existing and existing['outcome'] in ['complete','out_of_memory']:
            rows.append(dict(id=name,result=existing));continue
        status('running '+name,position)
        command=[str(PYTHON),'-u','scripts/local_memory_trial.py','--variant',variant,'--tokenizer',str(index),'--output',str(folder)]
        if index==652:
            command+=['--reference',str(reference)]
            if write:command+=['--write-reference']
        if 'host80' in name:command+=['--host-budget-gib','80']
        if 'optimizer' in name:command+=['--offload-optimizer']
        with (folder/'run.log').open('w') as stream:
            environment={**os.environ}
            if name.endswith('-expand'):environment['PYTORCH_ALLOC_CONF']='expandable_segments:True'
            process=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
            start=time.monotonic();last_phase=None
            while process.poll() is None:
                if time.monotonic()-start>1800:
                    process.terminate();process.wait(timeout=20);raise TimeoutError('Local trial exceeded its bound')
                phases=json.loads((folder/'phases.json').read_text()) if (folder/'phases.json').exists() else []
                phase=phases[-1]['phase'] if phases else 'loading exact checkpoint'
                if phase!=last_phase:
                    status(name+' · '+phase,position,trial_elapsed_seconds=time.monotonic()-start)
                    last_phase=phase
                time.sleep(3)
        if process.returncode:raise RuntimeError('Local trial failed: '+str(folder/'run.log'))
        r=json.loads((folder/'result.json').read_text());rows.append(dict(id=name,result=r));export(rows)
        status('completed '+name,position+1,outcome=r['outcome'],peak_gib=r.get('peak_allocated_bytes',0)/2**30)
    export(rows);status('complete',len(CASES))


if __name__=='__main__':
    try:main()
    except Exception as error:
        prior=json.loads((STATE/'status.json').read_text()) if (STATE/'status.json').exists() else {}
        status('failed',prior.get('completed',0),error=str(error));raise
