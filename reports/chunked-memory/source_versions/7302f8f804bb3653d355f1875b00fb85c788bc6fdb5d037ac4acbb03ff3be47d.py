"""Isolated chunking trials with complete real native tokenizer vocabularies."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from pool_generalization import save_json, file_sha

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT/'modal_results/chunked_memory'
REPORT = ROOT/'reports/chunked-memory'


def status(stage, completed, total, **extra):
    value = dict(at_utc=datetime.now(timezone.utc).isoformat(), stage=stage,
                 completed=completed,total=total,platform='Local · RTX PRO 6000 96 GB',
                 workers=[] if stage in ['complete','failed'] else [dict(id='local-gpu',profile='local',platform='Local · RTX PRO 6000 96 GB',state=stage)],**extra)
    save_json(STATE/'status.json',value)
    with (STATE/'events.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
    print(json.dumps(value),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--chunk-size',type=int,default=2048)
    p.add_argument('--variant',default='chunked');p.add_argument('--label',default='chunked');p.add_argument('--tokenizers',type=int,nargs='+',default=[652,989]);p.add_argument('--offload-optimizer',action='store_true');p.add_argument('--release-embedding-leaf',action='store_true');p.add_argument('--direct-embedding-backward',action='store_true');p.add_argument('--activation-offload-gib',type=float,default=0.);a=p.parse_args()
    STATE.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i,index in enumerate(a.tokenizers):
        name=f'{a.label}-{index}-c{a.chunk_size}';folder=STATE/name;folder.mkdir(exist_ok=True)
        status('running '+name,i,len(a.tokenizers))
        command=[str(ROOT/'.venv-memory/bin/python'),'-u','scripts/local_memory_trial.py','--variant',a.variant,
                 '--chunk-size',str(a.chunk_size),'--tokenizer',str(index),'--output',str(folder)]
        if index==652:command+=['--reference',str(ROOT/'modal_results/local_memory/reference-43k')]
        if a.offload_optimizer:command+=['--offload-optimizer']
        if a.release_embedding_leaf:command+=['--release-embedding-leaf']
        if a.direct_embedding_backward:command+=['--direct-embedding-backward']
        if a.activation_offload_gib:command+=['--activation-offload-gib',str(a.activation_offload_gib)]
        with (folder/'run.log').open('w') as f:
            env={**os.environ,'PYTORCH_ALLOC_CONF':'expandable_segments:True'}
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            begin=time.monotonic();last=None
            while proc.poll() is None:
                phases=folder/'phases.json'
                if phases.exists():
                    records=json.loads(phases.read_text());phase=records[-1]['phase'] if records else 'setup'
                    if phase!=last:status(name+' · last complete: '+phase,i,len(a.tokenizers));last=phase
                if time.monotonic()-begin>1200:proc.terminate();proc.wait(timeout=20);raise TimeoutError(name)
                time.sleep(3)
        if proc.returncode:raise RuntimeError(str(folder/'run.log'))
        result=json.loads((folder/'result.json').read_text());rows.append(dict(id=name,result=result))
        save_json(REPORT/(name+'.json'),result)
        status('completed '+name,i+1,len(a.tokenizers),outcome=result['outcome'])
    status('complete',len(a.tokenizers),len(a.tokenizers))


if __name__=='__main__':
    try:main()
    except Exception as e:
        STATE.mkdir(parents=True,exist_ok=True);status('failed',0,1,error=str(e));raise
