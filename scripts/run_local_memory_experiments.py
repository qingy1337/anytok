"""Run and report the user-authorized local memory experiment matrix."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

from pool_generalization import save_json, file_sha

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT/'modal_results/local_memory'
REPORT = ROOT/'reports/local-memory'
PYTHON = ROOT/'.venv-memory/bin/python'
CASES = [('baseline-43k','baseline',652,True), ('repeat-43k','baseline',652,False),
         ('offload-43k','offload',652,False), ('fine-43k','fine',652,False),
         ('fine-offload-43k','fine_offload',652,False),
         ('baseline-128k','baseline',989,False), ('offload-128k','offload',989,False),
         ('fine-128k','fine',989,False), ('fine-offload-128k','fine_offload',989,False)]


def status(stage, completed, **fields):
    row=dict(at_utc=datetime.now(timezone.utc).isoformat(),stage=stage,completed=completed,total=len(CASES),
        platform='Local · RTX PRO 6000 96 GB',workers=[] if stage in ['complete','failed'] else
        [dict(id='memory-gpu',profile='local',platform='Local · RTX PRO 6000 96 GB',state=stage)],**fields)
    save_json(STATE/'status.json',row)
    with (STATE/'events.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def export(rows):
    REPORT.mkdir(parents=True,exist_ok=True)
    save_json(REPORT/'results.json',rows)
    from plot_local_memory import plot
    plot(rows, ROOT/'plots/local_memory');plot(rows,REPORT/'local_memory')
    lines=['# Local memory diagnostics', '',
        'These are isolated memory experiments, not tokenizer-quality evaluations. The running Modal trainer is unchanged. The user approved a local diagnostic copy; all original artifacts remain in cloud storage.', '',
        'The complete 1,000-tokenizer pool and original corpora are present and hash-verified. Each trial uses a complete native 43k or 128k tokenizer, the real model/hypernetwork/optimizer checkpoint, 128 paired examples, microbatch 32 and the 128-position cap. Full-vocabulary attention and all original loss terms remain enabled. Trials force a tokenizer only in the separate diagnostic branch, not in the real RL run.', '',
        'Every trial uses its own process and the same checkpoint/corpus/RNG state. Numerical checks compare complete generated embeddings and every gradient-bearing LM/hypernetwork parameter before clipping. The repeated baseline measures the original implementation’s own reproducibility. Local PyTorch matches Modal 2.15.0; the local GPU model differs from the Modal B300.', '',
        '| Trial | Outcome | Peak allocated GiB | Failure phase |', '|---|---|---:|---|']
    for row in rows:
        r=row['result'];lines.append(f"| {row['id']} | {r['outcome']} | {r.get('peak_allocated_bytes',0)/2**30:.2f} | {r.get('phase','')} |")
    lines += ['', '![Memory diagnostic plots](local_memory.png)', '', 'A failed 128k trial is retained as an out-of-memory result. Its recorded peak is a lower bound on the memory needed to complete the step. A successful 43k comparison does not establish 128k equivalence or H200 compatibility.', '',
        'Detailed phases, numeric comparisons, source hashes and input identities are in results.json. Large local checkpoint, corpus, tokenizer and gradient files are excluded from this public report.', '']
    (REPORT/'README.md').write_text('\n'.join(lines))
    save_json(REPORT/'manifest.json',dict(files={name:dict(bytes=(REPORT/name).stat().st_size,sha256=file_sha(REPORT/name))
        for name in ['README.md','results.json','local_memory.png','local_memory.pdf']}))


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
        with (folder/'run.log').open('w') as stream:
            process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
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
