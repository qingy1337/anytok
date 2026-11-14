"""Matched native-vocabulary timing trials; no tokenizer truncation or training changes."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

from pool_generalization import save_json, file_sha

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'modal_results/vocab_scaling'
REPORT=ROOT/'reports/vocab-scaling'
PYTHON=ROOT/'.venv-memory/bin/python'
POINTS=[(32000,494),(34000,728),(36000,14),(37000,138),(40000,550),(43000,817)]
METHODS={
    'original':['--variant','baseline'],
    'gpu_new':['--variant','chunked_norm_mlp','--chunk-size','16384','--direct-embedding-backward'],
    'staged_new':['--variant','chunked_norm_staged','--chunk-size','16384','--direct-embedding-backward',
                  '--offload-optimizer','--offload-lm-gradients','--release-generated-output'],
}


def cases():
    result=[];order=list(METHODS);rng=random.Random(43128)
    for repeat in range(3):
        points=POINTS.copy();rng.shuffle(points)
        for position,(vocabulary,index) in enumerate(points):
            rotated=order[(repeat+position)%3:]+order[:(repeat+position)%3]
            for method in rotated:
                result.append(dict(id=f'{vocabulary}-{method}-r{repeat}',vocabulary=vocabulary,
                                   tokenizer_index=index,method=method,repeat=repeat,fit=True))
    for repeat in range(3):
        result.append(dict(id=f'128000-staged_new-r{repeat}',vocabulary=128000,tokenizer_index=989,
                           method='staged_new',repeat=repeat,fit=False))
    return result


def status(stage,done,total,durations,**extra):
    eta=(sum(durations[-9:])/len(durations[-9:])*(total-done)) if durations else None
    row=dict(at_utc=datetime.now(timezone.utc).isoformat(),stage=stage,completed=done,total=total,
             eta_seconds=eta,platform='Local · RTX PRO 6000 96 GB',
             workers=[] if stage in ['complete','failed'] else [dict(id='scaling-gpu',profile='local',
                 platform='Local · RTX PRO 6000 96 GB',state=stage)],**extra)
    save_json(STATE/'status.json',row)
    with (STATE/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def export(rows):
    save_json(REPORT/'results.json',rows)
    if len(rows)>=3:
        from plot_vocab_scaling import render
        render(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--resume',action='store_true');a=p.parse_args()
    STATE.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    archive=REPORT/'source_versions';archive.mkdir(exist_ok=True)
    for name in ['local_memory_trial','local_memory_variants','chunked_hypernetwork','lossless_layer_norm','lossless_mlp','run_vocab_scaling']:
        source=ROOT/'scripts'/f'{name}.py';shutil.copyfile(source,archive/(file_sha(source)+'.py'))
    matrix=cases();save_json(REPORT/'design.json',dict(cases=matrix,points=POINTS,repeats=3,
        checkpoint_step=2000,batch=128,microbatch=32,maximum_positions=128,
        full_step_warmup=True,checkpoint_reloaded_after_warmup=True,memory_stack_tracing=False,
        note='Fit only complete native 32k–43k BPE/no-regex tokenizers. 128k is an out-of-fit measurement of the exact staged recipe.'))
    rows=[];durations=[]
    for case in matrix:
        folder=STATE/case['id'];folder.mkdir(exist_ok=True);result_path=folder/'result.json'
        if a.resume and result_path.exists():
            result=json.loads(result_path.read_text())
            if result['outcome']=='complete':
                rows.append(dict(**case,result=result));continue
        status('warmup + exact reload + timing: '+case['id'],len(rows),len(matrix),durations)
        command=[str(PYTHON),'-u','scripts/local_memory_trial.py',*METHODS[case['method']],
                 '--tokenizer',str(case['tokenizer_index']),'--output',str(folder),
                 '--warmup','--disable-memory-history']
        begin=time.monotonic()
        with (folder/'run.log').open('w') as f:
            result=subprocess.run(command,cwd=ROOT,env={**os.environ,'PYTORCH_ALLOC_CONF':'expandable_segments:True'},
                                  stdout=f,stderr=subprocess.STDOUT,timeout=900)
        if result.returncode:raise RuntimeError(str(folder/'run.log'))
        result=json.loads(result_path.read_text())
        if result['outcome']!='complete':raise RuntimeError('Incomplete benchmark: '+str(result_path))
        if result['inputs']['vocabulary']!=case['vocabulary']:raise ValueError('Native vocabulary differs from requested point')
        if not result['warmup']['numeric_and_input_identity_equal']:raise ValueError('Warmup/reload mismatch')
        rows.append(dict(**case,result=result));durations.append(time.monotonic()-begin)
        # All methods at a given point use the same checkpoint, RNG and exact examples.
        peers=[r for r in rows if r['vocabulary']==case['vocabulary']]
        reference=peers[0]['result']
        for other in peers[1:]:
            for key in ['context_sha256','scored_tokens','scored_bytes','checkpoint_step']:
                if other['result']['inputs'][key]!=reference['inputs'][key]:raise ValueError('Paired workload changed: '+key)
            for key in ['raw_bpb','canonical_bpb','gap_bpb','lexical_loss','cuda_rng_sha256']:
                if other['result']['numeric'][key]!=reference['numeric'][key]:raise ValueError('Method changed the scored computation: '+key)
        export(rows)
        status('completed '+case['id'],len(rows),len(matrix),durations)
    export(rows);status('complete',len(rows),len(matrix),durations)


if __name__=='__main__':
    try:main()
    except Exception as error:
        STATE.mkdir(parents=True,exist_ok=True)
        prior=json.loads((STATE/'status.json').read_text()) if (STATE/'status.json').exists() else {}
        status('failed',prior.get('completed',0),prior.get('total',57),[],error=str(error));raise
