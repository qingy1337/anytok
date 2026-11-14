"""One complete native-128k GPU-only warmup and one measured update on Slurm."""
from datetime import datetime,timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from pool_generalization import save_json

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'modal_results/tillicum_smoke'


def main():
    STATE.mkdir(parents=True,exist_ok=True)
    metadata=dict(job_id=os.environ.get('SLURM_JOB_ID'),nodes=os.environ.get('SLURM_JOB_NODELIST'),
                  account=os.environ.get('SLURM_JOB_ACCOUNT'),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  python=sys.version,packages={k:importlib.metadata.version(k) for k in
                  ['torch','transformers','tokenizers','numpy','safetensors','accelerate']})
    save_json(STATE/'environment.json',metadata)
    folder=STATE/'gpu-only-128k'
    command=[sys.executable,'-u','scripts/local_memory_trial.py','--variant','chunked_norm_mlp',
             '--chunk-size','16384','--direct-embedding-backward','--tokenizer','989',
             '--output',str(folder),'--warmup','--disable-memory-history']
    save_json(STATE/'job-status.json',dict(stage='Running complete 128k GPU-only warmup and measured step',completed=0,total=1,
        platform='Tillicum · H200',workers=[dict(id=metadata['job_id'],profile='zlab',platform='Tillicum · H200',state='Full native 128k vocabulary; batch128; microbatch32; no CPU offload')]))
    started=time.monotonic()
    with (STATE/'trial.log').open('w') as stream:
        try:
            process=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=240)
            code=process.returncode
        except subprocess.TimeoutExpired:code=124
    path=folder/'result.json'
    if path.exists():result=json.loads(path.read_text())
    else:
        log=(STATE/'trial.log').read_text();warm=folder/'warmup/phases.json'
        phases=json.loads(warm.read_text()) if warm.exists() else []
        inputs=folder/'warmup/inputs.json'
        result=dict(outcome='out_of_memory' if 'out of memory' in log.lower() else 'error',
                    inputs=json.loads(inputs.read_text()) if inputs.exists() else {},phases=phases,
                    phase=phases[-1]['phase'] if phases else 'initialization',
                    peak_allocated_bytes=max((p['peak_allocated_bytes'] for p in phases),default=0),
                    error=log[-7000:])
    fits=result['outcome']=='complete'
    if fits:
        inputs=result['inputs']
        if (inputs['vocabulary']!=128000 or inputs['batch']!=128 or inputs['microbatch']!=32
            or inputs['offloaded_optimizer_bytes'] or inputs['offload_lm_gradients'] or inputs['activation_offload_gib']):
            raise RuntimeError('The requested full GPU-only workload changed')
        if not result['warmup']['numeric_and_input_identity_equal']:raise RuntimeError('Warmup/reload equivalence failed')
    summary=dict(at_utc=datetime.now(timezone.utc).isoformat(),fits=fits,outcome=result['outcome'],
                 subprocess_returncode=code,wall_seconds=time.monotonic()-started,environment=metadata,
                 gpu=result.get('inputs',{}).get('gpu'),gpu_total_bytes=result.get('inputs',{}).get('gpu_total_bytes'),
                 peak_allocated_bytes=result.get('peak_allocated_bytes'),
                 peak_reserved_bytes=max((p['peak_reserved_bytes'] for p in result.get('phases',[])),default=0),
                 model_step_seconds=sum(p['seconds'] for p in result.get('phases',[]) if not p['phase'].startswith('load ')),
                 full_result=result)
    save_json(STATE/'result-summary.json',summary)
    finished=result['outcome'] in ['complete','out_of_memory']
    save_json(STATE/'job-status.json',dict(stage='complete' if finished else 'failed',completed=1 if finished else 0,total=1,
        platform='Tillicum · H200',workers=[],fits=fits,outcome=result['outcome'],job_id=metadata['job_id']))
    print(json.dumps({k:v for k,v in summary.items() if k!='full_result'}),flush=True)
    if not finished:raise RuntimeError('Smoke test failed before a valid fit result')


if __name__=='__main__':main()
