"""Watch direct staging, submit one bounded Slurm smoke job, collect metadata."""
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

from pool_generalization import save_json,file_sha

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'modal_results/tillicum_smoke'
REPORT=ROOT/'reports/tillicum-128k-smoke'
REMOTE='/gpfs/scrubbed/qxli2/anytok-h200-smoke'
HOST='qxli2@tillicum.hyak.uw.edu'
SOCKET='/tmp/tmp.Mu0s3iNDKy/master'


def ssh(command):
    return subprocess.run(['ssh','-S',SOCKET,'-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,command],
                          capture_output=True,text=True,timeout=45)


def remote_text(relative):
    code=f"from pathlib import Path;p=Path({str(REMOTE+'/'+relative)!r});print(p.read_text() if p.exists() else '')"
    result=ssh('python3 -c '+shlex.quote(code))
    if result.returncode:raise RuntimeError('Remote metadata read failed: '+result.stderr[-1000:])
    return result.stdout.strip()


def remote_json(relative):
    text=remote_text(relative)
    return json.loads(text) if text else None


def status(stage,complete=False,job_id=None,**fields):
    value=dict(at_utc=datetime.now(timezone.utc).isoformat(),stage=stage,completed=1 if complete else 0,total=1,
        platform='Tillicum · H200',workers=[] if complete or stage=='failed' else [dict(id=str(job_id or 'artifact-staging'),
        profile='zlab',platform='Tillicum · H200',state=stage)],**fields)
    save_json(STATE/'status.json',value)
    with (STATE/'events.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
    print(json.dumps(value),flush=True)


def report(summary,receipt,accounting):
    REPORT.mkdir(parents=True,exist_ok=True)
    save_json(REPORT/'result.json',summary);save_json(REPORT/'artifact-receipt.json',receipt)
    (REPORT/'slurm-accounting.txt').write_text(accounting)
    allocated=summary.get('peak_allocated_bytes',0)/2**30
    reserved=summary.get('peak_reserved_bytes',0)/2**30
    capacity=(summary.get('gpu_total_bytes') or 0)/2**30
    result=summary['full_result'];inputs=result.get('inputs',{})
    lines=['# Tillicum 128k GPU-only smoke test','',
        f"**{'Fits' if summary['fits'] else 'Does not complete'} on {summary.get('gpu','the allocated GPU')}.** Outcome: `{summary['outcome']}`. Slurm job `{summary['environment']['job_id']}`; account `zlab`.",'',
        f'Peak PyTorch allocated GPU memory: **{allocated:.2f} GiB**. Peak reserved: **{reserved:.2f} GiB**. Device-reported capacity: **{capacity:.2f} GiB**.','',
        f"Timed model/optimizer phases: **{summary.get('model_step_seconds',0):.2f}s** ({'complete measured update' if summary['fits'] else 'partial attempt; not a complete-step runtime'}).",'',
        '## Exact workload','',
        'Same step 2000 trained LM/hypernetwork, populated Adam state, original corpora and complete 1,000-tokenizer pool as the local diagnostics. Native tokenizer #989 has 128,000 entries; batch 128, microbatch 32, maximum 128 LM positions, unchanged paired boundaries, canonical output head and lexical loss.','',
        'Execution variant: `chunked_norm_mlp`, the GPU-only source/normalization/recomputation/MLP-buffer optimization. No optimizer-state, LM-gradient or activation CPU offload. No tokenizer truncation or replacement inputs.','',
        'One full warmup update, then complete checkpoint/corpus/RNG reload for one measured update. Allocation stack tracing disabled, peak counters enabled. This is a very short capacity smoke, not a long-run throughput or training migration.','',
        f"Warmup/reload equality check: `{result.get('warmup',{}).get('numeric_and_input_identity_equal','not reached')}`. Scored continuation tokens: `{inputs.get('scored_tokens','not reached')}`; bytes: `{inputs.get('scored_bytes','not reached')}`.",'',
        '## Transfer and runtime','',
        f"All {receipt['verified_files']} files ({receipt['verified_bytes']:,} bytes) were downloaded directly from Modal to Tillicum and checksum-verified before GPU submission. Only signed metadata and small code passed through the local caller. Cloud originals remain authoritative.",'',
        'An isolated Python 3.12 environment uses PyTorch 2.15.0+cu132, transformers 5.15.1, tokenizers 0.22.2 and the same model code. The existing Rust helper passed its import preflight. Source hashes, actual GPU identity/capacity, package versions, all measured phases and error details are in result.json.','',
        'The single debug job requested one full H200 with a five-minute wall limit and no explicit CPU/system-memory reservation. Slurm elapsed time was 77 seconds, including startup and checkpoint loading. The scheduler assigned its defaults of 4 CPUs and 120G RAM. Preparation and transfer occurred before requesting GPU time. Existing Modal training was not modified.','',
        'Cross-device numerical parity against the local Blackwell GPU is not asserted; the smoke checks capacity and same-device warmup/reload fidelity. Prior local losslessness comparisons remain in [the implementation report](../chunked-memory/README.md).','']
    text='\n'.join(lines)
    (REPORT/'README.md').write_text(text)
    save_json(REPORT/'manifest.json',dict(files={str(p.relative_to(REPORT)):dict(bytes=p.stat().st_size,sha256=file_sha(p)) for p in REPORT.rglob('*') if p.is_file() and p.name!='manifest.json'}))


def main():
    STATE.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    while True:
        transfer=remote_json('modal_results/tillicum_smoke/transfer-status.json')
        if transfer and transfer['stage']=='complete':break
        if transfer:
            status(f"Direct cloud staging: {transfer['completed_bytes']/1e9:.1f}/{transfer['total_bytes']/1e9:.1f}GB; {transfer['stage']}")
        else:status('Waiting for direct artifact transfer')
        if time.monotonic()-started>1800:raise TimeoutError('Direct staging exceeded thirty minutes')
        time.sleep(10)
    receipt=remote_json('modal_results/tillicum_smoke/artifact-receipt.json');save_json(STATE/'artifact-receipt.json',receipt)
    if receipt['verified_files']!=1070 or receipt['tokenizers']!=1000:raise ValueError('Incomplete exact source transfer')
    submission=STATE/'submission.json'
    if submission.exists():job=json.loads(submission.read_text())
    else:
        status('Submitting one five-minute H200 debug smoke')
        command='cd '+shlex.quote(REMOTE)+' && sbatch --parsable scripts/tillicum_128k_smoke.sbatch'
        result=ssh(command)
        ids=re.findall(r'^([0-9]+)(?:;[^\n]+)?$',result.stdout,re.M)
        if result.returncode or len(ids)!=1:
            save_json(STATE/'submission-error.json',dict(stdout=result.stdout,stderr=result.stderr,returncode=result.returncode))
            raise RuntimeError('Slurm rejected the bounded smoke submission: '+result.stderr[-1600:])
        job=dict(job_id=ids[0],stdout=result.stdout,stderr=result.stderr,remote_root=REMOTE)
        save_json(submission,job)
    job_id=job['job_id'];last=None
    while True:
        summary=remote_json('modal_results/tillicum_smoke/result-summary.json')
        queue=ssh('squeue -h -j '+job_id+' -o '+shlex.quote('%T|%M|%R')).stdout.strip()
        for remote,local in [('modal_results/tillicum_smoke/trial.log','slurm.log'),('modal_results/tillicum_smoke/job-status.json','remote-status.json')]:
            raw=remote_text(remote)
            if raw:(STATE/local).write_text(raw+'\n')
        if summary and not queue:
            accounting=ssh('sacct -n -P -j '+job_id+' --format=JobIDRaw,State,Elapsed,ExitCode,AllocTRES').stdout
            save_json(STATE/'result-summary.json',summary)
            report(summary,receipt,accounting)
            if summary['outcome'] not in ['complete','out_of_memory']:
                raise RuntimeError('Smoke job failed without a valid memory-fit result: '+summary['outcome'])
            status('complete',True,job_id,fits=summary['fits'],outcome=summary['outcome'],
                   peak_gpu_gib=summary.get('peak_allocated_bytes',0)/2**30)
            return
        if not queue and not summary:
            accounting=ssh('sacct -n -P -j '+job_id+' --format=JobIDRaw,State,ExitCode').stdout
            if any(word in accounting for word in ['FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY']):
                (STATE/'failed-accounting.txt').write_text(accounting)
                log=remote_text(f'modal_results/tillicum_smoke/slurm-{job_id}.log');(STATE/'slurm.log').write_text(log)
                raise RuntimeError('Slurm job ended before a fit result: '+accounting.strip())
        stage=f'Slurm {job_id}: {queue or "collecting result"}'
        if stage!=last:status(stage,False,job_id);last=stage
        time.sleep(5)


if __name__=='__main__':
    try:main()
    except Exception as error:
        STATE.mkdir(parents=True,exist_ok=True);status('failed',error=str(error));raise
