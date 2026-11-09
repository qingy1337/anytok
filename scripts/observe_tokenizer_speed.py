"""Observe one existing benchmark and refresh its audited report and plot.

    MODAL_PROFILE=gildmember python scripts/observe_tokenizer_speed.py \
      --pointer modal_results/tokenizer_speed_v8/RUN-benchmark-call.json --watch

This launches no Modal app or compute. It polls the exact existing call, reads
only completion metadata, and never retries/restarts jobs. On completion it
archives the report and plots beside the cloud-resident tokenizer artifacts.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import modal
from pool_generalization import save_json

ROOT = Path(__file__).resolve().parents[1]


def archive(state, run_id, volume):
    files = [state/f'{run_id}-{suffix}' for suffix in ['results.json','analysis.json','analysis.csv','findings.md']]
    files += [ROOT/'plots'/f'tokenizer_speed.{suffix}' for suffix in ['png','pdf']]
    report = json.loads((state/f'{run_id}-analysis.json').read_text())
    for execution in report.get('execution_extensions',{}).values():
        script=state/f'{execution["id"]}-rescue-source.py'
        script.write_bytes(b''.join(volume.read_file(execution['id']+'/rescue-source.py')))
        if hashlib.sha256(script.read_bytes()).hexdigest()!=execution['rescue_launcher_sha256']:
            raise ValueError('Rescue execution source changed')
        if script not in files:files.append(script)
    for name,expected_hash in report['reference_manifest_sha256'].items():
        original = ROOT/name
        if hashlib.sha256(original.read_bytes()).hexdigest() != expected_hash:
            raise ValueError('Original text reference changed after analysis')
        files.append(original)
    verification = state/f'{run_id}-verified-all.json'
    current = json.loads(verification.read_text())
    measured = json.loads((state/f'{run_id}-results.json').read_text())
    audited = {(r['case'],r['variant']):r for r in current['models']}
    if len(audited)!=18:raise ValueError('All current artifacts must be reloaded')
    for row in measured:
        actual = audited[row['case'],row['variant']]
        if (actual['tokenizer_sha256']!=row['manifest']['tokenizer_sha256']
                or actual['artifact_dir']!=row['artifact_dir']):
            raise ValueError('Artifact verification predates a timing replay')
    files.append(verification)
    manifest = {p.name: {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                         'bytes': p.stat().st_size} for p in files}
    receipt = state/f'{run_id}-report-manifest.json'
    save_json(receipt, manifest)
    prefix = f'{run_id}/reports/complete'
    with volume.batch_upload(force=True) as batch:
        for path in files: batch.put_file(path, f'{prefix}/{path.name}')
        batch.put_file(receipt, f'{prefix}/manifest.json')
    actual = {Path(e.path).name: e.size for e in volume.iterdir(prefix)}
    if not all(actual.get(name) == item['bytes'] for name,item in manifest.items()):
        raise ValueError('Cloud report artifact sizes differ')
    # Read back the small manifest, not corpora or tokenizer files.
    if json.loads(b''.join(volume.read_file(prefix+'/manifest.json'))) != manifest:
        raise ValueError('Cloud report manifest differs')
    return prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pointer', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=int, default=60)
    args = parser.parse_args()
    if os.environ.get('MODAL_PROFILE') != 'gildmember':
        raise ValueError('Explicit gildmember profile required')
    if args.interval < 15: raise ValueError('Observation interval must be at least 15 seconds')
    saved = json.loads(args.pointer.read_text())
    if saved['action'] != 'benchmark': raise ValueError('Only full benchmarks may be observed')
    provenance = saved['provenance']; run_id = provenance['id']; candidate = provenance['candidate']
    state = args.pointer.resolve().parent
    call = modal.FunctionCall.from_id(saved['call_id'])
    volume = modal.Volume.from_name('anytok-fast-tokenizers', version=2)
    def run(*parts):
        subprocess.run([sys.executable, *map(str, parts)], check=True, cwd=ROOT)
    def event(**fields):
        row = {'at_utc':datetime.now(timezone.utc).isoformat(),'call_id':saved['call_id'],**fields}
        save_json(state/'observer-status.json',row)
        with (state/'observer-events.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    while True:
        done = False
        try:
            rows = call.get(timeout=1)
            if any(row['provenance'] != provenance for row in rows):
                raise ValueError('Coordinator returned different provenance')
            save_json(state/f'{run_id}-results.json',rows)
            done = True
        except (TimeoutError, modal.exception.TimeoutError):
            # An observation timeout means the same call is still pending.
            pass
        except Exception as error:
            event(stage='call observation failed; no restart attempted',error=str(error))
            raise
        if not done:
            run(ROOT/'scripts/collect_tokenizer_speed.py','--candidate',candidate,'--run-id',run_id)
        command = [ROOT/'scripts/analyze_tokenizer_speed.py','--candidate',candidate,'--run-id',run_id]
        if not done: command.append('--partial')
        run(*command)
        report_path = state/f'{run_id}-analysis.json'
        run(ROOT/'plotting_scripts/plot_tokenizer_speed.py','--analysis',report_path)
        report = json.loads(report_path.read_text())
        if done:
            event(stage='timings complete: reloading current artifacts',completed=report['completed_phases'])
            log=state/'verify.log'
            with log.open('w') as stream:
                subprocess.run(['modal','run','--detach',str(ROOT/'scripts/modal_verify_tokenizer_speed.py'),
                                '--run-id',run_id,'--action','verify'],check=True,cwd=ROOT,
                               stdout=stream,stderr=subprocess.STDOUT)
            cloud_prefix = archive(state,run_id,volume)
            event(stage='complete: report archived',completed=report['completed_phases'],
                  qualified_at_least_5x=report['qualified_at_least_5x'],
                  ready_speedup=report['total']['ready_speedup'],cloud_prefix=cloud_prefix)
            return
        event(stage='same call pending; provisional plot refreshed',
              completed=report['completed_phases'],total=report['expected_phases'])
        if not args.watch: return
        time.sleep(args.interval)


if __name__ == '__main__': main()
