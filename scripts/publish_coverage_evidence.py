"""Persist diagnostic metadata, figures and source; never move model/corpus data."""
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import subprocess

import modal

from pool_generalization import digest, file_sha, save_json

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'modal_results/coverage_evidence'


def main():
    if os.environ.get('MODAL_PROFILE')!='gildmember':raise ValueError('Use gildmember')
    spec=json.loads((STATE/'spec.json').read_text())
    report=json.loads((STATE/'analysis.json').read_text())
    if len(report['screen']['cases'])!=432 or report['screen']['maximum_uniform_reproduction_error']>1e-5:
        raise ValueError('Incomplete or unreproduced screen')
    if not report['reward']['rng_replay_verified'] or not report['reward']['uniform_rng_replay_verified']:
        raise ValueError('Unverified domain replay')
    if len(report['component_swaps'])!=12 or report['medical_token_audit']['checked_training_tokenizers']!=1000:
        raise ValueError('Incomplete mechanism audits')
    paths=[STATE/name for name in ['spec.json','analysis.json','paired-screen.csv','feature-audit.json',
        'medical-token-audit.json','component-swaps.json','nonlinear-reward.json','rng-audit.json','uniform-rng-audit.json']]
    paths.extend(ROOT/name for name in ['plans/coverage-evidence-v1.json','plans/rl-ood-20k.md',
        'scripts/modal_coverage_evidence.py','scripts/analyze_coverage_evidence.py',
        'scripts/publish_coverage_evidence.py','plotting_scripts/plot_coverage_evidence.py',
        'tests/test_coverage_evidence.py'])
    paths.extend(ROOT/f'plots/coverage_evidence_{name}.{ext}' for name in
                 ['overview','heatmap','policy','mechanism'] for ext in ['png','pdf'])
    files={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':file_sha(p)} for p in paths}
    prefix=spec['prefix']+'/analysis/'+digest(files)[:16]
    manifest={'at_utc':datetime.now(timezone.utc).isoformat(),'prefix':prefix,'files':files,
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'profile':'gildmember','volume':'anytok-rl-ood-20k'}
    save_json(STATE/'analysis-manifest.json',manifest)
    volume=modal.Volume.from_name(manifest['volume'],version=2)
    with volume.batch_upload(force=True) as batch:
        for p in paths:batch.put_file(p,prefix+'/'+str(p.relative_to(ROOT)))
        batch.put_file(STATE/'analysis-manifest.json',prefix+'/manifest.json')
    # Read back all uploaded metadata and figure bytes before publishing the pointer.
    import hashlib
    for name,item in files.items():
        raw=b''.join(volume.read_file(prefix+'/'+name))
        if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:
            raise ValueError('Published artifact differs: '+name)
    pointer={'prefix':prefix,'manifest_sha256':file_sha(STATE/'analysis-manifest.json'),
             'verified_files':len(files),'at_utc':datetime.now(timezone.utc).isoformat()}
    with volume.batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO(json.dumps(pointer).encode()),spec['prefix']+'/analysis/latest.json')
    save_json(STATE/'published.json',pointer)
    print(json.dumps(pointer,indent=2))


if __name__=='__main__':main()
