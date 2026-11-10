"""Validate, compare and plot RL20k against uniform on unchanged fixed cases."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from pool_generalization import digest, file_sha, save_json
from rl_ood_eval import compare

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'modal_results/rl_ood_20k'


def main():
    spec=json.loads((ROOT/'modal_results/uniform_trajectories/trajectory-spec.json').read_text())
    frozen=json.loads((STATE/'evaluation-spec.json').read_text())
    baseline=ROOT/'modal_results/uniform_trajectories/checkpoints/step-020000.json'
    rl=json.loads((STATE/'evaluate-result.json').read_text())
    if (file_sha(baseline)!=frozen['uniform_result_sha256'] or rl['evaluation_digest']!=digest(frozen)
            or spec['spec_sha256']!=frozen['frozen_spec_sha256']):raise ValueError('Evaluation provenance differs')
    report=compare(spec,json.loads(baseline.read_text()),rl)
    report.update(at_utc=datetime.now(timezone.utc).isoformat(),checkpoint_manifest=frozen['checkpoint_manifest'])
    save_json(STATE/'comparison.json',report)
    subprocess.run([sys.executable,str(ROOT/'plotting_scripts/plot_rl_ood.py')],check=True,cwd=ROOT)
    print(json.dumps([{ 'case':r['key'],'group':r['group'],
                       'gap':r['scores']['gap'],'bpb':r['scores']['transferred']} for r in report['cases']],indent=2))


if __name__=='__main__':main()
