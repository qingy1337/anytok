"""Finish full cloud scoring, then run only the frozen-model policy on local CPU."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from pool_generalization import digest, save_json

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'modal_results/frozen_policy_cache'
PROFILE = 'aurora'


def command(arguments, logfile, profile=None):
    env = os.environ.copy();env['OPENBLAS_NUM_THREADS'] = '1';env['OMP_NUM_THREADS'] = '1'
    if profile:
        env['MODAL_PROFILE'] = profile;env['FROZEN_POLICY_PROFILE'] = profile
    with (STATE / logfile).open('a') as stream:
        subprocess.run(arguments, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    # The already-started source preparation is allowed to finish; no duplicate job.
    deadline = time.monotonic() + 1800
    while not (STATE / 'preparation.json').exists():
        status = json.loads((STATE / 'status.json').read_text()) if (STATE / 'status.json').exists() else {}
        if status.get('stage') == 'failed' or time.monotonic() > deadline:
            raise RuntimeError('Full corrected source preparation did not complete')
        time.sleep(5)
    prepared = json.loads((STATE / 'preparation.json').read_text());signature = digest(prepared['plan'])
    marker = STATE / 'staged.json'
    if not marker.exists() or json.loads(marker.read_text()).get('plan_sha256') != signature:
        command([sys.executable, '-m', 'modal', 'run', '--detach', 'scripts/modal_frozen_policy_cache.py', '--action', 'stage'], 'stage.log', PROFILE)
        save_json(marker, dict(plan_sha256=signature, profile=PROFILE))
    if not all((STATE / f'scores-{k}.json.gz').exists() for k in range(2)):
        action = 'observe' if (STATE / 'calls.json').exists() else 'score'
        command([sys.executable, '-m', 'modal', 'run', '--detach', 'scripts/modal_frozen_policy_cache.py', '--action', action], 'observe.log' if action == 'observe' else 'score.log', PROFILE)
    command([sys.executable, 'scripts/frozen_policy_search.py'], 'policy.log')
    command([sys.executable, 'scripts/export_frozen_policy_search.py'], 'export.log')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        save_json(STATE / 'status.json', dict(at_utc=datetime.now(timezone.utc).isoformat(), stage='failed',
            completed=0, total=1000, profile=PROFILE, error=str(exc)))
        raise
