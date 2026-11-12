"""Finish the prepared pool sweep after its queued B300 pilot completes.

Run in tmux, using the existing dashboard for observation. This is a finite
experiment launcher, not a service. It never switches GPUs or trains anything.
"""
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from modal_credits import billing_row
from pool_generalization import save_json

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'modal_results/rl_pool_20k'
PROFILE = 'blackpantherx86'


def event(stage, completed=0, **fields):
    row = {'at_utc': datetime.now(timezone.utc).isoformat(), 'stage': stage,
           'completed': completed, 'total': 1000, 'profile': PROFILE, **fields}
    save_json(STATE / 'status.json', row)
    with (STATE / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def wait_for_pilot():
    last_event = 0.
    while not (STATE / 'pilot.json').is_file():
        status = json.loads((STATE / 'status.json').read_text())
        if status.get('state') == 'failed':
            raise RuntimeError('Pilot failed: ' + status.get('error', 'see pilot.log'))
        running = subprocess.run(['tmux', 'has-session', '-t', 'anytok-rl-pool-pilot'],
                                 capture_output=True).returncode == 0
        if not running:
            # Allow an atomic result write immediately before the session exits.
            if (STATE / 'pilot.json').is_file():
                break
            raise RuntimeError('Pilot launcher exited without a saved pilot result; see pilot.log')
        if time.monotonic() - last_event > 60:
            event('waiting for B300 pilot to finish; full sweep will follow automatically')
            last_event = time.monotonic()
        time.sleep(15)
    return json.loads((STATE / 'pilot.json').read_text())


def main():
    pilot = wait_for_pilot()
    if len(pilot['rows']) != 3:
        raise ValueError('Incomplete pilot')
    if pilot.get('gpu') != 'B300':
        raise ValueError('A B300 pilot is required before the B300 sweep')
    # The first HN invocation includes CUDA kernel/library initialization.
    # Charge it once per worker rather than once for each of 1,000 tokenizers.
    # The other pilot vocabularies have115k/123k entries, versus an80k pool mean.
    # Use the slower warmed measurement plus40% headroom and fixed startup costs.
    seconds_per_tokenizer = max(r['elapsed_seconds'] for r in pilot['rows'][1:])
    first_invocation_seconds = pilot['rows'][0]['elapsed_seconds']
    estimate = seconds_per_tokenizer * 1020 * .001972 * 1.4 + 1.5
    balance = billing_row(PROFILE, 'this month', Decimal(30))
    save_json(STATE / 'launch-budget.json', {'balance': balance,
        'estimated_remaining_run_usd': estimate,
        'pilot_seconds_per_tokenizer': seconds_per_tokenizer,
        'pilot_first_invocation_seconds': first_invocation_seconds,
        'fixed_startup_and_host_allowance_usd': 1.5,
        'b300_usd_per_second': .001972, 'pricing_source': 'https://modal.com/pricing'})
    if balance['status'] != 'ok':
        raise RuntimeError('Could not verify credits before the full sweep')
    if Decimal(balance['estimated_remaining_usd']) < Decimal(str(estimate + 1)):
        event('pilot finished; full sweep needs a larger verified credit balance', 3,
              state='budget_stop', estimated_run_usd=estimate)
        return
    event('launching full B300 pool sweep', 3, estimated_run_usd=estimate,
          estimated_seconds=seconds_per_tokenizer * 1000 / 4)
    env = {**os.environ, 'MODAL_PROFILE': PROFILE}
    with (STATE / 'run.log').open('a') as log:
        subprocess.run(['modal', 'run', '--detach', 'scripts/modal_eval_rl_pool.py',
                        '--action', 'run'], cwd=ROOT, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    with (STATE / 'analysis.log').open('w') as log:
        subprocess.run([sys.executable, 'scripts/analyze_rl_pool.py'], cwd=ROOT,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    report = ROOT / 'reports/rl-pool-sweep-20k'
    summary = json.loads((report / 'analysis.json').read_text())
    save_json(STATE / 'summary.json', summary)
    save_json(STATE / 'credits-after.json', billing_row(PROFILE, 'this month', Decimal(30)))
    # Report artifacts are score metadata and figures; raw text/weights stay
    # cloud-only. Preserve the completed human-readable report cloud-side too.
    import modal
    volume = modal.Volume.from_name('anytok-rl-all-pool-20k', version=2)
    with volume.batch_upload(force=True) as batch:
        batch.put_directory(report, 'report')
        batch.put_file(STATE / 'preparation-audit.json', 'preparation-audit.json')
    event('complete', 1000, result=summary, report_directory=str(report),
          report_cloud_path='blackpantherx86/anytok-rl-all-pool-20k/report')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        previous = json.loads((STATE / 'status.json').read_text())
        event('failed', previous.get('completed', 0), state='failed', error=str(error))
        raise
