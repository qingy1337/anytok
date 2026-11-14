"""Collect only policy distribution metadata from an existing training container.

Never starts a Modal app/container, reads LM weights, or downloads checkpoints.
Run --follow in tmux alongside the run monitor; --once performs one read.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import pty
import select
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from monitoring.state import DISABLED, clean, event

from anytok_pool_rl_config import STATE_DIR, RUN_PREFIX

STATE = ROOT / STATE_DIR
OUTPUT = STATE / 'policy-monitor.json'


def save(data):
    temporary = OUTPUT.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, allow_nan=False) + '\n')
    temporary.replace(OUTPUT)


def remote_code(known):
    source = (ROOT / 'monitoring/policy_summary.py').read_text()
    return source + '''
import hashlib,json,torch
from pathlib import Path
torch.set_num_threads(1)
root=Path('/mnt/rl')
plan=json.loads((root/'input/pool/plan.json').read_text())
configs=plan['configs']
assert len(configs)==1000 and all(c['index']==i for i,c in enumerate(configs))
known=KNOWN_STEPS
rows=[]
for folder in sorted((root/'RUN_PREFIX_VALUE').glob('step-*')):
 step=int(folder.name[5:])
 if step==0 or step%500 or step in known or not (folder/'manifest.json').is_file():continue
 manifest=json.loads((folder/'manifest.json').read_text())
 path=folder/'policy.pt'
 expected=manifest['files']['policy.pt']
 assert manifest['step']==step and path.stat().st_size==expected['bytes']
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected['sha256']
 state=torch.load(path,map_location='cpu',weights_only=True)
 assert state['updates']==step and state['settings']['kind']=='shared_settings_mlp'
 row=summarize(configs,state['logits'].softmax(0).tolist(),state['counts'].tolist(),step)
 row.update(policy_sha256=expected['sha256'],plan_sha256=plan['plan_sha256'])
 rows.append(row)
print(json.dumps({'policy_monitor_snapshots':rows},allow_nan=False),flush=True)
'''.replace('KNOWN_STEPS', repr(sorted(known))).replace('RUN_PREFIX_VALUE', RUN_PREFIX)


def capture(command, env, timeout=50):
    """Modal container exec needs a PTY even when stdout is consumed as JSON."""
    master, slave = pty.openpty()
    process = None
    chunks = []
    try:
        process = subprocess.Popen(command, env=env, stdin=slave, stdout=slave, stderr=slave, start_new_session=True)
        os.close(slave)
        slave = None
        deadline = time.monotonic() + timeout
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError('Policy metadata read timed out; training is unaffected')
            ready, _, _ = select.select([master], [], [], .2)
            if ready:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                chunks.append(chunk)
            elif process.poll() is not None:
                break
        process.wait(timeout=2)
        text = clean(b''.join(chunks).decode(errors='replace'))
        if process.returncode:
            raise RuntimeError(text[-900:])
        return text
    finally:
        if process is not None and process.poll() is None:
            process.kill()  # local CLI reader only, never the training process
            process.wait()
        os.close(master)
        if slave is not None:
            os.close(slave)


def collect():
    try:
        saved = json.loads(OUTPUT.read_text())
    except (OSError, ValueError):
        saved = {'snapshots': []}
    try:
        status = json.loads((STATE / 'status.json').read_text())
        profile = status.get('profile')
        if not profile or profile in DISABLED:
            raise ValueError('No enabled active training profile')
        known = {row['step'] for row in saved['snapshots']}
        # No remote read unless a new regular checkpoint can exist.
        due = set(range(500, int(status.get('step', 0)) // 500 * 500 + 1, 500)) - known
        if not due:
            return saved
        pointer = json.loads((STATE / f'{profile}-train-call.json').read_text())
        env = {**os.environ, 'MODAL_PROFILE': profile}
        result = subprocess.run(['modal', 'container', 'list', '--json'], env=env,
                                capture_output=True, text=True, check=True, timeout=20)
        containers = [c for c in json.loads(result.stdout) if c.get('app_id') == pointer['app_id']]
        if len(containers) != 1:
            raise RuntimeError('Waiting for the active training container; previous snapshots retained')
        text = capture(['modal', 'container', 'exec', containers[0]['container_id'], '--',
                        'python', '-c', remote_code(known)], env)
        payload = next((row for line in text.splitlines()
                        if (row := event(line)) and 'policy_monitor_snapshots' in row), None)
        if payload is None:
            raise RuntimeError('No policy metadata received; previous snapshots retained')
        expected_plan = json.loads((ROOT/'plans/tokenizer-pool-1000-v1.json').read_text())['plan_sha256']
        merged = {r['step']: r for r in saved['snapshots']}
        for row in payload['policy_monitor_snapshots']:
            if row['plan_sha256'] != expected_plan:
                raise ValueError('Policy snapshot pool differs from registered pool')
            merged[row['step']] = {**row, 'profile': profile}
        saved.update(snapshots=[merged[k] for k in sorted(merged)], error=None)
    except (OSError, ValueError, KeyError, RuntimeError, TimeoutError, subprocess.SubprocessError) as exc:
        saved['error'] = str(exc)[-900:]
    saved['checked_at'] = datetime.now(timezone.utc).isoformat()
    save(saved)
    print(json.dumps({'at_utc': saved['checked_at'], 'stage': 'policy_metadata',
                      'snapshots': len(saved['snapshots']), 'error': saved.get('error')}), flush=True)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--follow', action='store_true')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if not (args.follow or args.once):
        parser.print_help()
        return
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE/'policy-monitor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            collect()
            if not args.follow:
                break
            time.sleep(30)


if __name__ == '__main__':
    main()
