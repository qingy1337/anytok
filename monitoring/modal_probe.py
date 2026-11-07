"""Read-only Modal polling. No run, deploy, stop or container-exec operations."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

from monitoring.state import DISABLED, clean, event

CID = re.compile(r'\bta-[A-Za-z0-9]+\b')


def query(profile, arguments, timeout=20):
    if profile in DISABLED:
        raise ValueError(f'Disabled profile: {profile}')
    if tuple(arguments[:2]) not in {('app', 'list'), ('app', 'logs'), ('container', 'list'), ('billing', 'summary')}:
        raise ValueError('Dashboard only permits read-only Modal metadata commands')
    result = subprocess.run(['modal', *arguments], env={**os.environ, 'MODAL_PROFILE': profile},
                            capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(clean(result.stderr or result.stdout)[-700:].strip())
    return result.stdout


def log_details(text):
    jobs, logs = {}, {}
    for raw in text.splitlines():
        line = clean(raw)
        match = CID.search(line)
        if not match:
            continue
        container = match.group()
        logs.setdefault(container, []).append(line)
        logs[container] = logs[container][-40:]
        value = event(line)
        if not value:
            continue
        if value.get('stage') == 'training':
            jobs[container] = {**value, 'state': 'Training'}
        elif value.get('stage') == 'corpus_consumed':
            jobs.setdefault(container, {})['state'] = 'Training · corpus loaded'
    return jobs, logs


class ModalProbe:
    def __init__(self, cache_path):
        self.path = Path(cache_path)
        try:
            self.cache = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.cache = {'profiles': {}}

    def profile(self, profile, names, previous):
        row = deepcopy(previous)
        try:
            apps = json.loads(query(profile, ['app', 'list', '--json']))
            containers = json.loads(query(profile, ['container', 'list', '--json']))
            row.update(ok=True, apps=apps, containers=containers, checked_at=time.time(), error=None)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            row.update(ok=False, error=str(exc), attempted_at=time.time())
            return profile, row
        if time.time() - row.get('billing_checked_at', 0) > 300:
            try:
                billing = json.loads(query(profile, ['billing', 'summary', '--json']))
                row['remaining'] = 30 + float(billing['adjustments']['credits'])
                row['billing_checked_at'] = time.time()
                row['billing_error'] = None
            except (OSError, ValueError, RuntimeError, KeyError, subprocess.TimeoutExpired) as exc:
                row['billing_error'] = str(exc)
        live_ids = {c['container_id'] for c in containers}
        jobs = {key: value for key, value in row.get('jobs', {}).items() if key in live_ids}
        logs = {key: value for key, value in row.get('logs', {}).items() if key in live_ids}
        row['log_error'] = None
        for app in apps:
            if app.get('description') not in names or app.get('state') == 'stopped' or not int(app.get('tasks') or 0):
                continue
            try:
                text = query(profile, ['app', 'logs', app['app_id'], '--tail', '1000', '--timestamps', '--show-container-id'])
                app_jobs, app_logs = log_details(text)
                jobs.update(app_jobs)
                logs.update(app_logs)
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                row['log_error'] = str(exc)
        row.update(jobs=jobs, logs=logs)
        return profile, row

    def poll(self, profiles, app_names):
        cache = deepcopy(self.cache)
        selected = sorted(set(profiles) - DISABLED)
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = executor.map(lambda p: self.profile(p, app_names, cache.get('profiles', {}).get(p, {})), selected)
            for name, value in results:
                cache.setdefault('profiles', {})[name] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile('w', dir=self.path.parent, delete=False) as stream:
            json.dump(cache, stream)
            temporary = Path(stream.name)
        temporary.replace(self.path)
        self.cache = cache
        return cache
