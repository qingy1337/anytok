"""Read local run artifacts and derive progress without modifying run state."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import heapq
import json
from pathlib import Path
import re
import statistics

DISABLED = {"general", "research-everettcc"}
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def clean(text):
    return ''.join(c for c in ANSI.sub('', text) if c.isprintable() or c in '\n\t')


def stamp(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace('Z', '+00:00')).timestamp()
    except ValueError:
        return None


def duration(seconds):
    if seconds is None:
        return '—'
    seconds = max(0, int(seconds))
    if seconds >= 86400:
        return f'{seconds // 86400}d {seconds % 86400 // 3600}h'
    if seconds >= 3600:
        return f'{seconds // 3600}h {seconds % 3600 // 60}m'
    if seconds >= 60:
        return f'{seconds // 60}m {seconds % 60}s'
    return f'{seconds}s'


def tail(path, limit=24000):
    try:
        with path.open('rb') as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - limit))
            data = stream.read().decode('utf-8', errors='replace')
        if size > limit:
            data = data.partition('\n')[2]
        return clean(data)
    except OSError:
        return ''


def event(line):
    try:
        start = line.index('{')
        value = json.loads(line[start:])
        return value if isinstance(value, dict) else None
    except (ValueError, TypeError):
        return None


class Reader:
    """Cache manifests and consume logs incrementally, tolerating atomic writes."""
    def __init__(self, root):
        self.root = Path(root)
        self.cache = {}
        self.logs = {}

    def json(self, path, default=None):
        path = Path(path)
        try:
            stat = path.stat()
            key = (stat.st_mtime_ns, stat.st_size)
            cached = self.cache.get(path)
            if cached and cached[0] == key:
                return cached[1]
            value = json.loads(path.read_text())
            self.cache[path] = (key, value)
            return value
        except (OSError, ValueError):
            # A partial write may be transient. Never replace valid counts with zero.
            return self.cache.get(path, (None, default))[1]

    def attempts(self, path, profile):
        path = Path(path)
        try:
            stat = path.stat()
        except OSError:
            return []
        state = self.logs.get(path)
        if not state or stat.st_size < state['offset'] or state['inode'] != stat.st_ino:
            state = {'offset': 0, 'inode': stat.st_ino, 'clock': None, 'jobs': {}}
            self.logs[path] = state
        with path.open('rb') as stream:
            stream.seek(state['offset'])
            while raw := stream.readline():
                if not raw.endswith(b'\n'):
                    break
                state['offset'] = stream.tell()
                item = event(raw.decode('utf-8', errors='replace'))
                if not item:
                    continue
                state['clock'] = stamp(item.get('at_utc')) or state['clock']
                stage, index = item.get('stage'), item.get('index')
                if stage == 'training' and index is not None:
                    state['jobs'][index] = {**item, 'profile': profile, 'log': str(path),
                        'started': state['clock'], 'state': 'training', 'inferred': True}
                elif stage in {'complete', 'job_complete'} and index in state['jobs']:
                    state['jobs'][index]['state'] = 'complete'
                elif stage == 'failed':
                    for job in state['jobs'].values():
                        if job['state'] != 'complete':
                            job['state'] = 'failed'
        return list(state['jobs'].values())


def registry(root, path):
    runs = json.loads(Path(path).read_text())['runs']
    if not runs or len({r['id'] for r in runs}) != len(runs):
        raise ValueError('Dashboard run IDs must be unique and nonempty')
    for run in runs:
        if run['adapter'] not in {'tokenizer_pool', 'progress_json', 'markers'}:
            raise ValueError(f"Unknown adapter: {run['adapter']}")
        if set(run.get('profiles', [])) & DISABLED:
            raise ValueError('Disabled profiles cannot be monitored')
    return runs


def runtime_estimate(config, manifests, elapsed=0):
    values = [m['train_seconds'] for m in manifests.values()
              if (m['config']['algorithm'], m['config']['pretokenizer']) ==
                 (config['algorithm'], config['pretokenizer'])]
    if len(values) < 3:
        return None, f'Calibrating ({len(values)} finished)'
    median = statistics.median(values)
    if elapsed > max(values):
        return None, 'Beyond observed range'
    return max(1, median - elapsed), f'≈{duration(max(1, median - elapsed))} · {len(values)} samples'


def pool_eta(configs, manifests, assignments, workers, concurrency):
    pending = {c['index']: c for c in configs if c['index'] not in manifests}
    if not pending:
        return 'Complete'
    unknown = {(c['algorithm'], c['pretokenizer']) for c in pending.values()
               if runtime_estimate(c, manifests)[0] is None}
    if unknown:
        return f'Calibrating · {len(unknown)} families need ≥3 finishes'
    if any(i not in assignments for i in pending):
        return 'Waiting for remaining shard assignments'
    active = {w.get('index'): w for w in workers if w.get('index') is not None}
    tasks = defaultdict(list)
    for index, config in pending.items():
        elapsed = active.get(index, {}).get('elapsed', 0)
        estimate, reason = runtime_estimate(config, manifests, elapsed)
        if estimate is None:
            return reason
        tasks[assignments[index]].append(estimate)
    end = 0
    for durations in tasks.values():
        slots = [0.0] * min(concurrency, len(durations))
        for value in sorted(durations, reverse=True):
            heapq.heapreplace(slots, slots[0] + value)
        end = max(end, max(slots))
    return f'≈{duration(end)} · assumes {concurrency} slots/profile'


def snapshot(reader, run, remote=None, now=None):
    now = now or datetime.now(timezone.utc).timestamp()
    root, remote = reader.root, remote or {}
    if run['adapter'] != 'tokenizer_pool':
        return other_snapshot(reader, run, remote, now)
    directory = root / run['directory']
    plan = reader.json(root / run['plan'], {})
    manifests, owners, per_profile, gathered = {}, {}, defaultdict(set), set()
    warnings, assignments = [], {}
    profiles = set(run.get('profiles', []))
    for path in [directory / 'regex-shards.json', *sorted(directory.glob('farm-round-*.json'))]:
        mapping = reader.json(path, {}) or {}
        for profile, indices in mapping.get('assignments', {}).items():
            if profile in DISABLED:
                warnings.append(f'Disabled profile ignored: {profile}')
                continue
            profiles.add(profile)
            assignments.update({i: profile for i in indices})
    folders = [directory, *sorted(p for p in directory.iterdir() if p.is_dir())] if directory.exists() else []
    for folder in folders:
        profile = 'group-cron' if folder == directory else folder.name
        if profile in DISABLED:
            continue
        paths = sorted(folder.glob('[0-9][0-9][0-9][0-9].json'))
        if paths:
            profiles.add(profile)
        for path in paths:
            item = reader.json(path)
            if not item or item.get('status') != 'complete':
                continue
            index = item.get('config', {}).get('index')
            if not isinstance(index, int) or not 0 <= index < len(plan.get('configs', [])):
                continue
            if item['config'] != plan['configs'][index]:
                warnings.append(f'Configuration mismatch at {path.name}')
                continue
            if index in manifests and item.get('tokenizer_sha256') != manifests[index].get('tokenizer_sha256'):
                warnings.append(f'Conflicting checksum for #{index}')
            manifests[index] = item
            if folder == directory:
                gathered.add(index)
            else:
                owners[index] = profile
                per_profile[profile].add(index)
    for index in manifests:
        owner = owners.get(index, assignments.get(index, 'group-cron'))
        per_profile[owner].add(index)
    latest = {}
    log_paths = []
    for folder in folders:
        profile = 'group-cron' if folder == directory else folder.name
        if profile not in profiles or profile in DISABLED:
            continue
        for path in sorted(folder.glob('*.log')):
            log_paths.append(path)
            for attempt in reader.attempts(path, profile):
                index = attempt['index']
                if index not in latest or (attempt['started'] or 0) > (latest[index]['started'] or 0):
                    latest[index] = attempt
    workers = []
    for index, attempt in latest.items():
        if index not in manifests and attempt['state'] == 'training':
            workers.append({**attempt, 'id': f'local-{index}', 'platform': run['platform'],
                            'elapsed': max(0, now - attempt['started']) if attempt['started'] else 0})
            assignments[index] = attempt['profile']
    live_workers = remote_workers({**run, '_profiles': sorted(profiles)}, remote, now)
    if live_workers is not None:
        # Fresh remote data is authoritative per successful profile query;
        # errors retain local inference and show a stale-data warning.
        successful = {p for p, value in remote.get('profiles', {}).items()
                      if value.get('ok') and now - value.get('checked_at', 0) < 120}
        workers = [w for w in workers if w['profile'] not in successful]
        for worker in live_workers:
            if worker.get('index') in manifests:
                continue
            fallback = latest.get(worker.get('index'), {})
            workers.append({**fallback, **worker})
    for worker in workers:
        index = worker.get('index')
        if isinstance(index, int) and 0 <= index < len(plan.get('configs', [])):
            config = plan['configs'][index]
            worker.update({key: config[key] for key in ('algorithm', 'pretokenizer', 'vocab_size')})
            worker['eta'] = runtime_estimate(config, manifests, worker.get('elapsed', 0))[1]
            assignments[index] = worker['profile']
        else:
            worker['eta'] = '—'
    farm = reader.json(directory / 'farm-status.json', {}) or {}
    stage = farm.get('stage', 'Waiting for controller')
    if stage in {'stopped', 'failed', 'budget_stop'}:
        warnings.append(farm.get('reason', 'Controller stopped; inspect logs'))
    total, complete = plan.get('count', 0), len(manifests)
    if complete == total and total:
        stage = 'complete'
    elif stage in {'bootstrap', 'bootstrap_progress'}:
        stage = 'Measuring slow tokenizer families'
    counts = Counter(m['config']['pretokenizer'] for m in manifests.values())
    families = Counter(c['pretokenizer'] for c in plan.get('configs', []))
    breakdown = ' · '.join(f'{name} {counts[name]}/{count}' for name, count in families.items())
    started = min((a['started'] for a in latest.values() if a['started']), default=None)
    profile_rows = []
    for profile in sorted(profiles - DISABLED):
        assigned = {i for i, p in assignments.items() if p == profile}
        produced = per_profile[profile]
        target = len(assigned | produced)
        active = sum(w['profile'] == profile for w in workers)
        info = remote.get('profiles', {}).get(profile, {})
        profile_rows.append({'name': profile, 'platform': run['platform'], 'done': len(produced),
            'total': target, 'active': active, 'state': 'Running' if active else ('Finished shard' if produced else 'Waiting'),
            'remaining': info.get('remaining'), 'checked_at': info.get('checked_at'),
            'error': info.get('error'), 'inferred': not info.get('ok') or now - info.get('checked_at', 0) > 120})
        if info.get('billing_error'):
            warnings.append(f'{profile}: billing unavailable; showing cached balance')
        if info.get('log_error'):
            warnings.append(f'{profile}: worker logs unavailable; retrying next refresh')
        if info.get('error'):
            warnings.append(f"{profile}: live query failed; showing last/local state")
    return {'id': run['id'], 'name': run['name'], 'platform': run['platform'], 'completed': complete,
        'total': total, 'unit': 'tokenizers', 'collected': len(gathered), 'stage': stage,
        'elapsed': (stamp(farm.get('at_utc')) or now) - started if started and farm.get('stage') == 'complete' else (now - started if started else None),
        'eta': pool_eta(plan.get('configs', []), manifests, assignments, workers, run.get('concurrency', 100)),
        'profiles': profile_rows, 'workers': sorted(workers, key=lambda w: (w['profile'], w['id'])),
        'breakdown': breakdown, 'warnings': warnings, 'logs': log_paths,
        'output_bytes': sum(m.get('tokenizer_bytes', 0) for m in manifests.values()), 'at': now}


def remote_workers(run, remote, now):
    workers = []
    for profile in run.get('_profiles', run.get('profiles', [])):
        info = remote.get('profiles', {}).get(profile, {})
        if not info.get('ok') or now - info.get('checked_at', 0) >= 120:
            continue
        apps = {a['app_id']: a for a in info.get('apps', [])
                if a.get('description') in run.get('app_names', []) and a.get('state') != 'stopped'}
        for container in info.get('containers', []):
            app = apps.get(container.get('app_id'))
            if not app:
                continue
            container_id = container['container_id']
            details = info.get('jobs', {}).get(container_id, {})
            started = stamp(container.get('start_time'))
            workers.append({**details, 'id': container_id, 'profile': profile,
                'platform': run['platform'], 'elapsed': max(0, now - started) if started else 0,
                'started': started, 'inferred': False,
                'state': 'Transferring' if app.get('description', '').endswith('transfer') else details.get('state', 'Starting / active'),
                'remote_logs': info.get('logs', {}).get(container_id, [])})
    return workers


def dotted(data, field, default=None):
    for key in field.split('.'):
        if not isinstance(data, dict) or key not in data:
            return default
        data = data[key]
    return data


def other_snapshot(reader, run, remote, now):
    counts = Counter()
    if run['adapter'] == 'markers':
        identities = set()
        for path in reader.root.glob(run['markers']):
            value = reader.json(path, {}) or {}
            identity = dotted(value, run.get('identity_field', 'id'))
            if identity is not None and identity not in identities:
                identities.add(identity)
                for profile in run.get('profiles', []):
                    if profile in path.parts:
                        counts[profile] += 1
        complete, data = len(identities), {}
    else:
        data = reader.json(reader.root / run['status'], {}) or {}
        complete = dotted(data, run.get('completed_field', 'completed'), 0)
    total = run.get('total', dotted(data, run.get('total_field', 'total'), 0))
    complete = complete if isinstance(complete, (int, float)) else 0
    stage = 'complete' if total and complete >= total else data.get('stage', 'Waiting for progress')
    if run.get('completion_stage') and data.get('stage')!=run['completion_stage']:
        stage=data.get('stage','Waiting for progress')
    terminal = data.get('state') in {'stopped', 'failed', 'superseded', 'budget_stop'}
    profile_state = data['state'].replace('_', ' ').capitalize() if terminal else ('Complete' if stage == 'complete' else 'Active / waiting')
    workers = remote_workers(run, remote, now)
    for item in dotted(data, run.get('workers_field', 'workers'), []) or []:
        if not isinstance(item, dict):
            continue
        started = stamp(item.get('started_at'))
        workers.append({**item, 'id': str(item.get('id', 'local')), 'profile': item.get('profile', 'local'),
            'platform': item.get('platform', run['platform']),
            'elapsed': item.get('elapsed_seconds', now - started if started else 0),
            'eta': f"≈{duration(item['eta_seconds'])}" if isinstance(item.get('eta_seconds'), (int, float)) else '—',
            'state': item.get('state', item.get('task', 'Active'))})
    profiles = [{'name': p, 'platform': run['platform'], 'done': counts[p] if run['adapter'] == 'markers' else None,
                 'total': run.get('profile_targets', {}).get(p, 0), 'active': sum(w['profile'] == p for w in workers),
                 'remaining': remote.get('profiles', {}).get(p, {}).get('remaining'),
                 'state': profile_state} for p in sorted(set(run.get('profiles', [])) | {w['profile'] for w in workers})]
    eta = profile_state if terminal else ('Complete' if stage == 'complete' else 'Awaiting measurements')
    explicit_eta = dotted(data, run.get('eta_seconds_field', 'eta_seconds'))
    if not terminal and stage != 'complete' and isinstance(explicit_eta, (int, float)):
        eta = f'≈{duration(explicit_eta)} · reported by run'
    return {'id': run['id'], 'name': run['name'], 'platform': run['platform'], 'completed': complete,
        'total': total, 'unit': run.get('unit', 'items'), 'collected': None, 'stage': stage,
        'elapsed': dotted(data, run.get('elapsed_field', 'elapsed_seconds')), 'eta': eta,
        'profiles': profiles, 'workers': workers, 'breakdown': '', 'warnings': [],
        'logs': [reader.root / p for p in run.get('logs', [])], 'output_bytes': 0, 'at': now}
