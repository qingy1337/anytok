"""Incremental, read-only learning curves and policy metadata for the web UI."""
from __future__ import annotations

import math
from pathlib import Path

from monitoring.state import Reader, event


def sampled(rows, limit=400):
    """Keep endpoints and evenly spaced observations; never smooth scores."""
    if len(rows) <= limit:
        return rows
    return [rows[round(i * (len(rows) - 1) / (limit - 1))] for i in range(limit)]


class LearningReader:
    def __init__(self, root):
        self.root = Path(root)
        self.reader = Reader(root)
        self.files = {}

    def records(self, pattern):
        merged = {}
        for path in sorted(self.root.glob(pattern)):
            stat = path.stat()
            cache = self.files.get(path)
            if not cache or cache['inode'] != stat.st_ino or stat.st_size < cache['offset']:
                cache = {'inode': stat.st_ino, 'offset': 0, 'rows': {}}
                self.files[path] = cache
            with path.open('rb') as stream:
                stream.seek(cache['offset'])
                while raw := stream.readline():
                    if not raw.endswith(b'\n'):
                        break
                    cache['offset'] = stream.tell()
                    row = event(raw.decode('utf-8', errors='replace'))
                    if row and row.get('stage') in ('train', 'eval', 'initial_eval'):
                        step = row.get('step', 0)
                        if isinstance(step, int) and step >= 0:
                            cache['rows'][('eval' if row['stage'] == 'initial_eval' else row['stage'], step)] = row
            merged.update(cache['rows'])
        return merged

    def read(self, config):
        current = self.records(config['logs'])
        baseline = self.records(config['baseline_logs'])
        train = sorted((r for (stage, _), r in current.items() if stage == 'train'), key=lambda r: r['step'])
        latest = train[-1]['step'] if train else max((step for _, step in current), default=0)
        warnings = []
        reference = current.get(('eval', 0), {}).get('results', [])
        curves = {}
        for name, records in [('rl', current), ('uniform', baseline)]:
            points = []
            for (stage, step), row in sorted(records.items()):
                if stage != 'eval' or step > latest:
                    continue
                results = row.get('results', [])
                if len(reference) != 2 or len(results) != 2 or any(
                    any(r.get(k) != expected.get(k) for k in ('bytes', 'examples', 'truncated_forms'))
                    for r, expected in zip(results, reference)
                ):
                    warnings.append(f'{name} step {step}: sanity panel counts differ; point omitted')
                    continue
                if not all(isinstance(r.get(k), (int, float)) and math.isfinite(r[k])
                           for r in results for k in ('generated_bpb', 'canonical_bpb')):
                    warnings.append(f'{name} step {step}: invalid score omitted')
                    continue
                points.append({'step': step, 'tokenizers': [
                    {'bpb': r['generated_bpb'], 'canonical': r['canonical_bpb'],
                     'gap': r['generated_bpb'] - r['canonical_bpb']} for r in results]})
            curves[name] = points
        # The common initial scores provide a stronger cross-run identity check
        # than counts alone. Document/spec identity was verified at run launch.
        if reference and config.get('baseline_required', True) and baseline.get(('eval', 0), {}).get('results') != reference:
            warnings.append('Uniform initialization differs; baseline curves hidden')
            curves['uniform'] = []
        keys = ('policy_effective_tokenizers', 'policy_max_probability', 'policy_unique_tokenizers', 'policy_entropy')
        policy = [{'step': r['step'], **{k: r.get(k) for k in keys}} for r in train]
        snapshots = self.reader.json(self.root / config['snapshots'], {}) or {}
        plan = self.reader.json(self.root / config['plan'], {}) or {}
        configs = {c['index']: c for c in plan.get('configs', [])}
        recent = train[-500:]
        exposure = {}
        for field in ('algorithm', 'pretokenizer'):
            counts = {}
            for row in recent:
                value = configs.get(row.get('tokenizer_config_index'), {}).get(field)
                if value is not None:
                    counts[value] = counts.get(value, 0) + 1
            exposure[field] = {key: value / len(recent) for key, value in counts.items()} if recent else {}
        return {'latest_step': latest, 'curves': curves, 'protocol_note': config.get('protocol_note'), 'panel': [
                    {k: r[k] for k in ('bytes', 'examples', 'truncated_forms')} for r in reference],
                'policy': sampled(policy), 'latest_policy': policy[-1] if policy else None,
                'snapshots': snapshots.get('snapshots', []),
                'snapshot_checked_at': snapshots.get('checked_at'),
                'snapshot_error': snapshots.get('error'),
                'recent_exposure': exposure, 'recent_steps': len(recent),
                'pool_size': len(configs), 'warnings': sorted(set(warnings))}
