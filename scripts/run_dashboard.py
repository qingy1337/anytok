"""Serve the live experiment dashboard at http://0.0.0.0:1337.

    python scripts/run_dashboard.py

Reads local artifacts every two seconds, Modal metadata every 30 seconds,
and billing every five minutes. Never launches or stops experiment jobs.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monitoring.modal_probe import ModalProbe
from monitoring.learning import LearningReader
from monitoring.state import Reader, event, registry, snapshot, stamp, tail


def read_logs(data, run, root, profile=None, worker=None):
    """Bounded log tails from registered artifacts only; never client paths."""
    if worker:
        selected = next((w for w in data['workers'] if w['id'] == worker), None)
        if not selected:
            return 'Worker is no longer active.'
        if selected.get('remote_logs'):
            return '\n'.join(selected['remote_logs'][-80:])
        if selected.get('log'):
            return tail(Path(selected['log']))
        return 'Waiting for worker logs.'
    paths = data['logs']
    if profile:
        folder = root / run.get('directory', '')
        paths = [p for p in paths if (p.parent == folder if profile == 'group-cron'
                                     else p.parent.name == profile)]
    records, raw_tails = [], []
    for path in paths:
        content = tail(path)
        if content:
            raw_tails.append((path.name, content))
        for line in content.splitlines():
            item = event(line)
            when = stamp(item.get('at_utc')) if item else None
            # File mtime is NOT the timestamp of every old line in that file.
            if when is not None:
                records.append((when, f'{path.parent.name}/{path.name}  {line}'))
    if records:
        return '\n'.join(line for _, line in sorted(records)[-80:])
    return '\n\n'.join(f'{name}\n{content}' for name, content in raw_tails[-2:])[-24000:] or 'No logs available yet.'


class Monitor:
    def __init__(self, root=ROOT, config=None, offline=False):
        self.root = Path(root)
        self.runs = registry(self.root, config or self.root / 'plans/run-dashboard.json')
        self.reader = Reader(self.root)
        self.learning_reader = LearningReader(self.root)
        self.probe = ModalProbe(self.root / 'modal_results/run_dashboard/remote.json')
        self.offline = offline
        self.remote = self.probe.cache
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.data = {}
        self.error = self.poll_error = None
        self.polling = False
        self.plot_error = None
        self.threads = []
        self.refresh()

    def refresh(self):
        with self.lock:
            for run in self.runs:
                self.data[run['id']] = snapshot(self.reader, run, self.remote)
            self.error = None

    def state(self, run_id=None, profile=None, worker=None):
        with self.lock:
            run_id = run_id or self.runs[0]['id']
            run = next((r for r in self.runs if r['id'] == run_id), None)
            if run is None:
                raise KeyError(run_id)
            data = self.data[run_id]
            profiles = {p['name'] for p in data['profiles']}
            if profile and profile not in profiles:
                raise KeyError(profile)
            selected = {p: self.remote.get('profiles', {}).get(p, {}) for p in profiles}
            ages = [time.time() - r['checked_at'] for r in selected.values() if r.get('checked_at')]
            result = {**data, 'logs': read_logs(data, run, self.root, profile, worker),
                      'runs': [{'id': r['id'], 'name': r['name'],
                                'completed': self.data[r['id']]['completed'],
                                'total': self.data[r['id']]['total']} for r in self.runs],
                      'offline': self.offline, 'polling': self.polling,
                      'remote_age': max(ages) if ages else None,
                      'error': self.error or self.poll_error}
            plot = self.root / 'plots/anytok_pool_learning.png'
            result['plot_updated_at'] = plot.stat().st_mtime if plot.is_file() else None
            result['plot_error'] = self.plot_error
            rl_plot = self.root / 'plots/rl_ood_20k.png'
            result['rl_ood_plot_updated_at'] = rl_plot.stat().st_mtime if rl_plot.is_file() else None
            coverage_plot = self.root / 'plots/coverage_evidence_overview.png'
            result['coverage_plot_updated_at'] = coverage_plot.stat().st_mtime if coverage_plot.is_file() else None
            natural_plot = self.root / 'plots/natural_transfer.png'
            result['natural_plot_updated_at'] = natural_plot.stat().st_mtime if natural_plot.is_file() else None
            comparison_plot = self.root / 'plots/model_comparison.png'
            result['model_comparison_plot_updated_at'] = comparison_plot.stat().st_mtime if comparison_plot.is_file() else None
            reference_plot = self.root / 'plots/random_pool_50.png'
            result['pool_reference_plot_updated_at'] = reference_plot.stat().st_mtime if reference_plot.is_file() else None
            result['pool_reference'] = self.reader.json(self.root / 'reports/random-pool-50/analysis.json', {}) if run_id == 'pool-regime-comparison' else None
            corrected_plot = self.root / 'plots/corrected_pool_check.png'
            result['corrected_pool_plot_updated_at'] = corrected_plot.stat().st_mtime if corrected_plot.is_file() else None
            policy_search_plot = self.root / 'plots/frozen_policy_search.png'
            result['frozen_policy_plot_updated_at'] = policy_search_plot.stat().st_mtime if policy_search_plot.is_file() else None
            entropy_plot = self.root / 'plots/frozen_policy_entropy0.png'
            result['frozen_entropy_plot_updated_at'] = entropy_plot.stat().st_mtime if entropy_plot.is_file() else None
            result['workers'] = [{k: v for k, v in w.items() if k not in {'log', 'remote_logs'}}
                                 for w in data['workers']]
            return result

    def local_loop(self):
        while not self.stop.wait(2):
            try:
                self.refresh()
            except Exception as exc:
                with self.lock:
                    self.error = f'Artifact refresh failed; showing last valid state: {exc}'

    def learning(self, run_id):
        with self.lock:
            run = next((r for r in self.runs if r['id'] == run_id), None)
            if run is None or not run.get('learning'):
                raise KeyError(run_id)
            return self.learning_reader.read(run['learning'])

    def remote_loop(self):
        while not self.stop.is_set():
            with self.lock:
                active = [r for r in self.runs if self.data[r['id']]['stage'] != 'complete'
                          and r.get('app_names')]
                profiles = {p['name'] for r in active for p in self.data[r['id']]['profiles']}
                names = {n for r in active for n in r.get('app_names', [])}
                self.polling = bool(profiles)
            try:
                if profiles:
                    remote = self.probe.poll(profiles, names)
                    with self.lock:
                        self.remote = remote
                        self.poll_error = None
                    self.refresh()
            except Exception as exc:
                with self.lock:
                    self.poll_error = f'Modal refresh failed; showing last valid state: {exc}'
            finally:
                with self.lock:
                    self.polling = False
            self.stop.wait(30)

    def start(self):
        for target in [self.local_loop, self.plot_loop] + ([] if self.offline else [self.remote_loop]):
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            self.threads.append(thread)

    def plot_loop(self):
        script = self.root / 'plotting_scripts/plot_anytok_pool_learning.py'
        if not script.is_file():
            return
        previous = None
        while not self.stop.is_set():
            signature = [(str(p), p.stat().st_mtime_ns, p.stat().st_size)
                         for p in sorted((self.root / 'modal_results/anytok_pool_20k').glob('*-train.log'))]
            if signature and signature != previous:
                try:
                    result = subprocess.run([sys.executable, str(script)], cwd=self.root,
                        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=30)
                    if result.returncode:
                        raise RuntimeError(result.stderr[-700:])
                    previous = signature
                    self.plot_error = None
                except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
                    self.plot_error = f'Plot refresh failed; showing the previous image: {exc}'
            self.stop.wait(60)


ASSETS = {'/': ('index.html', 'text/html; charset=utf-8'),
          '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
          '/learning.js': ('learning.js', 'text/javascript; charset=utf-8'),
          '/style.css': ('style.css', 'text/css; charset=utf-8'),
          '/favicon.ico': ('favicon.ico', 'image/vnd.microsoft.icon'),
          '/favicon-32.png': ('favicon-32.png', 'image/png'),
          '/apple-touch-icon.png': ('apple-touch-icon.png', 'image/png'),
          '/Manrope.ttf': ('Manrope.ttf', 'font/ttf')}

PLOTS = {'/plots/natural_transfer.png': ('natural_transfer.png', 'image/png'),
         '/plots/frozen_policy_entropy0.png': ('frozen_policy_entropy0.png', 'image/png'),
         '/plots/frozen_policy_entropy0.pdf': ('frozen_policy_entropy0.pdf', 'application/pdf'),
         '/plots/frozen_policy_search.png': ('frozen_policy_search.png', 'image/png'),
         '/plots/frozen_policy_search.pdf': ('frozen_policy_search.pdf', 'application/pdf'),
         '/plots/corrected_pool_check.png': ('corrected_pool_check.png', 'image/png'),
         '/plots/corrected_pool_check.pdf': ('corrected_pool_check.pdf', 'application/pdf'),
         '/plots/random_pool_50.png': ('random_pool_50.png', 'image/png'),
         '/plots/random_pool_50.pdf': ('random_pool_50.pdf', 'application/pdf'),
         '/plots/natural_transfer.pdf': ('natural_transfer.pdf', 'application/pdf'),
         '/plots/model_comparison.png': ('model_comparison.png', 'image/png'),
         '/plots/model_comparison.pdf': ('model_comparison.pdf', 'application/pdf'),
         '/plots/anytok_pool_learning.png': ('anytok_pool_learning.png', 'image/png'),
         '/plots/anytok_pool_learning.pdf': ('anytok_pool_learning.pdf', 'application/pdf'),
         '/plots/pool_ood_48k.png': ('pool_ood_48k.png', 'image/png'),
         '/plots/pool_ood_48k.pdf': ('pool_ood_48k.pdf', 'application/pdf'),
         '/plots/uniform_weakness.png': ('uniform_weakness.png', 'image/png'),
         '/plots/uniform_weakness.pdf': ('uniform_weakness.pdf', 'application/pdf'),
         '/plots/uniform_weakness_rerun.png': ('uniform_weakness_rerun.png', 'image/png'),
         '/plots/uniform_weakness_rerun.pdf': ('uniform_weakness_rerun.pdf', 'application/pdf')}

for _figure in ('uniform_trajectory_bpb', 'uniform_trajectory_gap', 'uniform_trajectory_late', 'uniform_trajectory_plateau', 'tokenizer_speed', 'rl_ood_20k', 'coverage_evidence_overview', 'coverage_evidence_heatmap', 'coverage_evidence_policy', 'coverage_evidence_mechanism'):
    for _suffix, _mime in (('png', 'image/png'), ('pdf', 'application/pdf')):
        PLOTS[f'/plots/{_figure}.{_suffix}'] = (f'{_figure}.{_suffix}', _mime)


def make_server(monitor, host='0.0.0.0', port=1337, assets=None):
    assets = Path(assets or ROOT / 'monitoring/web')

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request = urlsplit(self.path)
            if request.path == '/api/state':
                args = parse_qs(request.query)
                try:
                    payload = monitor.state(*(args.get(k, [None])[0] for k in ('run', 'profile', 'worker')))
                    content = json.dumps(payload, ensure_ascii=False).encode()
                except KeyError:
                    self.send_error(404, 'Unknown run or profile')
                    return
                self.respond(content, 'application/json; charset=utf-8')
            elif request.path == '/api/learning':
                args = parse_qs(request.query)
                try:
                    payload = monitor.learning(args.get('run', [''])[0])
                except KeyError:
                    self.send_error(404, 'No learning metrics registered for this run')
                    return
                self.respond(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(),
                             'application/json; charset=utf-8')
            elif request.path == '/healthz':
                self.respond(b'{"ok":true}', 'application/json')
            elif request.path in ASSETS:
                name, mime = ASSETS[request.path]
                self.respond((assets / name).read_bytes(), mime)
            elif request.path in PLOTS:
                name, mime = PLOTS[request.path]
                path = monitor.root / 'plots' / name
                try:
                    content = path.read_bytes()
                except FileNotFoundError:
                    self.send_error(404, 'Plot has not been generated yet')
                    return
                self.respond(content, mime)
            else:
                self.send_error(404)

        def respond(self, content, mime):
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self'; font-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_):
            pass

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=1337)
    parser.add_argument('--config', type=Path, default=ROOT / 'plans/run-dashboard.json')
    parser.add_argument('--offline', action='store_true', help='Read artifacts and cache without Modal queries')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('port must be between 1 and 65535')
    monitor = Monitor(config=args.config, offline=args.offline)
    with make_server(monitor, args.host, args.port) as server:
        monitor.start()
        print(f'AnyTok run dashboard listening on http://{args.host}:{args.port}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            monitor.stop.set()


if __name__ == '__main__':
    main()
