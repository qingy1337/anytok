"""Monitoring counts, ETA limits, log integrity and the HTTP API."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
from urllib.request import urlopen, Request
from urllib.error import HTTPError
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from monitoring.state import Reader, registry, runtime_estimate, snapshot
from monitoring.modal_probe import ModalProbe, log_details, query

spec = importlib.util.spec_from_file_location('run_dashboard', ROOT / 'scripts/run_dashboard.py')
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)


class DashboardTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.directory = self.root / 'run'
        (self.directory / 'q').mkdir(parents=True)
        self.configs = [{'index': i, 'algorithm': 'bpe', 'pretokenizer': 'regex',
                         'vocab_size': 32000} for i in range(3)]
        self.write('plan.json', {'count': 3, 'configs': self.configs})
        self.run = {'id': 'pool', 'name': 'Test pool', 'adapter': 'tokenizer_pool',
                    'platform': 'Modal · CPU', 'plan': 'plan.json', 'directory': 'run',
                    'profiles': ['group-cron', 'q'], 'app_names': ['test-app'], 'concurrency': 100}
        self.write('registry.json', {'runs': [self.run]})
        self.manifest = {'status': 'complete', 'config': self.configs[0], 'train_seconds': 10,
                         'tokenizer_bytes': 100, 'tokenizer_sha256': 'same'}

    def tearDown(self):
        self.temp.cleanup()

    def write(self, path, data):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data))

    def test_consolidation_does_not_double_count_models(self):
        self.write('run/0000.json', self.manifest)
        self.write('run/q/0000.json', self.manifest)
        data = snapshot(Reader(self.root), self.run)
        self.assertEqual(data['completed'], 1)
        self.assertEqual(data['collected'], 1)
        self.assertEqual(data['output_bytes'], 100)
        self.assertEqual(next(p for p in data['profiles'] if p['name'] == 'q')['done'], 1)

    def test_partial_json_keeps_last_valid_state(self):
        reader = Reader(self.root)
        self.assertEqual(reader.json(self.root / 'plan.json')['count'], 3)
        (self.root / 'plan.json').write_text('{"unfinished":')
        self.assertEqual(reader.json(self.root / 'plan.json')['count'], 3)

    def test_resumed_job_keeps_completed_saved_work_in_progress(self):
        run = dict(id='resume', name='Resume', adapter='progress_json', platform='Modal',
            status='status.json', progress_log='events.jsonl', monotonic_progress=True,
            total=1000, profiles=[], logs=[])
        self.write('status.json', dict(completed=100, total=1000, stage='resuming'))
        (self.root / 'events.jsonl').write_text(json.dumps(dict(completed=758, stage='scoring')) + '\n')
        reader = Reader(self.root)
        self.assertEqual(snapshot(reader, run)['completed'], 758)
        (self.root / 'events.jsonl').write_text('')
        self.write('status.json', dict(completed=200, total=1000, stage='resuming'))
        self.assertEqual(snapshot(reader, run)['completed'], 758)

    def test_old_failure_does_not_hide_a_new_attempt(self):
        for name, when, failed in [('old.log', '2026-01-01T00:00:00Z', True),
                                   ('new.log', '2026-01-01T01:00:00Z', False)]:
            lines = [{'stage': 'starting', 'at_utc': when}, {'stage': 'training', 'index': 2}]
            if failed:
                lines.append({'stage': 'failed'})
            (self.directory / name).write_text('\n'.join(json.dumps(x) for x in lines) + '\n')
        data = snapshot(Reader(self.root), self.run)
        self.assertEqual([w['index'] for w in data['workers']], [2])
        self.assertTrue(data['workers'][0]['log'].endswith('new.log'))

    def test_eta_needs_comparable_samples_and_never_claims_overdue_job_finished(self):
        values = {i: {**deepcopy(self.manifest), 'train_seconds': seconds} for i, seconds in enumerate([10, 20, 30])}
        self.assertIsNone(runtime_estimate(self.configs[0], {0: values[0]})[0])
        self.assertEqual(runtime_estimate(self.configs[0], values, 5)[0], 15)
        self.assertIsNone(runtime_estimate(self.configs[0], values, 40)[0])
        different = {**self.configs[0], 'pretokenizer': 'no-regex'}
        self.assertIsNone(runtime_estimate(different, values)[0])

    def test_disabled_profiles_are_never_queried(self):
        with patch('monitoring.modal_probe.subprocess.run') as call:
            for profile in ('general', 'research-everettcc'):
                with self.assertRaises(ValueError):
                    query(profile, ['app', 'list', '--json'])
            call.assert_not_called()
        bad = {**self.run, 'profiles': ['general']}
        self.write('registry.json', {'runs': [bad]})
        with self.assertRaises(ValueError):
            registry(self.root, self.root / 'registry.json')

    def test_dashboard_cannot_launch_or_stop_jobs(self):
        with patch('monitoring.modal_probe.subprocess.run') as call:
            for args in (['run', 'example.py'], ['app', 'stop', 'app-id'], ['container', 'exec', 'id']):
                with self.assertRaises(ValueError):
                    query('q', args)
            call.assert_not_called()

    def test_remote_failure_retains_cached_data_and_marks_it_stale(self):
        probe = ModalProbe(self.root / 'cache.json')
        old = {'ok': True, 'checked_at': 123, 'containers': [{'container_id': 'a'}]}
        with patch('monitoring.modal_probe.query', side_effect=RuntimeError('offline')):
            _, row = probe.profile('q', [], old)
        self.assertFalse(row['ok'])
        self.assertEqual(row['checked_at'], 123)
        self.assertEqual(row['containers'], old['containers'])

    def test_container_logs_identify_tokenizer(self):
        jobs, logs = log_details('2026-01-01 ta-123 {"stage":"training","index":2}\n'
                                'ta-123 {"stage":"corpus_consumed"}')
        self.assertEqual(jobs['ta-123']['index'], 2)
        self.assertEqual(len(logs['ta-123']), 2)
        self.assertIn('corpus loaded', jobs['ta-123']['state'])

    def test_run_selector_supports_generic_local_workers(self):
        local = {'id': 'local', 'name': 'Local training', 'adapter': 'progress_json',
                 'platform': 'Local · CPU', 'status': 'status.json', 'completed_field': 'step',
                 'total': 10, 'profiles': [], 'logs': []}
        self.write('status.json', {'step': 5, 'eta_seconds': 12, 'workers': [
            {'id': 'worker1', 'task': 'Optimizer step', 'elapsed_seconds': 3, 'eta_seconds': 4}]})
        self.write('registry.json', {'runs': [self.run, local]})
        monitor = ui.Monitor(self.root, self.root / 'registry.json', offline=True)
        data = monitor.state('local')
        self.assertEqual(data['completed'], 5)
        self.assertIn('12s', data['eta'])
        self.assertEqual(data['workers'][0]['task'], 'Optimizer step')
        self.assertEqual(data['profiles'][0]['active'], 1)
        self.assertEqual(len(data['runs']), 2)

    def test_newer_logged_progress_overrides_stale_observer_metadata(self):
        run={'id':'probe','name':'Probe','adapter':'progress_json','platform':'Modal',
             'status':'probe.json','progress_log':'probe.log','total':432,'profiles':[],'logs':[]}
        self.write('probe.json',{'at_utc':'2026-11-10T01:00:00Z','stage':'failed','state':'failed','completed':100,'error':'old'})
        (self.root/'probe.log').write_text('{"at_utc":"2026-11-10T01:01:00Z","stage":"scoring","completed":200,"total":432}\n')
        result=snapshot(Reader(self.root),run)
        self.assertEqual(result['completed'],200)
        self.assertEqual(result['stage'],'scoring')
        (self.root/'probe.log').write_text('{"at_utc":"2026-11-10T00:59:00Z","stage":"scoring","completed":50}\n')
        self.assertEqual(snapshot(Reader(self.root),run)['completed'],100)

    def test_http_api_restricts_files_and_does_not_launch_compute(self):
        self.write('run/0000.json', self.manifest)
        (self.root/'plots').mkdir()
        (self.root/'plots/anytok_pool_learning.png').write_bytes(b'\x89PNG\r\n\x1a\nfixture')
        (self.root/'plots/anytok_pool_learning.pdf').write_bytes(b'%PDF-1.4\nfixture')
        (self.directory / 'q' / 'worker.log').write_text(
            '{"stage":"starting","at_utc":"2026-01-01T00:00:00Z"}\n'
            '{"stage":"training","index":2}\n')
        with patch('monitoring.modal_probe.query') as network:
            monitor = ui.Monitor(self.root, self.root / 'registry.json', offline=True)
            server = ui.make_server(monitor, '127.0.0.1', 0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f'http://127.0.0.1:{server.server_port}'
            try:
                with urlopen(url + '/api/state?run=pool') as response:
                    data = json.load(response)
                self.assertEqual(data['completed'], 1)
                self.assertNotIn('log', data['workers'][0])
                with urlopen(url + '/api/state?run=pool&worker=local-2') as response:
                    self.assertIn('"index":2', json.load(response)['logs'])
                for path in ('/', '/app.js', '/learning.js', '/style.css', '/Manrope.ttf', '/healthz',
                             '/favicon.ico?v=axolotl-1', '/favicon-32.png', '/apple-touch-icon.png'):
                    with urlopen(url + path) as response:
                        self.assertEqual(response.status, 200)
                for extension,mime in [('png','image/png'),('pdf','application/pdf')]:
                    with urlopen(url+f'/plots/anytok_pool_learning.{extension}?v=123') as response:
                        self.assertEqual(response.headers['Content-Type'],mime)
                        self.assertEqual(response.read(),(self.root/f'plots/anytok_pool_learning.{extension}').read_bytes())
                for path in ('/../../.modal.toml', '/scripts/run_dashboard.py', '/api/state?run=missing',
                             '/api/state?profile=general', '/api/learning?run=missing', '/api/learning?run=pool',
                             '/api/stop', '/plots/other.png',
                             '/plots/../../.modal.toml'):
                    with self.assertRaises(HTTPError) as error:
                        urlopen(url + path)
                    self.assertEqual(error.exception.code, 404)
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(url + '/api/state', data=b'{}', method='POST'))
                self.assertEqual(error.exception.code, 501)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
            network.assert_not_called()

    def test_log_merge_does_not_retimestamp_old_events_with_file_mtime(self):
        path = self.directory / 'worker.log'
        path.write_text('{"stage":"starting","at_utc":"2026-01-01T00:00:00Z"}\n'
                        '{"stage":"training","index":2}\n')
        logs = ui.read_logs({'workers': [], 'logs': [path]}, self.run, self.root)
        self.assertIn('2026-01-01T00:00:00Z', logs)
        self.assertNotIn('"index":2', logs)


if __name__ == '__main__':
    unittest.main()
