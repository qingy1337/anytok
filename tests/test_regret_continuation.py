"""The regret continuation (8,000 to 20,000): observer, driver, resume guard, restart fetch and status."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import fetch_regret_continuation_restart as fetch
import regret_continuation_status as collector
import resume_regret_continuation as guard
import run_regret_continuation as observer
from run_raw_bpb_training import prepare_code

USER = 'alice'
PASSED = dict(passed=True, checks=dict(trainer_exit_zero=True, same_batch_as_source=True))
POINTER = 'regret-rl-20k/latest_restart.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value))
    return path


def checkpoint(run, step, full):
    """A checkpoint folder as the trainer writes it: payload files listed with sizes and hashes."""
    folder = run / f'step-{step:06d}'
    files = {}
    for name in ['lm.pt', 'hypernet.pt'] + (['state.pt', 'optimizer.pt'] if full else []):
        write(folder / name, f'{step}:{name}')
        files[name] = dict(bytes=len(f'{step}:{name}'), sha256=digest(folder / name))
    kind = 'full_restart_checkpoint' if full else 'weights_only_checkpoint'
    write(folder / 'manifest.json', dict(step=step, kind=kind, files=files))
    return folder


def job(job_id, role, state):
    return dict(id=job_id, role=role, state=state, elapsed='00:10:00', exit_code='0:0', reason='')


def check_job(state):
    return job('1001', 'check', state)


def train_job(job_id, state):
    return job(job_id, 'allocation', state)


def cluster(**fields):
    """A status report from the cluster; nothing is submitted unless `fields` says so."""
    return dict(dict(jobs=[], check_receipt=None, driver_status=None, complete=None, run_status=None,
                     train=dict(last_step=None), backups=dict(receipts=0, pending=[])), **fields)


def cluster_state(root):
    """STATE after the first allocation stopped at update 12,000 and queued the second."""
    state = root / 'scrubbed' / USER / 'anytok-runs/regret-rl-20k'
    rows = [dict(stage='train', step=step, seconds=7.0, peak_gib=115.2) for step in range(7991, 8001)]
    rows += [dict(stage='execution_migration', step=8000)]
    rows += [dict(stage='train', step=step, seconds=6.85, peak_gib=115.9) for step in range(8001, 12001)]
    write(state / 'run/events.jsonl', ''.join(json.dumps(row) + '\n' for row in rows) + '{"stage": "train", "st')
    write(state / 'run/status.json', dict(step=12000, total=20000, stage='stopped'))
    write(state / 'check-receipt.json', dict(PASSED, update_8001={}))
    write(state / 'slurm-check-1001.log', '')
    write(state / 'allocations.json', ['2002'])
    write(state / 'current-job.json', dict(job_id='2003', resume_step=12000, queued_by='2002'))
    write(state / 'driver-status.json', dict(stage='handed_off', step=12000, next_job='2003', job_id='2002'))
    for step, full in [(8000, True), (11500, False), (12000, True)]:
        checkpoint(state / 'run', step, full)
    write(state / 'backups/step-011500.json', dict(step=11500))
    return state


def observer_args(local, *extra, state='/gpfs/scrubbed/$USER/anytok-runs/regret-rl-20k'):
    return ['--host', 'tillicum', '--control-path', '/x', '--python', sys.executable, '--remote-state', state,
            '--local-state', str(local), '--interval', '0', *extra]


def ssh_double(run):
    return SimpleNamespace(run=run, PIPE=subprocess.PIPE, DEVNULL=subprocess.DEVNULL,
                           TimeoutExpired=subprocess.TimeoutExpired)


STATES = [
    ('nothing submitted', {}, 'complete', 'waiting_for_check', None),
    ('check queued', dict(jobs=[check_job('PENDING')]), 'complete', 'checking', None),
    ('check running', dict(jobs=[check_job('RUNNING')]), 'check', 'checking', None),
    ('check passed', dict(jobs=[check_job('COMPLETED')], check_receipt=PASSED), 'check', 'check_passed', 0),
    ('training not submitted', dict(jobs=[check_job('COMPLETED')], check_receipt=PASSED),
     'complete', 'check_passed', None),
    ('check failed', dict(check_receipt=dict(passed=False, checks=dict(same_batch_as_source=False))),
     'check', 'check_failed', 2),
    ('check job failed', dict(jobs=[check_job('FAILED')]), 'check', 'check_failed', 2),
    ('check job timed out', dict(jobs=[check_job('TIMEOUT')]), 'check', 'check_failed', 2),
    ('training queued', dict(jobs=[check_job('COMPLETED'), train_job('2002', 'PENDING')], check_receipt=PASSED),
     'complete', 'queued', None),
    ('queued ends --until check', dict(jobs=[train_job('2002', 'PENDING')], check_receipt=PASSED),
     'check', 'queued', 0),
    ('training', dict(jobs=[train_job('2002', 'RUNNING')], check_receipt=PASSED, train=dict(last_step=9100)),
     'complete', 'training', None),
    ('walltime handoff', dict(jobs=[train_job('2002', 'TIMEOUT'), train_job('2003', 'PENDING')], check_receipt=PASSED),
     'complete', 'handed_off', None),
    ('clean handoff', dict(jobs=[train_job('2002', 'COMPLETED'), train_job('2003', 'PENDING')], check_receipt=PASSED),
     'complete', 'handed_off', None),
    ('second allocation', dict(jobs=[train_job('2002', 'COMPLETED'), train_job('2003', 'RUNNING')],
                               check_receipt=PASSED), 'complete', 'training', None),
    ('walltime without successor', dict(jobs=[train_job('2002', 'TIMEOUT')], check_receipt=PASSED),
     'complete', 'failed', 2),
    ('slurm failure', dict(jobs=[train_job('2002', 'FAILED')], check_receipt=PASSED), 'complete', 'failed', 2),
    ('cancelled', dict(jobs=[train_job('2002', 'CANCELLED by 5')], check_receipt=PASSED), 'complete', 'failed', 2),
    ('out of memory', dict(jobs=[train_job('2002', 'OUT_OF_MEMORY')], check_receipt=PASSED), 'complete', 'failed', 2),
    ('trainer failed', dict(jobs=[train_job('2002', 'RUNNING')], check_receipt=PASSED,
                            driver_status=dict(stage='failed', returncode=1)), 'complete', 'failed', 2),
    ('--until check ignores later failures', dict(check_receipt=PASSED, driver_status=dict(stage='failed')),
     'check', 'failed', 0),
    ('allocations used up', dict(jobs=[train_job('2003', 'COMPLETED')], check_receipt=PASSED,
                                 driver_status=dict(stage='needs_approval')), 'complete', 'failed', 2),
    ('paused', dict(check_receipt=PASSED, driver_status=dict(stage='paused')), 'complete', 'stopped', 2),
    ('final uploads', dict(jobs=[train_job('2002', 'COMPLETED')], check_receipt=PASSED, train=dict(last_step=20000),
                           backups=dict(pending=['step-020000'])), 'complete', 'uploading', None),
    ('complete', dict(jobs=[check_job('COMPLETED'), train_job('2002', 'COMPLETED')], check_receipt=PASSED,
                      complete=dict(step=20000)), 'complete', 'complete', 0),
    ('complete after a cancelled job', dict(jobs=[train_job('2002', 'CANCELLED')], complete=dict(step=20000)),
     'complete', 'complete', 0),
]


@pytest.mark.parametrize('fields,until,stage,code', [pytest.param(*case[1:], id=case[0]) for case in STATES])
def test_observer_maps_each_cluster_state_to_a_stage_and_exit_code(fields, until, stage, code):
    verdict = observer.evaluate(cluster(**fields), until)
    assert (verdict['stage'], verdict['exit'], verdict['ambiguous']) == (stage, code, False)


def test_observer_progress_stays_between_the_restart_and_the_target():
    assert observer.evaluate(cluster())['completed'] == 8000
    running = cluster(check_receipt=PASSED, run_status=dict(step=9000), train=dict(last_step=9100))
    assert observer.evaluate(running)['completed'] == 9100
    assert observer.evaluate(cluster(check_receipt=PASSED, train=dict(last_step=20004)))['completed'] == 20000
    assert observer.evaluate(cluster(complete={}))['completed'] == 20000


@pytest.mark.parametrize('text,seconds', [('1-00:00:00', 86400), ('22:50:25', 82225), ('00:01:59', 119),
                                          ('05:00', 300), ('', None), ('INVALID', None), (None, None)])
def test_slurm_elapsed_times_parse_to_seconds(text, seconds):
    assert observer.parse_elapsed(text) == seconds


@pytest.mark.parametrize('fields,first,final,error', [
    (dict(jobs=[check_job('COMPLETED')]), 'checking', 'check_failed', 'check-receipt.json is missing'),
    (dict(jobs=[train_job('2002', 'COMPLETED')], check_receipt=PASSED, train=dict(last_step=9000)),
     'training', 'failed', 'allocations ended at update 9000 (< 20000) without a handoff or completion'),
])
def test_an_unexplained_stop_is_reported_only_when_the_next_poll_confirms_it(tmp_path, fields, first, final, error):
    poller = observer.Observer(observer.parse_args(observer_args(tmp_path)))
    poller.fetch_status = lambda: cluster(**fields)
    assert poller.poll(0.0) is None
    status = read(tmp_path / 'status.json')
    assert status['stage'] == first and 'error' not in status
    assert poller.poll(60.0) == 2
    status = read(tmp_path / 'status.json')
    assert (status['stage'], status['state']) == (final, 'failed') and error in status['error']


@pytest.fixture
def local_ssh(tmp_path, monkeypatch):
    """ssh replaced by the local shell; sacct and squeue print the jobs of cluster_state()."""
    tools = tmp_path / 'tools'
    write(tools / 'sacct', '#!/bin/sh\necho "1001|COMPLETED|00:01:59|0:0"\necho "1001.batch|COMPLETED|00:01:59|0:0"\n'
                           'echo "2002|TIMEOUT|1-00:00:00|0:15"\n')
    write(tools / 'squeue', '#!/bin/sh\necho "2003|PENDING|0:00|(Priority)"\n')
    for path in tools.iterdir():
        path.chmod(0o755)
    monkeypatch.setenv('USER', USER)
    monkeypatch.setenv('PATH', f'{tools}{os.pathsep}{os.environ["PATH"]}')
    monkeypatch.setitem(observer.OPTS, 'control_path', '/x')
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.run(['sh', '-c', argv[-1]], **kwargs)
    monkeypatch.setattr(observer, 'subprocess', ssh_double(run))
    return calls


def test_observer_follows_the_cluster_through_one_batch_mode_ssh_command(tmp_path, local_ssh):
    state = cluster_state(tmp_path)
    remote = str(tmp_path / 'scrubbed/$USER/anytok-runs/regret-rl-20k')  # $USER expands on the cluster
    local = tmp_path / 'local'
    assert observer.Observer(observer.parse_args(observer_args(local, state=remote))).poll(0.0) is None
    status = read(local / 'status.json')
    assert (status['stage'], status['completed'], status['slurm']) == ('handed_off', 12000, 'ok')
    assert status['detail'] == 'allocation ended at update 12000; next allocation 2003 is queued'
    assert status['worker_jobs'] == {'1001': dict(state='COMPLETED', exit_code='0:0', elapsed_seconds=119),
                                     '2002': dict(state='TIMEOUT', exit_code='0:15', elapsed_seconds=86400),
                                     '2003': dict(state='PENDING', exit_code=None, elapsed_seconds=0)}
    assert status['backups'] == dict(receipts=1, pending=['step-012000'])
    assert status['train'] == dict(last_step=12000, seconds=6.85, peak_gib=115.9, rows_after_8000=4000)
    write(local / 'status.json', dict(status, observer_started_at='2026-11-21T10:45:03+00:00'))
    write(state / 'complete.json', dict(step=20000))
    assert observer.main(observer_args(local, state=remote)) == 0  # a restarted observer keeps its start time
    final = read(local / 'status.json')
    assert (final['stage'], final['completed']) == ('complete', 20000)
    assert final['observer_started_at'] == '2026-11-21T10:45:03+00:00' and final['elapsed_seconds'] > 0
    assert [row['stage'] for row in jsonl(local / 'events.jsonl')] == ['handed_off', 'complete']
    assert len(local_ssh) == 2
    assert all(argv[-2] == 'tillicum' and 'BatchMode=yes' in argv and 'ControlPath=/x' in argv and '-M' not in argv
               for argv in local_ssh)


def test_observer_waits_while_the_remote_state_folder_is_missing(tmp_path, local_ssh):
    poller = observer.Observer(observer.parse_args(observer_args(tmp_path, state=str(tmp_path / 'absent'))))
    assert poller.poll(0.0) is None
    status = read(tmp_path / 'status.json')
    assert (status['stage'], status['completed']) == ('waiting_for_check', 8000)
    assert 'does not exist' in status['detail']


@pytest.mark.parametrize('reply,reason', [
    ((255, '', 'no route to host'), '3 consecutive SSH failures'),
    ('timeout', '3 consecutive SSH failures'),
    ((1, '', 'Traceback'), 'the remote status command failed 3 times in a row'),
    ((0, 'not json\n', ''), 'the remote status command failed 3 times in a row'),
])
def test_observer_exits_3_when_it_loses_the_cluster(tmp_path, monkeypatch, reply, reason):
    def run(argv, **kwargs):
        if reply == 'timeout':
            raise subprocess.TimeoutExpired(argv, kwargs['timeout'])
        return subprocess.CompletedProcess(argv, *reply)
    monkeypatch.setattr(observer, 'subprocess', ssh_double(run))
    monkeypatch.setitem(observer.OPTS, 'control_path', None)
    assert observer.main(observer_args(tmp_path, '--max-ssh-failures', '3')) == 3
    status = read(tmp_path / 'status.json')
    assert status['stage'] == status['state'] == 'stopped'
    assert reason in status['error'] and 'the Slurm jobs are not affected' in status['error']
    assert [row['stage'] for row in jsonl(tmp_path / 'events.jsonl')] == ['waiting_for_check', 'stopped']


def test_interrupting_the_observer_records_a_stop_and_leaves_the_jobs_alone(tmp_path, monkeypatch):
    running = cluster(check_receipt=PASSED, jobs=[train_job('2002', 'RUNNING')],
                      train=dict(last_step=9000, seconds=7.0, peak_gib=115.2))
    monkeypatch.setattr(observer.Observer, 'fetch_status', lambda self: running)

    def interrupt(seconds):
        raise KeyboardInterrupt
    monkeypatch.setattr(observer, 'time', SimpleNamespace(time=time.time, sleep=interrupt))
    monkeypatch.setitem(observer.OPTS, 'control_path', None)
    assert observer.main(observer_args(tmp_path)) == 130
    training, stopped = jsonl(tmp_path / 'events.jsonl')
    assert (training['stage'], training['eta_seconds'], stopped['stage']) == ('training', 77000, 'stopped')
    status = read(tmp_path / 'status.json')
    assert (status['stage'], status['state'], status['completed']) == ('stopped', 'stopped', 9000)
    assert 'interrupted' in status['error'] and 'eta_seconds' not in status
    assert status['workers'][0]['state'] == 'Update 9000/20000 · 7.0 s/update · peak 115.2 GiB'


def test_observer_attaches_only_to_exactly_one_existing_control_socket(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    argv = ['--host', 'tillicum', '--python', 'python3', '--control-path', 'cm-tillicum-*']
    write(tmp_path / 'cm-tillicum-notes', '')
    with pytest.raises(SystemExit):
        observer.parse_args(argv)
    with socket.socket(socket.AF_UNIX) as first, socket.socket(socket.AF_UNIX) as second:
        first.bind('cm-tillicum-a')
        assert observer.parse_args(argv).control_path == 'cm-tillicum-a'
        second.bind('cm-tillicum-b')
        with pytest.raises(SystemExit):
            observer.parse_args(argv)


def test_status_reads_the_log_backwards_only_as_far_as_the_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(collector, 'TAIL_BLOCK', 50)  # many short reads that split lines
    out = collector.collect(str(cluster_state(tmp_path)), use_slurm=False)
    assert out['train'] == dict(last_step=12000, seconds=6.85, peak_gib=115.9, rows_after_8000=4000)
    assert out['backups'] == dict(receipts=1, pending=['step-012000'])
    assert out['check_receipt'] == PASSED and out['slurm'] == 'disabled'
    assert [(row['id'], row['role'], row['state']) for row in out['jobs']] == [
        ('1001', 'check', 'UNKNOWN'), ('2002', 'allocation', 'UNKNOWN'), ('2003', 'allocation', 'UNKNOWN')]
    monkeypatch.setattr(collector, 'TAIL_BLOCK', 3)
    path = write(tmp_path / 'short.jsonl', 'a\nbb\n\nccc\ndd')
    assert list(collector.reverse_lines(str(path))) == [b'dd', b'ccc', b'bb', b'a']


def test_a_check_job_stays_a_check_after_writing_the_driver_status(tmp_path):
    write(tmp_path / 'submissions.json', dict(check='1001', train=['2002']))
    write(tmp_path / 'driver-status.json', dict(stage='check_passed', job_id='1001'))
    jobs = collector.collect(str(tmp_path), use_slurm=False)['jobs']
    assert [(row['id'], row['role']) for row in jobs] == [('1001', 'check'), ('2002', 'allocation')]


def test_slurm_rows_come_from_sacct_and_live_squeue_rows_win(monkeypatch):
    replies = dict(sacct='1001|COMPLETED|00:01:59|0:0\n1001.batch|COMPLETED|00:01:59|0:0\n'
                         '2002|CANCELLED by 42|00:00:05|0:15\n2003|PENDING|00:00:00|0:0\n',
                   squeue='2003|RUNNING|1:02:03|g018\n')
    calls = []

    def run_cmd(argv):
        calls.append(argv)
        return replies[argv[0]]
    monkeypatch.setattr(collector, 'run_cmd', run_cmd)
    monkeypatch.setattr(collector, 'shutil', SimpleNamespace(which=lambda name: '/usr/bin/' + name))
    rows, slurm = collector.slurm_states(['1001', '2002', '2003'], True)
    assert slurm == 'ok'
    assert [rows[key]['state'] for key in ('1001', '2002', '2003')] == ['COMPLETED', 'CANCELLED', 'RUNNING']
    assert (rows['2003']['source'], rows['2003']['reason'], rows['2002']['exit_code']) == ('squeue', 'g018', '0:15')
    assert sorted(rows) == ['1001', '2002', '2003'] and all('1001,2002,2003' in argv for argv in calls)
    assert collector.slurm_states(['1001'], False) == ({}, 'disabled')
    monkeypatch.setattr(collector, 'shutil', SimpleNamespace(which=lambda name: None))
    assert collector.slurm_states(['1001'], True) == ({}, 'unavailable')


def test_status_prints_one_json_line_without_the_user_name(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('USER', USER)
    state = cluster_state(tmp_path)
    write(state / 'driver-status.json', dict(stage='failed', reason=f'/gpfs/projects/stf/{USER}/anytok missing'))
    assert collector.main(['--state', f'{state}/', '--no-slurm']) == 0
    text = capsys.readouterr().out
    assert text.count('\n') == 1 and USER not in text
    out = json.loads(text)
    assert out['state_dir'].endswith('/scrubbed/<user>/anytok-runs/regret-rl-20k')
    assert out['driver_status']['reason'] == '/gpfs/projects/stf/<user>/anytok missing'


@pytest.fixture
def driver(tmp_path, monkeypatch):
    """continue_regret_rl.py imported for user alice, its folders under tmp_path and a stand-in hub library."""
    monkeypatch.setenv('USER', USER)
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    monkeypatch.setattr(sys, 'path', list(sys.path))  # the driver prepends the cluster checkout's scripts/
    spec = importlib.util.spec_from_file_location('continue_regret_rl', ROOT / 'scripts/continue_regret_rl.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = tmp_path / 'scrubbed'
    for name, path in dict(STATE=state, RUN=state / 'run', CHECK_RUN=state / 'check-run', BACKUPS=state / 'backups',
                           INPUT=tmp_path / 'input').items():
        monkeypatch.setattr(module, name, path)
    hub = ModuleType('huggingface_hub')
    hub.CommitOperationAdd = lambda path_in_repo, path_or_fileobj: SimpleNamespace(
        path_in_repo=path_in_repo, path_or_fileobj=path_or_fileobj)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', hub)
    checkpoint(module.RUN, 8000, True)
    return module


class FakeHub:
    """In-memory Hugging Face repository; `damage` maps a file name to what its stored copy becomes."""

    def __init__(self, damage=None):
        self.files, self.commits, self.repos, self.damage = {}, [], set(), damage or {}

    def upload_folder(self, repo_id, repo_type, folder_path, path_in_repo, commit_message):
        for path in sorted(Path(folder_path).rglob('*')):
            if path.is_file():
                stored = self.damage.get(path.name, bytes)(path.read_bytes())
                self.files[f'{path_in_repo}/{path.relative_to(folder_path)}'] = stored
        return self.commit(repo_id, commit_message, [])

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        for operation in operations:
            self.files[operation.path_in_repo] = operation.path_or_fileobj
        return self.commit(repo_id, commit_message, [operation.path_in_repo for operation in operations])

    def commit(self, repo_id, message, paths):
        self.repos.add(repo_id)
        self.commits.append((message, paths))
        return SimpleNamespace(oid=f'rev{len(self.commits)}')

    def model_info(self, repo_id, revision, files_metadata):
        return SimpleNamespace(siblings=[SimpleNamespace(rfilename=name, size=len(data),
                                                         lfs=SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()))
                                         for name, data in self.files.items()])


@pytest.mark.parametrize('text,expected', [
    ('/gpfs/projects/stf/alice/anytok-data/input/model', '/gpfs/projects/stf/<user>/anytok-data/input/model'),
    ('"/gpfs/scrubbed/alice" and /mmfs1/home/alice', '"/gpfs/scrubbed/<user>" and /mmfs1/home/<user>'),
    ('/gpfs/scrubbed/alice2/run', '/gpfs/scrubbed/alice2/run'),
    ('/gpfs/scrubbed/alice.old/run', '/gpfs/scrubbed/alice.old/run'),
    ('/gpfs/scrubbed/alice-old/run', '/gpfs/scrubbed/alice-old/run'),
])
def test_driver_masks_the_user_as_a_whole_path_component(driver, text, expected):
    assert driver.redact(text) == expected
    assert driver.redact_value(dict(paths=[text])) == dict(paths=[expected])


def test_only_folders_with_state_and_manifest_are_full_restarts(driver):
    checkpoint(driver.RUN, 8500, False)
    (checkpoint(driver.RUN, 10000, True) / 'manifest.json').unlink()  # still being written
    checkpoint(driver.RUN, 12000, True)
    assert driver.full_restarts(driver.RUN) == [8000, 12000]


def test_set_aside_moves_work_past_the_resume_point_together_with_its_upload_receipts(driver):
    run, backups = driver.RUN, driver.BACKUPS
    write(run / 'events.jsonl', '')
    for step, full in [(8500, False), (10000, True), (10500, False)]:
        checkpoint(run, step, full)
    (run / '.writing-step-011000').mkdir()
    write(backups / 'step-008500.json', dict(step=8500))
    write(backups / 'step-010500.json', dict(step=10500))
    assert driver.set_aside('1001') == 10000
    assert sorted(p.name for p in run.iterdir()) == ['events.jsonl', 'interrupted-1001', 'step-008000', 'step-008500',
                                                     'step-010000']
    assert sorted(p.name for p in (run / 'interrupted-1001').iterdir()) == ['.writing-step-011000', 'step-010500']
    assert sorted(p.name for p in backups.glob('*.json')) == ['step-008500.json']
    assert (backups / 'interrupted-1001/step-010500.json').is_file()
    assert read(driver.STATE / 'driver-status.json')['moved'] == ['.writing-step-011000', 'step-010500']
    assert driver.set_aside('1002') == 10000 and not (run / 'interrupted-1002').exists()
    checkpoint(run, 10500, False)  # the resumed trainer writes 10,500 again; the old receipt must not hide it
    assert [p.name for p in driver.pending_backups()] == ['step-010000', 'step-010500']


def test_published_metadata_drops_a_torn_last_line_and_the_user_name(driver):
    write(driver.RUN / 'events.jsonl', '{"step": 8001, "p": "/gpfs/scrubbed/alice/run"}\n{"stage": "tra')
    write(driver.RUN / 'run_config.json', '{"model": "/gpfs/projects/stf/alice/anytok-data/input/model"}')
    assert driver.metadata_bytes('events.jsonl') == b'{"step": 8001, "p": "/gpfs/scrubbed/<user>/run"}\n'
    assert driver.metadata_bytes('run_config.json') == b'{"model": "/gpfs/projects/stf/<user>/anytok-data/input/model"}'


def test_completion_needs_update_20000_and_every_checkpoint_verified_on_the_hub(driver):
    hub = FakeHub()
    for name in driver.METADATA:
        write(driver.RUN / name, json.dumps(dict(model=f'/gpfs/projects/stf/{USER}/anytok-data/input/model')) + '\n')
    checkpoint(driver.RUN, 18000, True)
    assert driver.mark_complete_if_done(hub) is False and hub.commits == []
    checkpoint(driver.RUN, 19500, False)
    checkpoint(driver.RUN, 20000, True)
    assert driver.mark_complete_if_done(hub) is True
    assert read(driver.STATE / 'complete.json')['step'] == 20000
    assert read(driver.STATE / 'driver-status.json')['stage'] == 'complete'
    assert sorted(p.name for p in driver.BACKUPS.glob('step-*.json')) == ['step-018000.json', 'step-019500.json',
                                                                          'step-020000.json']
    assert [message for message, paths in hub.commits if POINTER in paths] == [
        'Verified metadata for regret continuation checkpoint 18000',
        'Verified metadata for regret continuation checkpoint 20000']
    pointer = json.loads(hub.files[POINTER])
    assert (pointer['step'], pointer['kind'], pointer['prefix']) == (20000, 'full_restart_checkpoint',
                                                                     'regret-rl-20k/checkpoints/step-020000')
    assert hub.repos == {'Dekiofsa/RLtokenizers'} and all(name.startswith('regret-rl-20k/') for name in hub.files)
    assert all(USER.encode() not in data for name, data in hub.files.items() if '/metadata/' in name)


@pytest.mark.parametrize('damage,error', [(lambda data: data[:-1], 'Backup size mismatch: state.pt'),
                                          (lambda data: data[:-1] + b'?', 'Backup hash mismatch: state.pt')])
def test_a_damaged_upload_is_retried_and_never_becomes_the_latest_restart(driver, monkeypatch, damage, error):
    sleeps = []
    monkeypatch.setattr(driver, 'time', SimpleNamespace(sleep=sleeps.append))
    checkpoint(driver.RUN, 20000, True)
    hub = FakeHub(damage={'state.pt': damage})
    assert driver.mark_complete_if_done(hub) is False
    assert len(sleeps) == 5 and POINTER not in hub.files
    assert not (driver.STATE / 'complete.json').exists() and not list(driver.BACKUPS.glob('step-*.json'))
    status = read(driver.STATE / 'driver-status.json')
    assert (status['stage'], status['pending']) == ('needs_upload', ['step-020000'])
    assert error in read(driver.STATE / 'backup-status.json')['error']


def test_a_checkpoint_is_checked_on_disk_before_anything_is_uploaded(driver):
    hub = FakeHub()
    folder = checkpoint(driver.RUN, 8500, False)
    write(folder / 'lm.pt', '8500:lm.pX')
    with pytest.raises(ValueError, match='Checkpoint changed before backup: lm.pt'):
        driver.backup(hub, folder)
    renamed = driver.RUN / 'step-009000'
    os.rename(checkpoint(driver.RUN, 8600, False), renamed)
    with pytest.raises(ValueError, match='Checkpoint step mismatch'):
        driver.backup(hub, renamed)
    assert hub.commits == [] and not driver.BACKUPS.exists()


def test_an_older_full_restart_never_moves_the_latest_pointer_back(driver):
    hub = FakeHub()
    write(driver.BACKUPS / 'step-020000.json', dict(step=20000, kind='full_restart_checkpoint'))
    driver.backup(hub, checkpoint(driver.RUN, 18000, True))
    assert POINTER not in hub.files and read(driver.BACKUPS / 'step-018000.json')['revision'] == 'rev1'


def test_provenance_is_published_once_and_without_the_user_name(driver):
    write(driver.STATE / 'restart-receipt.json', dict(
        files={'state.pt': dict(bytes=1, sha256='0' * 64)}, metadata={},
        source_events=dict(path='events-source-through-8001.jsonl', last_step=8001), events=dict(dropped_after_step=1)))
    write(driver.STATE / 'check-receipt.json', dict(passed=True, handoff=dict(
        stage='execution_migration', step=8000, note=f'/gpfs/projects/stf/{USER}/anytok-data/input/model')))
    hub = FakeHub()
    driver.publish_source(hub, 'bundle-sha256')
    driver.publish_source(hub, 'bundle-sha256')
    assert len(hub.commits) == 1 and read(driver.BACKUPS / 'source.json')['revision'] == 'rev1'
    raw = hub.files['regret-rl-20k/SOURCE.json']
    source = json.loads(raw)
    assert USER.encode() not in raw
    assert source['check']['handoff']['note'] == '/gpfs/projects/stf/<user>/anytok-data/input/model'
    assert (source['continuation_of'], source['source_revision'], source['start_step'], source['stop_step']) == (
        'cminst/anytok-regret-rl-20k', '89bbf8d4cb9f5fb40399d4c4ea5ed51083499b51', 8000, 20000)
    assert source['trainer_sha256'] == driver.TRAINER_SHA256
    assert source['input_bundle_manifest_sha256'] == 'bundle-sha256'
    assert hub.files['regret-rl-20k/README.md'].startswith(b'# Regret RL continuation, updates 8,000 to 20,000')


def test_model_and_warmup_inputs_must_match_the_bundle_manifest(driver):
    files = {}
    for name in ['model/config.json', 'model/model.safetensors', 'model/tokenizer.json', 'warmup/hypernet.pt',
                 'warmup/settings.json']:
        files[name] = dict(bytes=len(name), sha256=digest(write(driver.INPUT / name, name)))
    manifest = write(driver.INPUT / 'manifest.json', dict(files=files))
    assert driver.verify_inputs() == digest(manifest)
    write(driver.INPUT / 'warmup/hypernet.pt', 'warmup/hypernet.pX')
    with pytest.raises(ValueError, match='Input differs from its bundle manifest: warmup/hypernet.pt'):
        driver.verify_inputs()
    del files['warmup/hypernet.pt']
    write(manifest, dict(files=files))
    with pytest.raises(ValueError, match='Expected five'):
        driver.verify_inputs()


def test_this_checkout_reproduces_the_as_run_trainer_and_staged_code(driver, tmp_path, monkeypatch):
    trainer = ROOT / 'runtime/anytok_pool_raw_bpb_rl_20k/scripts/train_joint_smollm_input.py'
    assert digest(trainer) == driver.TRAINER_SHA256 == guard.TRAINER_SHA256
    code = prepare_code(tmp_path / 'runs')
    assert code.name == driver.CODE.name
    monkeypatch.setattr(driver, 'CODE', code)
    assert driver.verify_code() == code
    edited = code / 'scripts/lossless_mlp.py'
    edited.write_text(edited.read_text() + '# edited\n')
    with pytest.raises(ValueError, match='Staged code differs from the checkout: scripts/lossless_mlp.py'):
        driver.verify_code()


def configs():
    """The source run's saved config at step 8,000 and the continuation's request: new input locations only."""
    saved = read(ROOT / 'docs/raw-bpb-handoff/training-verification.json')['config']
    saved.update(reward_objective='regret', policy_reward=guard.REGRET_REWARD,
                 protected_steps=list(range(500, 20001, 500)), model='/gpfs/scrubbed/source/anytok/input/model',
                 warmup_checkpoint='/gpfs/scrubbed/source/anytok/input/warmup')
    requested = deepcopy(saved)
    requested.update(model=f'/gpfs/projects/stf/{USER}/anytok-data/input/model',
                     warmup_checkpoint=f'/gpfs/projects/stf/{USER}/anytok-data/input/warmup')
    return saved, requested


def test_training_and_check_commands_request_the_saved_settings(driver):
    saved, _ = configs()
    for check, stop, protected in [(False, 20000, saved['protected_steps']), (True, 8002, list(range(500, 8001, 500)))]:
        args, cmd = driver.trainer_command(Path('/code'), driver.RUN, stop, check)
        split = cmd.index('--')
        assert cmd[:split] == [sys.executable, '-u', str(ROOT / 'scripts/resume_regret_continuation.py'),
                               '--code', '/code'] + ['--check'] * check
        arguments = cmd[split + 1:]
        value = dict(zip(arguments, arguments[1:]))
        assert arguments[0] == '--recipe' and arguments[-1] == '--resume' and args.stop_at_update == stop
        assert [int(step) for step in arguments[arguments.index('--protected-steps') + 1:-1]] == protected
        assert (value['--reward-objective'], value['--memory-execution'], int(value['--microbatch-size']),
                int(value['--seed'])) == (saved['reward_objective'], saved['memory_execution'],
                                          saved['microbatch_size'], saved['seed'])
        assert [float(value[flag]) for flag in ('--policy-entropy', '--policy-lr', '--policy-baseline-decay')] == [
            saved['policy'][key] for key in ('entropy_weight', 'learning_rate', 'baseline_decay')]
        assert value['--model'] == str(driver.INPUT.resolve() / 'model')


def resume_check_run(driver, monkeypatch, change, returncode):
    """The source run's logged update 8,001 and a fake trainer that logs the check's handoff and two updates."""
    source = dict(stage='train', step=8001, tokenizer_pool_index=771, paired_batch_sha256='add5', paired_gap_bpb=0.0451,
                  paired_canonical_bpb=0.87, policy_reward_bpb=0.0451, train_bpb=0.91,
                  loss_nats_per_token=2.0617054746295866, policy_action_probability=0.98, seconds=7.1, peak_gib=115.18)
    for name in driver.METADATA[:-1]:
        write(driver.RUN / name, '{}\n')
    write(driver.RUN / 'events.jsonl', json.dumps(dict(stage='train', step=8000)) + '\n')
    write(driver.RUN / 'events-source-through-8001.jsonl', json.dumps(dict(stage='train', step=8000)) + '\n'
          + json.dumps(source) + '\n')

    def run(cmd, args, env):
        assert '--check' in cmd[:cmd.index('--')] and (args.output, args.stop_at_update) == (driver.CHECK_RUN, 8002)
        rows = [dict(stage='execution_migration', step=8000, changed=['model', 'protected_steps', 'warmup_checkpoint'])]
        rows += [dict(source, step=step, policy_updates=step, paired_valid_examples=128, paired_maximum_positions=128,
                      reward_objective='regret') for step in (8001, 8002)]
        if change:
            change(rows)
        with open(driver.CHECK_RUN / 'events.jsonl', 'a') as stream:
            stream.writelines(json.dumps(row) + '\n' for row in rows)
        checkpoint(driver.CHECK_RUN, 8002, True)
        return returncode
    monkeypatch.setattr(driver, 'run_and_monitor', run)


@pytest.mark.parametrize('change,returncode,failed', [
    (None, 0, []),
    (lambda rows: rows[1].update(tokenizer_pool_index=314), 0, ['same_tokenizer_as_source']),
    (lambda rows: rows[1].update(paired_batch_sha256='other'), 0, ['same_batch_as_source']),
    (lambda rows: rows[0].update(changed=['hn_peak_lr', 'model']), 0, ['handoff_recorded']),
    (lambda rows: rows[2].update(policy_reward_bpb=0.87), 0, ['regret_reward']),
    (lambda rows: rows[2].update(paired_valid_examples=127), 0, ['valid_examples']),
    (lambda rows: rows.pop(), 0, ['policy_counter', 'updates_8001_8002']),
    (None, 1, ['trainer_exit_zero']),
], ids=['reproduced', 'other-tokenizer', 'other-batch', 'material-handoff', 'raw-reward', 'short-batch', 'one-update',
        'trainer-failed'])
def test_resume_check_passes_only_when_update_8001_reproduces_the_source_run(driver, monkeypatch, change, returncode,
                                                                              failed):
    resume_check_run(driver, monkeypatch, change, returncode)
    if failed:
        with pytest.raises(SystemExit):
            driver.check(Path('code'), 'bundle')
    else:
        driver.check(Path('code'), 'bundle')
    receipt = read(driver.STATE / 'check-receipt.json')
    assert receipt['passed'] == (not failed) and len(receipt['checks']) == 10
    assert sorted(name for name, ok in receipt['checks'].items() if not ok) == failed
    assert receipt['update_8001']['paired_batch_sha256']['source'] == 'add5'
    assert read(driver.STATE / 'driver-status.json')['stage'] == ('check_failed' if failed else 'check_passed')
    restart = 'step-008000/state.pt'
    assert os.stat(driver.CHECK_RUN / restart).st_ino == os.stat(driver.RUN / restart).st_ino  # linked, not copied
    with pytest.raises(SystemExit, match='check-run/ already exists'):
        driver.check(Path('code'), 'bundle')


def training_ready(driver, monkeypatch, allocations=None):
    """A passed check, provenance already published, uploads stubbed and sbatch recorded."""
    write(driver.STATE / 'check-receipt.json', PASSED)
    write(driver.BACKUPS / 'source.json', dict(revision='rev0'))
    if allocations is not None:
        write(driver.STATE / 'allocations.json', allocations)
    monkeypatch.setenv('HF_TOKEN', 'placeholder')
    monkeypatch.setattr(driver, 'hf_api', FakeHub)
    monkeypatch.setattr(driver, 'backup_all', lambda api: None)
    monkeypatch.setattr(os, 'setpriority', lambda *args: None)  # the upload thread renices itself on Linux
    submitted = []

    def sbatch(argv, **kwargs):
        submitted.append(argv)
        return SimpleNamespace(stdout='1002;tillicum\n')
    monkeypatch.setattr(driver, 'subprocess', SimpleNamespace(run=sbatch, SubprocessError=subprocess.SubprocessError))
    return submitted


def fake_trainer(driver, monkeypatch, job_id, saves, returncode=0):
    """Run as Slurm job `job_id`; the trainer saves a full restart at update `saves` and exits."""
    monkeypatch.setenv('SLURM_JOB_ID', job_id)
    calls = []

    def run(cmd, args, env):
        calls.append((cmd, args, env))
        if saves:
            checkpoint(driver.RUN, saves, True)
        return returncode
    monkeypatch.setattr(driver, 'run_and_monitor', run)
    return calls


def test_a_third_allocation_or_an_unchecked_run_is_refused_before_anything_moves(driver, monkeypatch):
    submitted = training_ready(driver, monkeypatch, allocations=['1001', '1002'])
    checkpoint(driver.RUN, 12000, True)
    checkpoint(driver.RUN, 12500, False)
    calls = fake_trainer(driver, monkeypatch, '1003', 14000)
    with pytest.raises(SystemExit, match='Both approved training allocations are used'):
        driver.train(Path('code'), 'bundle')
    assert calls == [] and submitted == [] and (driver.RUN / 'step-012500').is_dir()
    assert read(driver.STATE / 'allocations.json') == ['1001', '1002']
    assert read(driver.STATE / 'driver-status.json')['stage'] == 'needs_approval'
    write(driver.STATE / 'check-receipt.json', dict(passed=False))
    with pytest.raises(SystemExit, match='resume check has not passed'):
        driver.train(Path('code'), 'bundle')


def test_the_first_allocation_queues_one_follow_up_and_the_second_finishes_without_another(driver, monkeypatch):
    submitted = training_ready(driver, monkeypatch)
    calls = fake_trainer(driver, monkeypatch, '1001', 13217)  # a walltime stop between milestones
    driver.train(Path('code'), 'bundle')
    assert submitted == [['sbatch', '--parsable', '--dependency=afterany:1001',
                          str(ROOT / 'scripts/continue_regret_rl.sbatch')]]
    current = read(driver.STATE / 'current-job.json')
    assert (current['job_id'], current['resume_step'], current['queued_by']) == ('1002', 13217, '1001')
    assert read(driver.STATE / 'driver-status.json')['stage'] == 'handed_off'
    cmd, args, env = calls[0]
    assert (args.output, args.stop_at_update, '--check' in cmd) == (driver.RUN, 20000, False)
    assert 'HF_TOKEN' not in env and env['HF_HUB_OFFLINE'] == '1'
    fake_trainer(driver, monkeypatch, '1002', 18000)
    driver.train(Path('code'), 'bundle')
    status = read(driver.STATE / 'driver-status.json')
    assert len(submitted) == 1 and (status['stage'], status['step']) == ('needs_approval', 18000)
    fake_trainer(driver, monkeypatch, '1002', 20000)  # a requeued second allocation keeps its approval
    driver.train(Path('code'), 'bundle')
    assert len(submitted) == 1 and read(driver.STATE / 'allocations.json') == ['1001', '1002']
    assert read(driver.STATE / 'complete.json')['step'] == 20000
    assert read(driver.STATE / 'driver-status.json')['stage'] == 'complete'


@pytest.mark.parametrize('saves,returncode,reason', [(14000, 1, 'Trainer exited 1'),
                                                     (None, 0, 'No committed training progress')])
def test_a_failed_or_stalled_segment_queues_nothing(driver, monkeypatch, saves, returncode, reason):
    submitted = training_ready(driver, monkeypatch)
    fake_trainer(driver, monkeypatch, '1001', saves, returncode)
    with pytest.raises(RuntimeError, match=reason):
        driver.train(Path('code'), 'bundle')
    assert submitted == [] and read(driver.STATE / 'driver-status.json')['stage'] == 'failed'


def test_guard_allows_only_new_input_locations_and_only_at_step_8000(monkeypatch):
    monkeypatch.setenv('USER', USER)
    saved, requested = configs()
    event = guard.handoff(saved, requested, 8000)
    assert (event['stage'], event['step'], event['check_run'], event['changed']) == (
        'execution_migration', 8000, False, ['model', 'warmup_checkpoint'])
    assert event['locations'] == dict(model='/gpfs/projects/stf/<user>/anytok-data/input/model',
                                      warmup_checkpoint='/gpfs/projects/stf/<user>/anytok-data/input/warmup')
    assert guard.handoff(requested, requested, 14000) is None
    with pytest.raises(ValueError, match='only when resuming the step-8,000 restart'):
        guard.handoff(saved, requested, 10000)


@pytest.mark.parametrize('key,value', [('pool_size', 999), ('sequence_length', 256), ('recipe_sha256', 'different'),
    ('data_manifest_sha256', 'different'), ('hn_peak_lr', 0.1), ('microbatch_size', 4), ('seed', 7),
    ('pairing_protocol', 'old-cuts'), ('main_updates', 20000), ('protected_steps', list(range(500, 8001, 500))),
    ('added_setting', 1)])
def test_guard_refuses_material_changes(key, value):
    saved, requested = configs()
    requested[key] = value
    with pytest.raises(ValueError, match='Material training settings changed: ' + key):
        guard.handoff(saved, requested, 8000)


@pytest.mark.parametrize('side', [0, 1], ids=['saved', 'requested'])
@pytest.mark.parametrize('change', [
    lambda config: config.update(reward_objective='raw_bpb'),
    lambda config: config.update(policy_reward='paired_pre_model_update_transferred_bpb'),
    lambda config: config['policy'].update(entropy_weight=0.01),
    lambda config: config.update(training_script_sha256='0' * 64),
    lambda config: config.update(memory_execution='reference'),
], ids=['raw-objective', 'raw-reward', 'entropy', 'other-trainer', 'reference-memory'])
def test_guard_continues_only_the_entropy_zero_regret_run_with_its_trainer(change, side):
    pair = configs()
    change(pair[side])
    with pytest.raises(ValueError, match='Only the entropy-zero regret run'):
        guard.handoff(*pair, 8000)


def test_guard_refuses_a_different_input_behind_a_new_location():
    saved, requested = configs()
    requested['warmup_checkpoint'] += '-v2'
    with pytest.raises(ValueError, match='Input changed beyond its location: warmup_checkpoint'):
        guard.handoff(saved, requested, 8000)


def test_a_check_run_may_only_protect_fewer_of_the_saved_milestones():
    saved, requested = configs()
    requested['protected_steps'] = list(range(500, 8001, 500))
    event = guard.handoff(saved, requested, 8000, check=True)
    assert event['check_run'] is True and event['changed'] == ['model', 'protected_steps', 'warmup_checkpoint']
    requested['protected_steps'] = list(range(500, 20501, 500))
    with pytest.raises(ValueError, match='only protect a subset'):
        guard.handoff(saved, requested, 8000, check=True)


TRAINER = '''import json
from pathlib import Path
import sys


def check_resume_settings(saved, requested, allow_microbatch, checkpoint_step, allow_min_chars=False):
    return 'trainer'


def main():
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    saved, requested = (json.loads((output / name).read_text()) for name in ('saved.json', 'requested.json'))
    result = check_resume_settings(saved, requested, False, 8000)
    (output / 'result.json').write_text(json.dumps(dict(argv=sys.argv[1:], result=result)))
'''


@pytest.fixture
def fake_code(tmp_path, monkeypatch):
    """A staged tree whose trainer records what its resume-settings check returned."""
    monkeypatch.setenv('USER', USER)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    write(tmp_path / 'scripts/train_joint_smollm_input.py', TRAINER)
    for name, config in zip(['saved.json', 'requested.json'], configs()):
        write(tmp_path / name, config)
    return tmp_path


def run_guard(code, monkeypatch, *options, trainer_args):
    monkeypatch.setattr(sys, 'argv', ['resume_regret_continuation.py', '--code', str(code), *options, '--',
                                      *trainer_args])
    guard.main()
    return read(code / 'result.json')


@pytest.mark.parametrize('options', [[], ['--check']])
def test_guard_replaces_only_the_trainers_resume_settings_check(fake_code, monkeypatch, options):
    trainer_args = ['--output', str(fake_code), '--reward-objective', 'regret', '--resume']
    result = run_guard(fake_code, monkeypatch, *options, trainer_args=trainer_args)
    assert result['argv'] == trainer_args and USER not in json.dumps(result['result'])
    assert (result['result']['stage'], result['result']['check_run']) == ('execution_migration', bool(options))
    shutil.copy(fake_code / 'requested.json', fake_code / 'saved.json')
    assert run_guard(fake_code, monkeypatch, *options, trainer_args=trainer_args)['result'] == 'trainer'


@pytest.mark.parametrize('trainer_args', [['--reward-objective', 'regret'],
                                          ['--resume', '--reward-objective', 'raw_bpb']])
def test_guard_requires_an_explicit_regret_resume(fake_code, monkeypatch, trainer_args):
    with pytest.raises(ValueError, match='Explicit regret resume required'):
        run_guard(fake_code, monkeypatch, trainer_args=['--output', str(fake_code), *trainer_args])
    assert not (fake_code / 'result.json').exists()


def source_repository(root, monkeypatch):
    """What snapshot_download fetches from the source run: the step-8,000 restart, its metadata and pointer."""
    manifest = checkpoint(root / 'checkpoints', 8000, True) / 'manifest.json'
    monkeypatch.setattr(fetch, 'POINTER_MANIFEST_SHA256', digest(manifest))
    write(root / 'latest_restart.json', dict(step=8000, kind='full_restart_checkpoint',
                                             prefix='checkpoints/step-008000', manifest_sha256=digest(manifest)))
    write(root / 'metadata/run_config.json', dict(reward_objective='regret', seed=42,
                                                  policy=dict(entropy_weight=0.0, seed=43)))
    for name in ['run_config.modal.json', 'initial_eval.json', 'heldout_tokenizers.json']:
        write(root / 'metadata' / name, {})
    rows = [dict(stage='initial_eval'), dict(stage='train', step=8000), dict(stage='checkpoint', step=8000),
            dict(stage='train', step=8001)]
    write(root / 'metadata/events.jsonl', ''.join(json.dumps(row) + '\n' for row in rows))
    return root


def fake_download(monkeypatch, source):
    calls = []

    def snapshot_download(repo_id, repo_type, revision, local_dir, allow_patterns, max_workers):
        calls.append((repo_id, revision, allow_patterns))
        shutil.copytree(source, local_dir, dirs_exist_ok=True)
    hub = ModuleType('huggingface_hub')
    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, 'huggingface_hub', hub)
    return calls


def test_fetch_arranges_a_verified_read_only_restart_and_sets_update_8001_aside(tmp_path, monkeypatch):
    source = source_repository(tmp_path / 'hub', monkeypatch)
    calls = fake_download(monkeypatch, source)
    state = tmp_path / 'state'
    monkeypatch.setattr(sys, 'argv', ['fetch_regret_continuation_restart.py', '--state', str(state)])
    fetch.main()
    assert calls == [('cminst/anytok-regret-rl-20k', '89bbf8d4cb9f5fb40399d4c4ea5ed51083499b51',
                      ['checkpoints/step-008000/*', 'metadata/*', 'latest_restart.json'])]
    run, restart = state / 'run', state / 'run/step-008000'
    assert [row.get('step') for row in jsonl(run / 'events.jsonl')] == [None, 8000, 8000]
    assert (run / 'events-source-through-8001.jsonl').read_bytes() == (source / 'metadata/events.jsonl').read_bytes()
    assert sorted(p.name for p in run.iterdir()) == ['events-source-through-8001.jsonl', 'events.jsonl',
                                                     'heldout_tokenizers.json', 'initial_eval.json', 'run_config.json',
                                                     'run_config.modal.json', 'step-008000']
    assert not any(path.stat().st_mode & 0o222 for path in [restart, *restart.rglob('*')])
    receipt = read(state / 'restart-receipt.json')
    assert receipt['events'] == dict(rows=3, dropped_after_step=1, sha256=digest(run / 'events.jsonl'))
    assert receipt['source_events']['last_step'] == 8001
    assert receipt['total_bytes'] == sum(row['bytes'] for row in read(restart / 'manifest.json')['files'].values())
    assert read(state / 'fetch-status.json')['stage'] == 'complete'
    with pytest.raises(SystemExit, match='refusing to overwrite'):
        fetch.main()
    for path in [restart, *restart.rglob('*')]:
        path.chmod(path.stat().st_mode | 0o200)  # so pytest can remove tmp_path


@pytest.mark.parametrize('tamper,error', [
    (lambda hub: write(hub / 'checkpoints/step-008000/state.pt', '8000:state.pX'), 'Checksum mismatch: state.pt'),
    (lambda hub: write(hub / 'checkpoints/step-008000/extra.pt', 'x'), 'differ from the manifest: extra.pt'),
    (lambda hub: write(hub / 'checkpoints/step-008000/manifest.json', '{}'),
     'manifest differs from the published pointer'),
    (lambda hub: write(hub / 'latest_restart.json', dict(step=6000, kind='full_restart_checkpoint',
                                                        prefix='checkpoints/step-006000')), 'pointer does not name'),
    (lambda hub: write(hub / 'metadata/run_config.json', dict(reward_objective='regret', seed=42,
                                                             policy=dict(entropy_weight=0.01, seed=43))),
     'not the entropy-zero regret run'),
], ids=['changed-file', 'extra-file', 'other-manifest', 'other-pointer', 'entropy-run'])
def test_fetch_arranges_nothing_unless_every_file_verifies(tmp_path, monkeypatch, tamper, error):
    tamper(source_repository(tmp_path / 'hub', monkeypatch))
    fake_download(monkeypatch, tmp_path / 'hub')
    monkeypatch.setattr(sys, 'argv', ['fetch_regret_continuation_restart.py', '--state', str(tmp_path / 'state')])
    with pytest.raises(ValueError, match=error):
        fetch.main()
    assert not (tmp_path / 'state/run').exists()
