import ast
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
pytest.importorskip('anytok_online')

import online_tokenizer_training as online
from run_online_tokenizer_training import TRAINER, command, prepare_code


class TestGenerator:
    __test__ = False
    corpus_bytes = 200_000_000
    names = [name for buckets in online.DOMAINS.values() for name in buckets]
    capacity = dict.fromkeys(names, corpus_bytes)

    def __init__(self, *args, **kwargs):
        self.reject_next = False

    def sample(self, **settings):
        if self.reject_next:
            self.reject_next = False
            raise ValueError('Only 32000 distinct eligible pieces for this request')
        tokenizer = SimpleNamespace(to_str=lambda: json.dumps(settings, sort_keys=True))
        priors = np.zeros(2, dtype=np.float32)
        forms = np.zeros((2, 10), dtype=np.int64)
        return SimpleNamespace(vocabulary=['a', 'b'], bytes_used=self.corpus_bytes, bucket_quotas={},
                               training_inputs=lambda *_: (tokenizer, forms, priors, 0))


@pytest.mark.parametrize('selector', ['reinforce', 'predictor', 'random'])
def test_online_restart_replays_proposals_and_observations(monkeypatch, selector):
    monkeypatch.setattr(online, 'Generator', TestGenerator)

    def make():
        return online.OnlineTokenizerTraining(selector, None, {}, 42, None)

    run = make()
    run.generator.reject_next = True
    _, _, _, event = run.sample(None, None, None)
    assert len(event['proposal_ineligible']) == 1
    assert event['proposal_ineligible'][0]['settings']['seed'] == event['tokenizer_settings']['seed']
    assert run.update(1.1) == {'policy_reward_bpb': 1.1, 'policy_updates': 1}
    run.sample(None, None, None)
    run.update(1.3)

    stored = io.BytesIO()
    torch.save(run.state_dict(), stored)
    stored.seek(0)
    resumed = make()
    resumed.load_state_dict(torch.load(stored, weights_only=True))
    original = run.sample(None, None, None)[-1]
    replayed = resumed.sample(None, None, None)[-1]
    original.pop('tokenizer_seconds')
    replayed.pop('tokenizer_seconds')
    assert replayed == original
    assert run.update(1.2) == resumed.update(1.2)
    for name, tensor in (run.weights_dict()['network'] or {}).items():
        assert torch.equal(tensor, resumed.weights_dict()['network'][name])


def test_recover_uncommitted_checkpoints_and_events(tmp_path):
    for step in (2000, 2500):
        (tmp_path / f'step-{step:06}').mkdir()
    events = tmp_path / 'events.jsonl'
    original = b'{"step":2000}\n{"step":2500}\n{"step":'
    events.write_bytes(original)
    online.rollback_uncommitted_steps(tmp_path, 2000)
    assert events.read_bytes() == b'{"step":2000}\n'
    assert (tmp_path / 'step-002000').is_dir()
    archive, = (tmp_path / 'superseded').iterdir()
    assert (archive / 'step-002500').is_dir()
    assert (archive / events.name).read_bytes() == original
    online.rollback_uncommitted_steps(tmp_path, 2000)
    assert list((tmp_path / 'superseded').iterdir()) == [archive]


def test_launcher_keeps_training_helpers_and_full_schedule(tmp_path):
    code = prepare_code(tmp_path)
    assert prepare_code(tmp_path) == code
    original = ast.parse((ROOT / 'runtime/anytok_pool_raw_bpb_rl_20k' / TRAINER).read_text())
    staged = ast.parse((code / TRAINER).read_text())
    helpers = {'evaluate', 'make_optimizer_groups', 'set_learning_rates', 'save_checkpoint', 'verify_corpus'}
    for name in helpers:
        before = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == name)
        after = next(node for node in staged.body if isinstance(node, ast.FunctionDef) and node.name == name)
        assert ast.dump(before) == ast.dump(after)
    args = SimpleNamespace(inputs=tmp_path / 'input', output=tmp_path / 'out', selector='reinforce',
                           stop_at_update=20000, resume=False)
    flags = command(args, code)
    assert flags[flags.index('--reward-objective') + 1] == 'raw_bpb'
    assert flags[flags.index('--stop-at-update') + 1] == '20000'
    assert flags[flags.index('--protected-steps') + 1:] == list(map(str, range(500, 20001, 500)))
