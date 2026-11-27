from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import io
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
pytest.importorskip('anytok_online')

from online_tokenizer_training import JointScorer, rollback_uncommitted_steps
from run_online_tokenizer_training import PREFIXES, PolicyState, can_prefetch, prepare_update, propose


class FakeBuilder:
    def __init__(self):
        self.settings = []

    def prepare(self, settings, _shared):
        self.settings.append(settings)
        if len(self.settings) == 1:
            raise ValueError('wiki-ja needs 200 bytes; this corpus provides 125')
        return settings


def test_direct_settings_restart_replays_draw_and_parameter_update():
    torch.set_num_threads(1)
    policy = PolicyState()
    rng = np.random.default_rng(43)
    preview, _ = policy.model.draw(np.random.default_rng(43))
    assert preview['vocab_size'] == 84900 and preview['pretokenizer'] == 'regex'

    def step(actor, generator, score):
        builder = FakeBuilder()
        settings, log_prob, rejected = propose(builder, actor, generator, None)
        assert len(builder.settings) == 2 and len(rejected) == 1
        assert settings == builder.settings[-1]
        assert settings['seed'] == rejected[0]['settings']['seed']
        return settings, actor.observe(log_prob, score)

    step(policy, rng, .9)
    step(policy, rng, 1.)
    assert any(p.detach().abs().max() > 0 for p in policy.model.parameters())
    stored = io.BytesIO()
    torch.save(dict(policy=policy.state_dict(), rng=rng.bit_generator.state), stored)
    stored.seek(0)
    saved = torch.load(stored, weights_only=True)
    resumed = PolicyState()
    resumed.restore(saved['policy'])
    next_rng = np.random.default_rng()
    next_rng.bit_generator.state = saved['rng']
    assert step(policy, rng, 1.2) == step(resumed, next_rng, 1.2)
    assert all(torch.equal(a, b) for a, b in zip(policy.model.parameters(), resumed.model.parameters(), strict=True))


def test_preparing_during_training_keeps_draws_rewards_and_cursors():
    class Corpus:
        names, weights = ['text'], [1]

        def __init__(self):
            self.position = 0

        def next_text(self, _, chars):
            self.position += 1
            return f'Window {self.position}'

    def run(worker):
        policy = PolicyState()
        scorer = SimpleNamespace(prepare=lambda config, shared: (config, shared.windows))
        rng, data_rng, corpus = np.random.default_rng(43), random.Random(42), Corpus()
        args = (scorer, policy, rng, corpus, data_rng)
        pending, rows = None, []
        student = 0.
        for step in range(1, 103):
            (strategy, windows), log_prob, _, proof, _ = pending.result() if pending else prepare_update(*args)
            raw_bpb = strategy['vocab_size'] / 100000 + student
            advantage = policy.observe(log_prob, raw_bpb)
            pending = worker.submit(prepare_update, *args) if worker and can_prefetch(step, 102) else None
            student += strategy['vocab_size'] / 1000000
            rows.append((strategy, windows, proof, raw_bpb, advantage, student))
            if step in PREFIXES or step == 102:
                assert pending is None and corpus.position == 128 * step
        return rows, policy.state_dict(), rng.bit_generator.state, data_rng.getstate()

    torch.set_num_threads(1)
    expected = run(None)
    with ThreadPoolExecutor(max_workers=1) as worker:
        actual = run(worker)
    assert expected[0] == actual[0] and expected[2:] == actual[2:]
    for key, value in expected[1]['model'].items():
        assert torch.equal(value, actual[1]['model'][key])


def test_update_uses_training_continuation_bpb_as_reward(monkeypatch):
    scorer = JointScorer.__new__(JointScorer)
    scorer.source, generated = torch.zeros(5, 2), torch.ones(3, 2)
    scorer.params = [generated.requires_grad_()]

    class HN:
        config = SimpleNamespace(pad_token_id=0)

        def train(self):
            pass

        def __call__(self, *_, **kw):
            return generated, None, None

    scorer.hn = HN()
    monkeypatch.setattr(torch, 'autocast', lambda *_, **kw: nullcontext())
    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', lambda *_, **kw: torch.tensor(1.))
    examples = [([1], [2, 3], 10)] * 128
    controls = [([4], [2, 3], 10)] * 128
    losses = iter((.5, .6, .7, .8))
    seen, rewards = [], []

    def make(micro, prefix_table, source, device):
        assert source is scorer.source
        seen.extend(micro)
        leaf = prefix_table
        return leaf.sum().reshape(1, 1, 1), None, None, None

    class LM:
        def train(self):
            pass

        def __call__(self, **kwargs):
            value = next(losses)
            return SimpleNamespace(loss=kwargs['inputs_embeds'].sum() * 0 + value)

    class Optimizer:
        def zero_grad(self, **_):
            pass

        def step(self):
            pass

    scorer.lm = LM()
    scorer.trainer = SimpleNamespace(
        set_learning_rates=lambda *_: (1e-4, 2e-5), make_lm_microbatch=make,
        input_lexical_loss=lambda *_: generated.sum() * 0,
        backward_hypernetwork=lambda generated_, gradient, lexical, optimized: generated_.sum().backward())
    metrics = dict(scored_bytes=1280, scored_tokens=256)
    row = scorer.update((examples, controls, range(3), None, metrics), Optimizer(), 1, rewards.append)
    expected_nats = sum(value * 64 for value in (.5, .6, .7, .8))
    assert seen == examples and row['transferred_nats'] == pytest.approx(expected_nats)
    assert row['raw_bpb'] == pytest.approx(expected_nats / (1280 * math.log(2)))
    assert rewards == [row['raw_bpb']]


def test_recover_uncommitted_checkpoints_and_events(tmp_path):
    for step in (500, 1000):
        (tmp_path / f'step-{step:06}').mkdir()
    events = tmp_path / 'events.jsonl'
    original = b'{"step":500}\n{"step":1000}\n{"step":'
    events.write_bytes(original)
    rollback_uncommitted_steps(tmp_path, 500)
    assert events.read_bytes() == b'{"step":500}\n'
    assert (tmp_path / 'step-000500').is_dir()
    archive, = (tmp_path / 'superseded').iterdir()
    assert (archive / 'step-001000').is_dir()
    assert (archive / events.name).read_bytes() == original
