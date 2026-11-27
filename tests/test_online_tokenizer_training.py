from contextlib import nullcontext
import io
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
pytest.importorskip('anytok_online')

from online_tokenizer_training import JointScorer, rollback_uncommitted_steps
from run_online_tokenizer_training import PolicyState, propose


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


def test_reward_keeps_canonical_continuation_embeddings(monkeypatch):
    scorer = JointScorer.__new__(JointScorer)
    scorer.inference = nullcontext
    scorer.source, generated = torch.zeros(5, 2), torch.ones(3, 2)
    scorer.hn = lambda *_, **kw: (generated, None, None)
    scorer.lm = lambda **kw: SimpleNamespace(loss=torch.tensor(.5))
    monkeypatch.setattr(torch, 'autocast', lambda *_, **kw: nullcontext())
    examples, controls = [([1], [2, 3], 10)] * 128, [([4], [2, 3], 10)] * 128
    seen = []

    def make(micro, prefix_table, source, device):
        assert prefix_table is generated and source is scorer.source
        seen.extend(micro)
        return (None,) * 4

    def canonical_loss(_model, rows, source, batch, _make):
        assert rows == controls and source is scorer.source and batch == 32
        return 64.

    def gap(transfer, canonical, size):
        return transfer / size / math.log(2), canonical / size / math.log(2), (transfer - canonical) / size / math.log(2)

    scorer.trainer = SimpleNamespace(make_lm_microbatch=make, canonical_loss_nats=canonical_loss, paired_gap_bpb=gap)
    row = scorer.reward((examples, controls, range(3), None, dict(scored_bytes=1280)))
    assert seen == examples and row['transferred_nats'] == 128. and row['canonical_nats'] == 64.


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
