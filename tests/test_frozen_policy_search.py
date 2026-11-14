from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from frozen_policy_search import balanced_scores, entropy_optimum, learner_step, policy_class, variant_settings, reward_values
from uniform_trajectory import PROFILE_DOMAINS


def test_full_text_coverage_is_required_and_all_nine_domains_are_weighted():
    import math
    rows = [dict(domain=d, continuation_bytes=10 if d == 'web-en' else 1)
            for d in PROFILE_DOMAINS for _ in range(32)]
    nats = np.array([r['continuation_bytes'] * math.log(2) * (9 if r['domain'] == 'web-en' else 0) for r in rows])
    assert balanced_scores(rows, nats) == pytest.approx(1.)
    with pytest.raises(ValueError, match='All 32 original'):
        balanced_scores(rows[:-1], nats[:-1])


def test_entropy_reference_matches_known_softmax_and_zero_weight():
    np.testing.assert_allclose(entropy_optimum([0., 1.], 1.), [1 / (1 + np.e), np.e / (1 + np.e)])
    np.testing.assert_array_equal(entropy_optimum([0., 1.], 0.), [0., 1.])


def test_ablation_only_changes_entropy_and_rejects_partial_coverage():
    from pool_generalization import digest
    cache = dict(plan=dict(indices=list(range(1000)), policy=dict(seed=43, learning_rate=.001,
        entropy_weight=.01, baseline_decay=.99, steps=20000)), rows=[{}] * 576,
        transferred_nats=np.zeros((1000, 576)))
    plan = dict(parent_plan_sha256=digest(cache['plan']), policy_changes={'entropy_weight': 0.})
    changed = variant_settings(cache, plan)
    assert changed == {**cache['plan']['policy'], 'entropy_weight': 0.}
    with pytest.raises(ValueError, match='Only the approved'):
        variant_settings(cache, {**plan, 'policy_changes': {'entropy_weight': 0., 'learning_rate': .01}})
    cache['transferred_nats'] = np.zeros((50, 576))
    with pytest.raises(ValueError, match='complete 1000'):
        variant_settings(cache, plan)


def test_learner_receives_only_selected_reward():
    class SelectedRewardOnly:
        def __getitem__(self, index):
            assert index == 7
            return 1.25
    class Policy:
        def sample(self): return 7
        def update(self, action, reward):
            assert action == 7 and reward == 1.25
            return {'ok': True}
    assert learner_step(Policy(), SelectedRewardOnly()) == (7, {'ok': True})


def test_regret_uses_same_canonical_baseline_and_preserves_objective_differences():
    ref = dict(bpb=np.array([.9, 1.1, 1.2]), canonical_bpb=1.)
    raw = reward_values(ref);regret = reward_values(ref, 'canonical_gap')
    np.testing.assert_allclose(regret, [-.1, .1, .2])
    for weight in [0., .01]:
        np.testing.assert_allclose(entropy_optimum(raw, weight), entropy_optimum(regret, weight), atol=1e-14)
    with pytest.raises(ValueError, match='Unknown'):
        reward_values(ref, 'unapproved')


def test_regret_plan_preserves_full_coverage_and_policy_settings():
    from pool_generalization import digest
    cache = dict(plan=dict(indices=list(range(1000)), policy=dict(seed=43, learning_rate=.001,
        entropy_weight=.01, baseline_decay=.99, steps=20000)), rows=[{}] * 576,
        transferred_nats=np.zeros((1000, 576)))
    plan = dict(parent_plan_sha256=digest(cache['plan']), reward_objective='canonical_gap', policy_changes={})
    assert variant_settings(cache, plan) == cache['plan']['policy']
    with pytest.raises(ValueError, match='Only the approved'):
        variant_settings(cache, {**plan, 'policy_changes': {'learning_rate': .1}})
    cache['transferred_nats'] = np.zeros((999, 576))
    with pytest.raises(ValueError, match='complete 1000'):
        variant_settings(cache, plan)


def test_actual_policy_ema_cancels_constant_reward_offset():
    torch.set_num_threads(1)
    Policy = policy_class()
    configs = [dict(algorithm='bpe', pretokenizer='regex', vocab_size=32000 + i * 32000,
        noise_per_million=0, domain_bytes={d: 1 for d in ['web', 'wikipedia', 'code', 'papers', 'math', 'qa', 'books', 'medical', 'legal']}) for i in range(3)]
    raw = np.array([.9, 1.1, 1.2]);offset = 1.
    for entropy in [0., .01]:
        left = Policy(configs, entropy_weight=entropy);right = Policy(configs, entropy_weight=entropy)
        for _ in range(200):
            a, m = learner_step(left, raw);b, n = learner_step(right, raw - offset)
            assert a == b
            assert m['policy_advantage_bpb'] == pytest.approx(n['policy_advantage_bpb'], abs=1e-12)
            assert m['policy_baseline_bpb'] - offset == pytest.approx(n['policy_baseline_bpb'], abs=1e-12)
        np.testing.assert_allclose(left.probabilities(), right.probabilities(), atol=1e-11, rtol=0.)


def test_confirmation_values_cannot_change_policy_trajectory():
    torch.set_num_threads(1)
    Policy = policy_class()
    configs = [dict(algorithm='bpe', pretokenizer='regex', vocab_size=32000 + i * 32000,
        noise_per_million=0, domain_bytes={d: 1 for d in ['web', 'wikipedia', 'code', 'papers', 'math', 'qa', 'books', 'medical', 'legal']}) for i in range(3)]
    left = Policy(configs);right = Policy(configs);train = np.array([1., 1.1, 1.2])
    # The training step has no confirmation argument or reference-ranking argument.
    for _ in range(50):
        assert learner_step(left, train) == learner_step(right, train)
    np.testing.assert_array_equal(left.probabilities(), right.probabilities())
