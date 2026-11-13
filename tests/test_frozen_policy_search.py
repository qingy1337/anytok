from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from frozen_policy_search import balanced_scores, entropy_optimum, learner_step, policy_class
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
