import math
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
pytest.importorskip('anytok_online')

from measure_dropout_reward import summarize


def test_dropout_summary_reports_noise_and_ranking():
    rows = [
        dict(dropout_off_bpb=1., dropout_on_bpb=[1.01, .99]),
        dict(dropout_off_bpb=2., dropout_on_bpb=[2.02, 2.01]),
        dict(dropout_off_bpb=3., dropout_on_bpb=[3.03, 3.02]),
        dict(dropout_off_bpb=4., dropout_on_bpb=[4.04, 4.03]),
    ]
    result = summarize(rows, 2)
    assert result['actions'] == 4 and result['dropout_repeats'] == 2
    assert result['spearman_rank_correlation']['minimum'] == pytest.approx(1.)
    assert result['top_quarter_overlap']['minimum'] == pytest.approx(1.)
    assert result['pairwise_order_disagreement']['maximum'] == pytest.approx(0.)
    expected = [0.01, -0.01, 0.02, 0.01, 0.03, 0.02, 0.04, 0.03]
    assert result['mean_signed_shift_bpb'] == pytest.approx(sum(expected) / len(expected))
    assert result['rms_shift_bpb'] == pytest.approx(math.sqrt(sum(value * value for value in expected) / len(expected)))
