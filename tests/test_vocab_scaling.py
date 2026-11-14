"""Guard the timing definition, held-out boundary, and paired uncertainty."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from plot_vocab_scaling import analyze,seconds


def row(size,method,repeat,value,fit=True):
    return dict(vocabulary=size,method=method,repeat=repeat,fit=fit,
                result=dict(phases=[dict(phase='load model and complete optimizer',seconds=1000),
                                    dict(phase='hypernetwork forward',seconds=value)]))


def test_target_measurement_cannot_leak_into_fitted_curve():
    rows=[]
    for size in [32000,34000,36000,37000,40000,43000]:
        for repeat in range(3):
            for method,(a,b) in {'original':(1,.05),'gpu_new':(1,.06),'staged_new':(5,.1)}.items():
                rows.append(row(size,method,repeat,a+b*size/1000))
    rows.append(row(128000,'staged_new',0,99,False))
    analysis,_=analyze(rows,draw_count=60)
    assert analysis['fits']['original']['prediction_128_seconds']==pytest.approx(7.4)
    assert analysis['fits']['staged_new']['prediction_128_seconds']==pytest.approx(17.8)
    assert analysis['slowdown_128']['staged_new_measured']['value']==pytest.approx(99/7.4)
    assert all(x==pytest.approx(7.4) for x in analysis['fits']['original']['leave_one_size_out_128_range'])


def test_uncertainty_preserves_matched_repetition_noise_between_methods():
    rows=[]
    for size in [32000,36000,40000,43000]:
        for repeat,noise in enumerate([-.13,.05,.21]):
            original=1+.05*size/1000+noise
            for method,multiplier in [('original',1),('gpu_new',2),('staged_new',3)]:
                rows.append(row(size,method,repeat,original*multiplier))
    analysis,_=analyze(rows,draw_count=100)
    np.testing.assert_allclose(analysis['slowdown_128']['gpu_new_predicted']['bootstrap_95'],[2,2],atol=1e-10)


def test_timed_total_includes_transfer_phases_and_excludes_loading():
    result=dict(phases=[dict(phase='load model and complete optimizer',seconds=1000),
                        dict(phase='offload inactive optimizer state',seconds=2),
                        dict(phase='hypernetwork forward',seconds=3),
                        dict(phase='restore complete LM gradients',seconds=1)])
    assert seconds(result)==6
    assert seconds(result,hn=True)==3
