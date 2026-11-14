"""Paired text fidelity, loss arithmetic and isolation of the canonical forward."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models, decoders
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('paired_reward', ROOT / 'runtime/anytok_pool_regret_rl_20k/hf_hypernet/paired_reward.py')
module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def tokenizers():
    vocab = {'<eos>': 0, 'a': 1, 'b': 2, 'c': 3, ' ': 4, 'ab': 5, 'é': 6}
    native = Tokenizer(models.BPE(vocab=vocab, merges=[]));native.decoder = decoders.Fuse()
    canonical = Tokenizer(models.BPE(vocab=vocab, merges=[('a', 'b')]))
    canonical.decoder = decoders.Fuse()
    return native, PreTrainedTokenizerFast(tokenizer_object=canonical, bos_token='<eos>', eos_token='<eos>')


@pytest.mark.parametrize('text', ['ababab', 'ab c ab c ' * 70, 'é ab cé ab ' * 60])
def test_paired_boundaries_are_reversible_and_share_targets_and_bytes(text):
    native, canonical = tokenizers()
    result = module.make_paired_prefix_example(text, native, canonical, min_chars=2)
    assert result is not None
    a, b, proof = result
    assert a[1:] == b[1:]
    prefix = native.decode(a[0]);suffix = canonical.decode(a[1], clean_up_tokenization_spaces=False)
    assert canonical.decode(b[0], clean_up_tokenization_spaces=False) == prefix
    assert prefix + suffix == text[:proof['window_chars']]
    assert canonical(prefix + suffix, add_special_tokens=False).input_ids == b[0] + b[1]
    assert a[2] == len(suffix.encode())
    assert 1 + max(len(a[0]), len(b[0])) + len(a[1]) <= 128
    assert proof['canonical_joint_encoding_matches']


def test_old_midpoint_merge_is_corrected_without_changing_a_valid_window():
    native, canonical = tokenizers()
    a, b, proof = module.make_paired_prefix_example('ababab', native, canonical, min_chars=2)
    assert proof['cut_chars'] in [2, 4]  # Character midpoint 3 splits an 'ab' token.
    assert proof['window_chars'] == 6
    assert module.make_paired_prefix_example('ab', native, canonical, min_chars=32) is None
    assert module.make_paired_prefix_example('ababab', native, canonical, min_chars=2, max_positions=2) is None


def test_regret_arithmetic_allows_negative_rewards_and_rejects_bad_losses():
    raw, canonical, reward = module.paired_gap_bpb(3., 4., 10)
    assert reward < 0 and reward == pytest.approx(raw - canonical)
    for args in [(1., 2., 0), (float('nan'), 2., 10), (-1., 2., 10)]:
        with pytest.raises(ValueError):module.paired_gap_bpb(*args)


def test_canonical_forward_cannot_accumulate_gradients_or_change_rng_or_mode():
    class LM(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight = torch.nn.Parameter(torch.tensor(2.))
        def forward(self, **kwargs):
            assert not torch.is_grad_enabled()
            torch.rand(4)
            return SimpleNamespace(loss=self.weight * 1.5)
    model = LM();model.train();model.weight.grad = torch.tensor(7.)
    source = torch.zeros(8, 3)
    examples = [([1], [2, 3], 2), ([1], [2], 1)]
    def microbatch(*args):return (None, None, None, None)
    before = torch.get_rng_state().clone()
    nats = module.canonical_loss_nats(model, examples, source, 1, microbatch)
    assert nats == 9.
    assert model.training and model.weight.grad.item() == 7.
    assert torch.equal(before, torch.get_rng_state())


def test_regret_configuration_cannot_overwrite_original_outputs():
    command = [sys.executable, 'scripts/run_anytok_pool_rl_20k.py']
    plans = {name: json.loads(subprocess.check_output(command, cwd=ROOT,
        env={**os.environ, 'ANYTOK_RL_EXPERIMENT': name}, text=True)) for name in ['original', 'regret']}
    assert plans['original']['policy']['entropy_weight'] == .01
    assert plans['regret']['policy']['entropy_weight'] == 0.
    assert plans['original']['output_prefix'] != plans['regret']['output_prefix']
    assert plans['regret']['steps'] == 20000 and plans['regret']['reward'] == 'paired_gap_bpb'
    assert not set(plans['regret']['profiles']) & {'general', 'research-everettcc'}
