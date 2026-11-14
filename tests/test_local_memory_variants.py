"""Checkpoint changes must preserve the original operator and dropout semantics."""
import copy
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'runtime/anytok_pool_20k'))
from hf_hypernet.identity_warmup import build_input_hypernet, freeze_rescalers
from local_memory_variants import apply_variant, BoundedOffload


@pytest.mark.parametrize('deterministic', [True, False])
def test_fine_checkpoints_preserve_full_vocabulary_outputs_gradients_and_rng(deterministic):
    torch.set_num_threads(1);torch.manual_seed(67)
    original = build_input_hypernet(48, 16, hidden_width=16, intermediate_width=32,
                                   layers=2, attention_heads=2)
    freeze_rescalers(original);modified = copy.deepcopy(original)
    apply_variant(modified, 'fine')
    assert list(original.state_dict()) == list(modified.state_dict())
    forms = torch.randint(0, 49, (23, 10));priors = torch.randn(23);source = torch.randn(48, 16)
    records = []
    for model in [original, modified]:
        torch.manual_seed(814)
        out, _, _ = model(forms, priors, source, None, deterministic=deterministic)
        out.square().sum().backward()
        records.append((out.detach(), {n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}, torch.get_rng_state()))
    assert torch.equal(records[0][0], records[1][0])
    assert torch.equal(records[0][2], records[1][2])
    assert records[0][1].keys() == records[1][1].keys()
    for name in records[0][1]:
        torch.testing.assert_close(records[0][1][name], records[1][1][name], rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Activation offload requires CUDA')
@pytest.mark.parametrize('transpose', [False, True])
def test_offload_preserves_every_gradient_element_and_respects_host_budget(transpose):
    torch.manual_seed(124)
    values = torch.randn(1024,1024,device='cuda')
    expected = values.cos()
    parameter = torch.nn.Parameter(torch.ones(1024,1024,device='cuda'))
    ledger = BoundedOffload([parameter], maximum_bytes=16*2**20)
    assert ledger.pack(parameter) is parameter
    actual = values.clone().requires_grad_(True)
    with ledger.context(True):
        view = actual.T if transpose else actual
        loss = view.sin().sum()
    loss.backward()
    assert torch.equal(actual.grad,expected)
    assert ledger.copies > 0 and ledger.peak <= ledger.maximum_bytes
    limited = BoundedOffload([], maximum_bytes=1)
    assert limited.pack(values) is values
    assert limited.skipped_for_budget == 1 and limited.copies == 0
