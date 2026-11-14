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
from local_memory_variants import apply_variant, BoundedOffload, DiskTensor, move_optimizer_state


@pytest.mark.parametrize('deterministic', [True, False])
@pytest.mark.parametrize('variant', ['fine','split_norm_offload'])
def test_fine_checkpoints_preserve_full_vocabulary_outputs_gradients_and_rng(deterministic,variant):
    torch.set_num_threads(1);torch.manual_seed(67)
    original = build_input_hypernet(48, 16, hidden_width=16, intermediate_width=32,
                                   layers=2, attention_heads=2)
    freeze_rescalers(original);modified = copy.deepcopy(original)
    apply_variant(modified, variant)
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
@pytest.mark.parametrize('layout', ['contiguous', 'transpose', 'slice'])
def test_offload_preserves_every_gradient_element_and_respects_host_budget(layout):
    torch.manual_seed(124)
    values = torch.randn(2048 if layout=='slice' else 1024,2048 if layout=='slice' else 1024,device='cuda')
    expected = values.cos()
    if layout=='slice':
        expected.zero_();expected[1::2,1::2] = values[1::2,1::2].cos()
    parameter = torch.nn.Parameter(torch.ones(1024,1024,device='cuda'))
    ledger = BoundedOffload([parameter], maximum_bytes=16*2**20)
    assert ledger.pack(parameter) is parameter
    actual = values.clone().requires_grad_(True)
    with ledger.context(True):
        view = actual.T if layout=='transpose' else (actual[1::2,1::2] if layout=='slice' else actual)
        loss = view.sin().sum()
    loss.backward()
    assert torch.equal(actual.grad,expected)
    assert ledger.copies > 0 and ledger.peak <= ledger.maximum_bytes
    packed = ledger.pack(view.detach())
    assert tuple(ledger.unpack(packed).stride()) == tuple(view.stride())
    assert ledger.unpack(packed).storage_offset() == view.storage_offset()
    limited = BoundedOffload([], maximum_bytes=1)
    assert limited.pack(values) is values
    assert limited.skipped_for_budget == 1 and limited.copies == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Activation offload requires CUDA')
def test_disk_spill_preserves_values_layout_gradient_and_releases_temporary_files(tmp_path):
    torch.manual_seed(129)
    values=torch.randn(2048,2048,device='cuda');actual=values.clone().requires_grad_(True)
    expected=torch.zeros_like(values);expected[1::2,1::2]=values[1::2,1::2].cos()
    ledger=BoundedOffload([],maximum_bytes=8*2**20,spill_directory=tmp_path)
    ledger.cpu_capacity=0
    with ledger.context(True):loss=actual[1::2,1::2].sin().sum()
    assert ledger.files_spilled>0 and ledger.disk_peak>0
    loss.backward()
    assert torch.equal(actual.grad,expected)
    assert not list(tmp_path.glob('activation-*'))
    assert ledger.disk_live==0 and ledger.peak<=ledger.maximum_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Fused CUDA Adam state transfer')
def test_inactive_optimizer_transfer_preserves_fused_adam_update_exactly():
    torch.manual_seed(249)
    a=torch.nn.Parameter(torch.randn(32,32,device='cuda'));b=torch.nn.Parameter(a.detach().clone())
    left=torch.optim.AdamW([a],lr=.001,betas=(.9,.95),fused=True)
    right=torch.optim.AdamW([b],lr=.001,betas=(.9,.95),fused=True)
    for parameter,opt in [(a,left),(b,right)]:
        parameter.square().sum().backward();opt.step();opt.zero_grad(set_to_none=True)
    assert move_optimizer_state(right,'cpu')>0
    a.sin().sum().backward();b.sin().sum().backward()
    assert torch.equal(a.grad,b.grad)
    move_optimizer_state(right,'cuda');left.step();right.step()
    assert torch.equal(a,b)
    for key in left.state[a]:assert torch.equal(left.state[a][key],right.state[b][key])
