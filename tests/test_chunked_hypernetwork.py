"""Full-vocabulary chunking must preserve global attention and native dropout."""
import copy
from pathlib import Path
import sys

import pytest
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'runtime/anytok_pool_20k'))
from hf_hypernet.identity_warmup import build_input_hypernet, freeze_rescalers
from chunked_hypernetwork import native_packed_mask, masked_dropout, apply_chunked

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='Native CUDA dropout semantics')


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_native_dropout_output_mask_and_rng_are_exact(dtype):
    x = torch.randn(40,11,64, device='cuda', dtype=dtype)
    torch.cuda.manual_seed(814)
    expected = F.dropout(x, .1, training=True); expected_rng = torch.cuda.get_rng_state()
    torch.cuda.manual_seed(814)
    mask = native_packed_mask(x.shape, dtype, x.device, .1, False)
    actual = masked_dropout(x, mask, .1)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.cuda.get_rng_state(), expected_rng)


@pytest.mark.parametrize('deterministic', [True, False])
@pytest.mark.parametrize('autocast', [False, True])
@pytest.mark.xfail(strict=True, reason='Rejected whole-entry chunking changes full-model outputs or gradients')
def test_chunked_full_global_model_outputs_gradients_and_rng(deterministic, autocast):
    torch.set_num_threads(1);torch.manual_seed(67)
    original = build_input_hypernet(96, 64, hidden_width=64, intermediate_width=128,
                                    layers=2, attention_heads=8).cuda()
    freeze_rescalers(original)
    original.model.activation_checkpointing = True
    original.activation_checkpoint_input = True
    modified = copy.deepcopy(original)
    apply_chunked(modified, chunk=8)
    assert list(original.state_dict()) == list(modified.state_dict())
    forms = torch.randint(0, 97, (40,10), device='cuda')
    priors = torch.randn(40, device='cuda');source = torch.randn(96,64, device='cuda')
    records = []
    for model in [original, modified]:
        torch.cuda.manual_seed(814)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
            out, _, _ = model(forms, priors, source, None, deterministic=deterministic)
        out.float().square().sum().backward()
        records.append((out.detach(), {n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}, torch.cuda.get_rng_state()))
    assert torch.equal(records[0][2], records[1][2])
    assert torch.equal(records[0][0], records[1][0]), float((records[0][0]-records[1][0]).abs().max())
    assert records[0][1].keys() == records[1][1].keys()
    for name in records[0][1]:
        torch.testing.assert_close(records[0][1][name], records[1][1][name], rtol=1e-5, atol=1e-5, msg=name)
