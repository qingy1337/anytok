"""Chunked native normalization must retain the full original backward."""
import copy
from pathlib import Path
import sys

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'runtime/anytok_pool_20k'))
from hf_hypernet.identity_warmup import build_input_hypernet,freeze_rescalers
from local_memory_variants import apply_variant
from lossless_layer_norm import ChunkedLayerNorm,apply_chunked_norm,apply_chunked_source

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='Native CUDA LayerNorm')


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('rows',[40,43000])
def test_chunked_norm_outputs_input_and_affine_gradients_are_bitwise_exact(dtype,rows):
    torch.manual_seed(32)
    x=torch.randn(rows,2048,device='cuda',dtype=dtype)
    w=torch.randn(2048,device='cuda',requires_grad=True);b=torch.randn_like(w,requires_grad=True);g=torch.randn_like(x)
    records=[]
    for candidate in [False,True]:
        a=x.detach().clone().requires_grad_();w.grad=None;b.grad=None
        y=ChunkedLayerNorm.apply(a,w,b,1e-12,4096) if candidate else torch.nn.functional.layer_norm(a.float(),(2048,),w,b,eps=1e-12).to(a.dtype)
        y.backward(g);records.append((y.detach(),a.grad,w.grad,b.grad))
    for expected,actual in zip(*records):assert torch.equal(expected,actual)


@pytest.mark.parametrize('deterministic',[True,False])
@pytest.mark.parametrize('autocast',[False,True])
@pytest.mark.parametrize('chunk_source',[False,True])
@pytest.mark.parametrize('stage_output_norm',[False,True])
def test_full_global_model_preserves_outputs_gradients_and_rng(deterministic,autocast,chunk_source,stage_output_norm):
    torch.set_num_threads(1);torch.manual_seed(67)
    original=build_input_hypernet(96,64,hidden_width=64,intermediate_width=128,layers=2,attention_heads=8).cuda()
    freeze_rescalers(original);original.model.activation_checkpointing=True;original.activation_checkpoint_input=True
    modified=copy.deepcopy(original);apply_variant(modified,'split_norm_offload');apply_chunked_norm(modified,31,stage_output_norm=stage_output_norm)
    if chunk_source:apply_chunked_source(modified,8)
    forms=torch.randint(0,97,(40,10),device='cuda');priors=torch.randn(40,device='cuda');source=torch.randn(96,64,device='cuda')
    records=[]
    for model in [original,modified]:
        torch.cuda.manual_seed(814)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=autocast):out,_,_=model(forms,priors,source,None,deterministic=deterministic)
        out.float().square().sum().backward()
        records.append((out.detach(),{n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None},torch.cuda.get_rng_state()))
    assert torch.equal(records[0][0],records[1][0]);assert torch.equal(records[0][2],records[1][2])
    assert records[0][1].keys()==records[1][1].keys()
    for name in records[0][1]:torch.testing.assert_close(records[0][1][name],records[1][1][name],rtol=1e-6,atol=1e-6,msg=name)
