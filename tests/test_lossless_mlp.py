"""Native full-shape parameter reductions must survive private-buffer reuse."""
import copy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from lossless_mlp import RecomputedMLP

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='Native CUDA mixed precision')


@pytest.mark.parametrize('autocast',[False,True])
@pytest.mark.parametrize('width,batch',[(64,40),(2048,4096)])
def test_mlp_all_outputs_input_and_parameter_gradients_match_bitwise(autocast,width,batch):
    torch.manual_seed(24)
    dtype=torch.bfloat16 if autocast else torch.float32
    x=torch.randn(batch,11,width,device='cuda',dtype=dtype)
    original=torch.nn.Sequential(torch.nn.Linear(width,width*2),torch.nn.GELU(),torch.nn.Linear(width*2,width)).cuda()
    modified=copy.deepcopy(original);gradient=torch.randn_like(x);records=[]
    for candidate,model in [(False,original),(True,modified)]:
        values=x.detach().clone().requires_grad_()
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=autocast):
            output=RecomputedMLP.apply(values,model[0].weight,model[0].bias,model[2].weight,model[2].bias) if candidate else model(values)
        output.backward(gradient)
        records.append((output.detach(),values.grad,*[p.grad for p in model.parameters()]))
    for expected,actual in zip(*records):assert torch.equal(expected,actual)
