"""Native full-shape MLP math with explicit activation lifetime management."""
import torch
from torch.nn import functional as F


class RecomputedMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, weight1, bias1, weight2, bias2):
        hidden=F.linear(values,weight1,bias1)
        activated=F.gelu(hidden)
        compute_dtype=hidden.dtype
        del hidden
        result=F.linear(activated,weight2,bias2)
        ctx.save_for_backward(values,weight1,bias1,weight2,bias2)
        ctx.compute_dtype=compute_dtype
        return result

    @staticmethod
    def backward(ctx, gradient):
        values,weight1,bias1,weight2,bias2=ctx.saved_tensors
        dtype=ctx.compute_dtype
        with torch.autocast('cuda',enabled=False):
            x=values.to(dtype);w1=weight1.to(dtype);w2=weight2.to(dtype)
            z=F.linear(x,w1,bias1.to(dtype));activated=F.gelu(z)
            x2=x.reshape(-1,x.shape[-1]);g2=gradient.reshape(-1,gradient.shape[-1]).to(dtype)
            activated2=activated.reshape(-1,activated.shape[-1])
            dw2=g2.T.mm(activated2).to(weight2.dtype)
            db2=g2.sum(0).to(bias2.dtype)
            del activated2,activated
            da=g2.mm(w2)
            # da is newly allocated, private to this custom backward. Reuse it
            # only after all consumers of its pre-GELU-gradient values finish.
            torch.ops.aten.gelu_backward.grad_input(da,z.reshape_as(da),approximate='none',grad_input=da)
            del z
            dw1=da.T.mm(x2).to(weight1.dtype)
            db1=da.sum(0).to(bias1.dtype)
            dx=da.mm(w1).reshape_as(values).to(values.dtype)
        return dx,dw1,db1,dw2,db2


def mlp(values,layer):
    return RecomputedMLP.apply(values,layer.intermediate.weight,layer.intermediate.bias,
                               layer.output_dense.weight,layer.output_dense.bias)
