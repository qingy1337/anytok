"""Chunk LayerNorm forward; retain the original full native backward reduction.

BF16 inputs are exactly representable in FP32. Save those original inputs,
not their enlarged FP32 copies. No weight-gradient reduction is split into
chunks. This is diagnostic code and never imported by the live trainer.
"""
import math
from types import MethodType

import torch


class ChunkedLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, weight, bias, eps, chunk_rows):
        if values.device.type != 'cuda' or values.ndim < 2:
            raise ValueError('Diagnostic LayerNorm requires CUDA and at least two dimensions')
        width = values.shape[-1]
        if values.dtype not in [torch.float32,torch.bfloat16] or weight.dtype != torch.float32:
            raise ValueError('Use the original FP32/BF16 model configuration')
        flat = values.reshape(-1, width)
        output = torch.empty_like(values, memory_format=torch.contiguous_format)
        out = output.view(-1, width)
        mean = torch.empty((len(flat),1), dtype=torch.float32, device=values.device)
        rstd = torch.empty_like(mean)
        for start in range(0,len(flat),chunk_rows):
            end = min(start+chunk_rows,len(flat))
            y,m,r = torch.ops.aten.native_layer_norm(flat[start:end].float(),[width],weight,bias,eps)
            out[start:end].copy_(y)
            mean[start:end].copy_(m);rstd[start:end].copy_(r)
        ctx.save_for_backward(values,weight,bias,mean.reshape(*values.shape[:-1],1),rstd.reshape(*values.shape[:-1],1))
        return output

    @staticmethod
    def backward(ctx, gradient):
        values,weight,bias,mean,rstd = ctx.saved_tensors
        # Preserve the original full-shape native CUDA reduction for the
        # trainable scale and bias. Ordinary chunked autograd changes this.
        with torch.autocast('cuda',enabled=False):
            dx,dw,db = torch.ops.aten.native_layer_norm_backward(
                gradient.float().contiguous(),values.float().contiguous(),[values.shape[-1]],
                mean,rstd,weight,bias,list(ctx.needs_input_grad[:3]))
        return dx.to(values.dtype) if dx is not None else None,dw,db,None,None


def apply_chunked_norm(hypernetwork, chunk_rows=16384):
    if chunk_rows < 1: raise ValueError('Positive row chunk required')
    for module in hypernetwork.modules():
        if isinstance(module,torch.nn.LayerNorm):
            if len(module.normalized_shape)!=1 or module.weight is None or module.bias is None:
                raise ValueError('Only the original affine vector LayerNorm is supported')
            def forward(self, values):
                with torch.autocast('cuda',enabled=False):
                    return ChunkedLayerNorm.apply(values,self.weight,self.bias,self.eps,chunk_rows)
            module.forward=MethodType(forward,module)
