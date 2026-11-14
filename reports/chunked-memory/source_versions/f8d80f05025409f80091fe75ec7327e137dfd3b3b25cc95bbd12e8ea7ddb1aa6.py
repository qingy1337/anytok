"""Chunk LayerNorm forward; retain the original full native backward reduction.

BF16 inputs are exactly representable in FP32. Save those original inputs,
not their enlarged FP32 copies. No weight-gradient reduction is split into
chunks. This is diagnostic code and never imported by the live trainer.
"""
from types import MethodType

import torch
from torch.nn import functional as F


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
        ctx.chunk_rows=chunk_rows
        return output

    @staticmethod
    def backward(ctx, gradient):
        values,weight,bias,mean,rstd = ctx.saved_tensors
        # Preserve the full-shape native scale/bias reductions. Compute input
        # gradients separately in independent rows, avoiding a full FP32 dx.
        with torch.autocast('cuda',enabled=False):
            width=values.shape[-1]
            dw=db=None
            if any(ctx.needs_input_grad[1:3]):
                _,dw,db = torch.ops.aten.native_layer_norm_backward(
                    gradient.float().contiguous(),values.float().contiguous(),[width],
                    mean,rstd,weight,bias,[False,*ctx.needs_input_grad[1:3]])
            dx=None
            if ctx.needs_input_grad[0]:
                dx=torch.empty_like(values,memory_format=torch.contiguous_format)
                xs=values.reshape(-1,width);gs=gradient.reshape(-1,width);out=dx.view(-1,width)
                means=mean.reshape(-1,1);rstds=rstd.reshape(-1,1)
                for start in range(0,len(xs),ctx.chunk_rows):
                    end=min(start+ctx.chunk_rows,len(xs))
                    gx,_,_=torch.ops.aten.native_layer_norm_backward(
                        gs[start:end].float().contiguous(),xs[start:end].float().contiguous(),[width],
                        means[start:end],rstds[start:end],weight,bias,[True,False,False])
                    out[start:end].copy_(gx)
        return dx,dw,db,None,None


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


class SourceEmbeddingInputs(torch.autograd.Function):
    """Construct the exact Linear input in chunks; use one native scatter backward."""
    @staticmethod
    def forward(ctx, forms, source, fallback_weight, scale, offset, original_size, dtype, chunk_rows):
        if source.requires_grad or scale.requires_grad or offset.requires_grad:
            raise ValueError('The original source embeddings and rescalers must be frozen')
        result=torch.empty((*forms.shape,source.shape[1]),device=source.device,dtype=dtype)
        for start in range(0,len(forms),chunk_rows):
            ids=forms[start:start+chunk_rows]
            use_fallback=ids>=original_size
            main=F.embedding(ids.clamp(max=original_size-1),source)
            main=scale*main+offset
            fb=F.embedding((ids-original_size).clamp(min=0),fallback_weight)
            result[start:start+chunk_rows].copy_(torch.where(use_fallback[...,None],fb,main))
        ctx.save_for_backward(forms)
        ctx.original_size=original_size;ctx.fallback_rows=len(fallback_weight)
        return result

    @staticmethod
    def backward(ctx, gradient):
        forms,=ctx.saved_tensors
        with torch.autocast('cuda',enabled=False):
            grad=torch.where((forms>=ctx.original_size)[...,None],gradient.float(),0.)
            ids=(forms-ctx.original_size).clamp(min=0)
            fallback_grad=torch.ops.aten.embedding_dense_backward(grad,ids,ctx.fallback_rows,-1,False)
        return None,None,fallback_grad,None,None,None,None,None


def apply_chunked_source(hn,chunk_rows=2048):
    if not hn.config.hn_rescale_embeddings:
        raise ValueError('This diagnostic expects the actual rescaled source configuration')
    from chunked_hypernetwork import recompute
    def project(self,forms,source):
        dtype=torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else source.dtype
        embedded=SourceEmbeddingInputs.apply(forms,source,self.fallback_embeddings.weight,
            self.in_scaler.w,self.in_scaler.b,self.config.original_vocab_size,dtype,chunk_rows)
        projected=recompute(self.input_projection[0],embedded)
        block=self.input_projection[1]
        intermediate=recompute(lambda x:F.gelu(block.dense1(x),approximate='tanh'),projected)
        output=recompute(lambda x:F.gelu(block.dense2(x),approximate='tanh'),intermediate)
        return recompute(lambda x:block.ln(x).to(dtype),output+projected)
    hn._project_source_inputs=MethodType(project,hn)
    hn.activation_checkpoint_input=True


def deep_checkpoint_layer(self, hidden, mask, priors, latents, lang_index=None, head_mask=None, deterministic=True):
    from chunked_hypernetwork import recompute
    if self.adapter_width:raise ValueError('Use the actual adapter-free model')
    attended=self.attention(hidden,mask=mask,head_mask=head_mask,deterministic=deterministic)
    bias=priors*self.prior_bias_scale if self.prior_bias else None
    blocks=self.inter_token_attention_down(latents[None],keys=attended[None,:,0],bias=bias,deterministic=deterministic)[0]
    attended=self.inter_token_attention_up(attended.reshape(1,-1,attended.shape[-1]),keys=blocks[None],deterministic=deterministic).reshape_as(attended)
    intermediate=recompute(self.intermediate,attended)
    activated=recompute(self.act,intermediate)
    dense=recompute(self.output_dense,activated)
    dtype=attended.dtype
    return recompute(lambda x:self.output_norm(x).to(dtype),attended+F.dropout(dense,p=self.output_dropout.p,training=not deterministic))


def native_mlp_layer(self, hidden, mask, priors, latents, lang_index=None, head_mask=None, deterministic=True):
    from chunked_hypernetwork import recompute
    from lossless_mlp import mlp
    if self.adapter_width:raise ValueError('Use the actual adapter-free model')
    attended=self.attention(hidden,mask=mask,head_mask=head_mask,deterministic=deterministic)
    bias=priors*self.prior_bias_scale if self.prior_bias else None
    blocks=self.inter_token_attention_down(latents[None],keys=attended[None,:,0],bias=bias,deterministic=deterministic)[0]
    attended=self.inter_token_attention_up(attended.reshape(1,-1,attended.shape[-1]),keys=blocks[None],deterministic=deterministic).reshape_as(attended)
    dense=mlp(attended,self)
    dtype=attended.dtype
    return recompute(lambda x:self.output_norm(x).to(dtype),attended+F.dropout(dense,p=self.output_dropout.p,training=not deterministic))
