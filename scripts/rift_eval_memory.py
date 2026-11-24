"""Release attention temporaries as soon as the inference pass finishes with them."""

from types import MethodType

import torch

from pool_training_memory import apply_gpu_memory_fix


def inference_attention(self, hidden, keys=None, mask=None, bias=None, head_mask=None, deterministic=True):
    if not deterministic or torch.is_grad_enabled():
        raise ValueError('The Rift attention path requires deterministic inference')
    keys = hidden if keys is None else keys
    batch, query_length, width = hidden.shape
    key_length = keys.shape[1]
    query = self.query(hidden).view(batch, query_length, self.n_heads, self.head_width).transpose(1, 2)
    key = self.key(keys).view(batch, key_length, self.n_heads, self.head_width).transpose(1, 2)
    scores = torch.matmul(query, key.transpose(-2, -1))
    del query, key
    scores = scores / (self.head_width ** .5)
    if mask is not None:
        scores = scores.masked_fill(~mask[:, None, None, :].bool(), torch.finfo(scores.dtype).min)
    if bias is not None:
        scores = scores + bias.to(scores.dtype)[None, None, None, :]
    weights = torch.softmax(scores, dim=-1)
    del scores
    if head_mask is not None:
        weights = weights * head_mask[None, :, None, None]
    value = self.value(keys).view(batch, key_length, self.n_heads, self.head_width).transpose(1, 2)
    attended = torch.matmul(weights, value).transpose(1, 2).reshape(batch, query_length, width)
    del weights, value
    return self.LayerNorm(hidden + self.dense(attended)).to(hidden.dtype)


def apply_rift_memory_fix(hypernetwork):
    apply_gpu_memory_fix(hypernetwork)
    for layer in hypernetwork.model.layers:
        layer.forward = MethodType(inference_layer, layer)
        for attention in (layer.attention, layer.inter_token_attention_down, layer.inter_token_attention_up):
            attention.forward = MethodType(inference_attention, attention)


def inference_layer(self, hidden, mask, priors, latents, lang_index=None, head_mask=None, deterministic=True):
    if self.adapter_width or not deterministic or not self.inter_token_attention or torch.is_grad_enabled():
        raise ValueError('Expected deterministic inference with the latent-attention hypernetwork')
    attended = self.attention(hidden, mask=mask, head_mask=head_mask, deterministic=deterministic)
    bias = priors * self.prior_bias_scale if self.prior_bias else None
    blocks = self.inter_token_attention_down(latents[None], keys=attended[None, :, 0], bias=bias,
                                             deterministic=deterministic)[0]
    attended = self.inter_token_attention_up(attended.reshape(1, -1, attended.shape[-1]),
                                            keys=blocks[None], deterministic=deterministic).reshape_as(attended)
    intermediate = self.intermediate(attended)
    # This private buffer has no reader that needs the pre-GELU values.
    torch.ops.aten.gelu.out(intermediate, approximate='none', out=intermediate)
    dense = self.output_dense(intermediate)
    del intermediate
    return self.output_norm(attended + dense).to(attended.dtype)
