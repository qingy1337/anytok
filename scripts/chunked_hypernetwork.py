"""Isolated full-vocabulary chunking experiment, preserving native dropout masks.

This keeps complete layer states and the original vocabulary-wide latent
attention. Only per-entry work is chunked. GPU storage still has a small linear
component; this is not a claim of constant total memory or a production patch.
"""
from types import MethodType, SimpleNamespace

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def recompute(function, *args):
    if not torch.is_grad_enabled():
        return function(*args)
    return checkpoint(function, *args, use_reentrant=False, preserve_rng_state=True)


def pack_mask(mask):
    flat = mask.reshape(-1)
    if flat.numel() % 8:
        flat = F.pad(flat, (0, 8 - flat.numel() % 8))
    packed = torch.zeros(flat.numel() // 8, dtype=torch.uint8, device=flat.device)
    for bit in range(8):
        packed.bitwise_or_(flat[bit::8].to(torch.uint8).bitwise_left_shift(bit))
    return packed


def unpack_mask(packed, shape):
    shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
    return ((packed.reshape(-1, 1) >> shifts) & 1).reshape(-1)[:__import__('math').prod(shape)].reshape(shape).bool()


def native_packed_mask(shape, dtype, device, p, deterministic):
    if deterministic or p == 0:
        return torch.empty(0, dtype=torch.uint8, device=device)
    # The same full-shape native CUDA dropout kernel preserves the original
    # Philox mapping and RNG advance. Its temporary output is immediately freed.
    with torch.no_grad():
        dummy = torch.empty(shape, dtype=dtype, device=device)
        output, mask = torch.ops.aten.native_dropout(dummy, p, True)
        del output, dummy
        return pack_mask(mask)


def masked_dropout(values, packed, p):
    if packed.numel() == 0:
        return values
    mask = unpack_mask(packed, values.shape)
    # Op-math is FP32 in native CUDA dropout, including BF16 inputs.
    return (values.float() * mask * (1.0 / (1.0 - p))).to(values.dtype)


def _slice_rows(packed, start, stop, row_elements):
    if packed.numel() == 0:
        return packed
    if row_elements % 8:
        raise ValueError('Row-aligned mask slicing requires a multiple of eight')
    return packed[start * (row_elements // 8):stop * (row_elements // 8)]


def local_attention(module, hidden, mask, weights_mask, residual_mask):
    batch, length, width = hidden.shape
    query = module.query(hidden).view(batch, length, module.n_heads, module.head_width).transpose(1, 2)
    key = module.key(hidden).view(batch, length, module.n_heads, module.head_width).transpose(1, 2)
    value = module.value(hidden).view(batch, length, module.n_heads, module.head_width).transpose(1, 2)
    scores = torch.matmul(query, key.transpose(-2, -1)) / module.head_width**.5
    scores = scores.masked_fill(~mask[:, None, None, :].bool(), torch.finfo(scores.dtype).min)
    weights = masked_dropout(torch.softmax(scores, dim=-1), weights_mask, module.attention_dropout.p)
    attended = torch.matmul(weights, value).transpose(1, 2).reshape(batch, length, width)
    dense = module.dense(attended)
    return module.LayerNorm(hidden + masked_dropout(dense, residual_mask, module.dropout.p)).to(hidden.dtype)


def upward_attention(module, hidden, blocks, weights_mask, residual_mask):
    batch, length, width = hidden.shape
    flat = hidden.reshape(1, -1, width)
    query = module.query(flat).view(1, batch * length, module.n_heads, module.head_width).transpose(1, 2)
    key = module.key(blocks[None]).view(1, blocks.shape[0], module.n_heads, module.head_width).transpose(1, 2)
    value = module.value(blocks[None]).view(1, blocks.shape[0], module.n_heads, module.head_width).transpose(1, 2)
    scores = torch.matmul(query, key.transpose(-2, -1)) / module.head_width**.5
    weights = masked_dropout(torch.softmax(scores, dim=-1), weights_mask, module.attention_dropout.p)
    result = torch.matmul(weights, value).transpose(1, 2).reshape(1, batch * length, width)
    residual = flat + masked_dropout(module.dense(result), residual_mask, module.dropout.p)
    return module.LayerNorm(residual).to(hidden.dtype).reshape_as(hidden)


def chunked_layer(layer, hidden, mask, priors, latents, chunk, deterministic):
    if layer.adapter_width or not layer.inter_token_attention:
        raise ValueError('This diagnostic supports the actual adapter-free latent-attention model')
    count, length, width = hidden.shape
    attention = layer.attention
    # Native RNG order: local attention weights, local residual, down attention,
    # up attention weights, up residual, MLP residual. Each mask is bit packed.
    weights_dtype = torch.float32 if torch.is_autocast_enabled('cuda') else hidden.dtype
    weight_masks = native_packed_mask((count, attention.n_heads, length, length), weights_dtype,
                                      hidden.device, attention.attention_dropout.p, deterministic)
    residual_masks = native_packed_mask(hidden.shape, hidden.dtype, hidden.device, attention.dropout.p, deterministic)
    pieces = []
    for start in range(0, count, chunk):
        end = min(start + chunk, count)
        weights = _slice_rows(weight_masks, start, end, attention.n_heads * length * length)
        residual = _slice_rows(residual_masks, start, end, length * width)
        pieces.append(recompute(lambda h, m, w, r: local_attention(attention, h, m, w, r),
                                hidden[start:end], mask[start:end], weights, residual))
    attended = torch.cat(pieces, dim=0)
    del pieces, weight_masks, residual_masks
    bias = priors * layer.prior_bias_scale if layer.prior_bias else None
    # Deliberately unchanged: all vocabulary entries participate in this single
    # global softmax. No per-chunk summaries, sampling or approximations.
    blocks = recompute(lambda a, l: layer.inter_token_attention_down(
        l[None], keys=a[None, :, 0], bias=bias, deterministic=deterministic)[0], attended, latents)
    up = layer.inter_token_attention_up
    weights = native_packed_mask((1, up.n_heads, count * length, blocks.shape[0]), weights_dtype,
                                 hidden.device, up.attention_dropout.p, deterministic)
    up_residual = native_packed_mask((1, count * length, width), hidden.dtype,
                                     hidden.device, up.dropout.p, deterministic)
    mlp_residual = native_packed_mask(hidden.shape, hidden.dtype, hidden.device, layer.output_dropout.p, deterministic)
    pieces = []
    for start in range(0, count, chunk):
        end = min(start + chunk, count)
        if weights.numel():
            # Packed rows follow [head, vocabulary * pieces, latent].
            w = weights.reshape(up.n_heads, count * length, blocks.shape[0] // 8)[:, start*length:end*length].contiguous().reshape(-1)
        else:
            w = weights
        r = _slice_rows(up_residual, start, end, length * width)
        m = _slice_rows(mlp_residual, start, end, length * width)
        def finish(h, b, wm, rm, mm):
            a = upward_attention(up, h, b, wm, rm)
            dense = layer.output_dense(layer.act(layer.intermediate(a)))
            return layer.output_norm(a + masked_dropout(dense, mm, layer.output_dropout.p)).to(a.dtype)
        pieces.append(recompute(finish, attended[start:end], blocks, w, r, m))
    return torch.cat(pieces, dim=0)


def apply_chunked(hn, chunk=2048):
    if chunk <= 0 or chunk % 8:
        raise ValueError('Chunk size must be a positive multiple of eight')
    if hn.config.hn_embed_lang_id or hn.config.hn_language_adapter_bottleneck_dim:
        raise ValueError('Use the actual adapter-free, language-free training architecture')
    hn.activation_checkpoint_input = False
    original_projection = hn._project_source_inputs
    def project(self, forms, source):
        return torch.cat([recompute(original_projection, forms[start:start+chunk], source)
                          for start in range(0, len(forms), chunk)], dim=0)
    hn._project_source_inputs = MethodType(project, hn)
    model = hn.model
    def model_forward(self, input_ids, inputs_embeds=None, attention_mask=None, position_ids=None,
                      token_type_ids=None, target_priors=None, lang_index=None, head_mask=None, deterministic=True):
        if inputs_embeds is None or head_mask is not None:
            raise ValueError('Only the original source-embedding training path is supported')
        count = len(input_ids)
        mask = F.pad(attention_mask, (0, 1), value=1)
        full_shape = (count, inputs_embeds.shape[1] + 1, inputs_embeds.shape[-1])
        drops = native_packed_mask(full_shape, inputs_embeds.dtype, inputs_embeds.device, self.dropout.p, deterministic)
        pieces = []
        def initialize(x, positions, priors, dropout_mask):
            hidden = x + self.position_embeddings(positions).to(x.dtype) + self.token_type_embeddings(torch.zeros_like(positions)).to(x.dtype)
            hidden = torch.cat((hidden, self.target_priors_projection(priors[:, None, None].to(hidden.dtype))), dim=1)
            return masked_dropout(self.LayerNorm(hidden).to(hidden.dtype), dropout_mask, self.dropout.p)
        for start in range(0, count, chunk):
            end = min(start+chunk, count)
            pieces.append(recompute(initialize, inputs_embeds[start:end], position_ids[start:end], target_priors[start:end],
                                    _slice_rows(drops, start, end, full_shape[1]*full_shape[2])))
        hidden = torch.cat(pieces, dim=0)
        del pieces, drops
        for layer in self.layers:
            def forward_layer(h, m, p, l, current=layer):
                return chunked_layer(current, h, m, p, l, chunk, deterministic)
            hidden = recompute(forward_layer, hidden, mask, target_priors, self.inter_token_blocks)
        return SimpleNamespace(last_hidden_state=hidden)
    model.forward = MethodType(model_forward, model)
    original_output = hn.output_projection.forward
    def output_projection(values):
        return torch.cat([recompute(original_output, values[start:start+chunk])
                          for start in range(0, len(values), chunk)], dim=0)
    hn.output_projection.forward = output_projection
