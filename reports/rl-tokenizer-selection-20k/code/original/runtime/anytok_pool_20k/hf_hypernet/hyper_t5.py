"""PyTorch port of ZeTT's FlaxHyperT5EncoderModule."""

import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


class T5LayerNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        variance = x.float().square().mean(dim=-1, keepdim=True)
        normalized = x.float() * torch.rsqrt(variance + self.eps)
        return self.weight * normalized.to(x.dtype)


class T5Attention(nn.Module):
    def __init__(self, config, relative=False):
        super().__init__()
        self.n_heads = config.num_heads
        self.d_kv = config.d_kv
        self.inner_dim = self.n_heads * self.d_kv
        self.q = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.relative = relative
        self.n_buckets = config.relative_attention_num_buckets
        self.max_distance = config.relative_attention_max_distance
        if relative:
            self.relative_attention_bias = nn.Embedding(self.n_buckets, self.n_heads)

    def _bucket(self, relative_position):
        half = self.n_buckets // 2
        offset = (relative_position > 0).long() * half
        distance = relative_position.abs()
        max_exact = half // 2
        large = max_exact + (
            torch.log(distance.float().clamp(min=1) / max_exact)
            / math.log(self.max_distance / max_exact)
            * (half - max_exact)
        ).long()
        return offset + torch.where(distance < max_exact, distance, large.clamp(max=half - 1))

    def forward(self, hidden, keys=None, mask=None, position_bias=None, deterministic=True):
        if keys is None:
            keys = hidden
        batch, q_length, _ = hidden.shape
        k_length = keys.shape[1]
        def split(x):
            return x.view(batch, -1, self.n_heads, self.d_kv).transpose(1, 2)
        q = split(self.q(hidden))
        k = split(self.k(keys))
        v = split(self.v(keys))
        scores = torch.matmul(q, k.transpose(-2, -1))
        if position_bias is None:
            if self.relative:
                q_pos = torch.arange(q_length, device=hidden.device)[:, None]
                k_pos = torch.arange(k_length, device=hidden.device)[None, :]
                bucket = self._bucket(k_pos - q_pos)
                position_bias = self.relative_attention_bias(bucket).permute(2, 0, 1)[None]
            else:
                position_bias = torch.zeros((1, self.n_heads, q_length, k_length),
                                            dtype=scores.dtype, device=scores.device)
            if mask is not None:
                mask_bias = torch.zeros((batch, 1, 1, k_length), dtype=scores.dtype, device=scores.device)
                mask_bias = mask_bias.masked_fill(~mask[:, None, None, :].bool(), torch.finfo(scores.dtype).min)
                position_bias = position_bias + mask_bias
        scores = scores + position_bias.to(scores.dtype)
        weights = F.dropout(torch.softmax(scores, dim=-1), p=self.dropout.p, training=not deterministic)
        output = torch.matmul(weights, v).transpose(1, 2).reshape(batch, q_length, self.inner_dim)
        return self.o(output), position_bias


class T5FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = T5LayerNorm(config.d_model, config.layer_norm_epsilon)
        self.is_gated = config.is_gated_act
        self.activation = {"relu": F.relu, "gelu": F.gelu, "gelu_new": lambda x: F.gelu(x, approximate="tanh")}[config.dense_act_fn]
        if self.is_gated:
            self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
            self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        else:
            self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden, deterministic=True):
        x = self.layer_norm(hidden)
        if self.is_gated:
            x = self.activation(self.wi_0(x)) * self.wi_1(x)
        else:
            x = self.activation(self.wi(x))
        x = F.dropout(x, p=self.dropout.p, training=not deterministic)
        return hidden + F.dropout(self.wo(x), p=self.dropout.p, training=not deterministic)


class T5Block(nn.Module):
    def __init__(self, config, first):
        super().__init__()
        self.self_norm = T5LayerNorm(config.d_model, config.layer_norm_epsilon)
        self.self_attention = T5Attention(config, relative=first)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.feed_forward = T5FeedForward(config)
        self.inter_token_attention = config.add_inter_token_attention
        self.prior_bias = config.inter_token_attention_bias_by_priors
        self.prior_bias_scale = config.inter_token_attention_bias_scaler
        if self.inter_token_attention:
            self.inter_token_layer_norm1 = T5LayerNorm(config.d_model, config.layer_norm_epsilon)
            self.inter_token_layer_norm2 = T5LayerNorm(config.d_model, config.layer_norm_epsilon)
            self.inter_token_attention_down = T5Attention(config)
            self.inter_token_attention_up = T5Attention(config)

    def forward(self, hidden, mask, priors, latents, position_bias, deterministic=True):
        self_output, position_bias = self.self_attention(self.self_norm(hidden), mask=mask,
                                                          position_bias=position_bias, deterministic=deterministic)
        hidden = hidden + F.dropout(self_output, p=self.dropout.p, training=not deterministic)
        if self.inter_token_attention:
            normed = self.inter_token_layer_norm1(hidden)
            bias = priors[None] * self.prior_bias_scale if self.prior_bias else None
            down, _ = self.inter_token_attention_down(latents[None], keys=normed[None, :, 0],
                                                       position_bias=bias, deterministic=deterministic)
            blocks = self.inter_token_layer_norm2(latents + down[0])
            up, _ = self.inter_token_attention_up(normed[None, :, 0], keys=blocks[None],
                                                   deterministic=deterministic)
            hidden = hidden + up[0][:, None, :]
        return self.feed_forward(hidden, deterministic), position_bias


class HyperT5Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.target_priors_projection = nn.Linear(1, config.d_model)
        self.blocks = nn.ModuleList(T5Block(config, i == 0) for i in range(config.num_layers))
        self.final_layer_norm = T5LayerNorm(config.d_model, config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        if config.add_inter_token_attention:
            self.inter_token_blocks = nn.Parameter(torch.empty(config.n_inter_token_blocks, config.d_model))
            nn.init.normal_(self.inter_token_blocks, std=config.initializer_factor)

    def forward(self, input_ids, attention_mask, target_priors, lang_index=None, deterministic=True):
        if target_priors is None:
            raise ValueError("ZeTT's Flax T5 encoder requires target_priors")
        hidden = self.shared(input_ids)
        if self.config.embed_target_priors:
            prior = self.target_priors_projection(
                (target_priors[:, None, None].float() / math.log(1e-12)).to(hidden.dtype)
            )
            hidden = torch.cat((hidden, prior), dim=1)
            attention_mask = F.pad(attention_mask, (0, 1), value=1)
        hidden = F.dropout(hidden, p=self.dropout.p, training=not deterministic)
        bias = None
        latents = self.inter_token_blocks if self.config.add_inter_token_attention else None
        for block in self.blocks:
            hidden, bias = block(hidden, attention_mask, target_priors, latents, bias, deterministic)
        hidden = F.dropout(self.final_layer_norm(hidden), p=self.dropout.p, training=not deterministic)
        return SimpleNamespace(last_hidden_state=hidden)
