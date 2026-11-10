"""PyTorch implementation of ZeTT's Flax HyperRobertaModule.

This follows zett/model/hyper_roberta.py, including the optional prior token
and the two vocabulary-wide cross-attention blocks in each encoder layer.
The learned latent queries are reused at each layer, as in the Flax code.
"""

import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class HyperAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_width = width // self.n_heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.dense = nn.Linear(width, width)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.attention_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.LayerNorm = nn.LayerNorm(width, eps=config.layer_norm_eps)

    def forward(self, hidden, keys=None, mask=None, bias=None, head_mask=None, deterministic=True):
        if keys is None:
            keys = hidden
        batch, q_len, width = hidden.shape
        k_len = keys.shape[1]
        q = self.query(hidden).view(batch, q_len, self.n_heads, self.head_width).transpose(1, 2)
        k = self.key(keys).view(batch, k_len, self.n_heads, self.head_width).transpose(1, 2)
        v = self.value(keys).view(batch, k_len, self.n_heads, self.head_width).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_width ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :].bool(), torch.finfo(scores.dtype).min)
        if bias is not None:
            scores = scores + bias.to(scores.dtype)[None, None, None, :]
        weights = F.dropout(torch.softmax(scores, dim=-1), p=self.attention_dropout.p, training=not deterministic)
        if head_mask is not None:
            weights = weights * head_mask[None, :, None, None]
        result = torch.matmul(weights, v).transpose(1, 2).reshape(batch, q_len, width)
        return self.LayerNorm(hidden + F.dropout(self.dense(result), p=self.dropout.p,
                                                 training=not deterministic)).to(hidden.dtype)


class HyperLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = HyperAttention(config)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.act = F.gelu if config.hidden_act == "gelu" else None
        if self.act is None:
            raise ValueError(f"Unsupported hidden activation: {config.hidden_act}")
        self.inter_token_attention = config.add_inter_token_attention
        self.prior_bias = config.inter_token_attention_bias_by_priors
        self.prior_bias_scale = config.inter_token_attention_bias_scaler
        if self.inter_token_attention:
            self.inter_token_attention_down = HyperAttention(config)
            self.inter_token_attention_up = HyperAttention(config)
        self.adapter_width = config.language_adapter_bottleneck_dim
        if self.adapter_width:
            self.lang_adapter_down_weights = nn.Parameter(torch.empty(config.n_langs, config.hidden_size, self.adapter_width))
            self.lang_adapter_down_bias = nn.Parameter(torch.zeros(config.n_langs, self.adapter_width))
            self.lang_adapter_up_weights = nn.Parameter(torch.empty(config.n_langs, self.adapter_width, config.hidden_size))
            self.lang_adapter_up_bias = nn.Parameter(torch.zeros(config.n_langs, config.hidden_size))
            self.lang_adapter_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            nn.init.normal_(self.lang_adapter_down_weights, std=1 / math.sqrt(config.hidden_size))
            nn.init.normal_(self.lang_adapter_up_weights, std=1 / math.sqrt(self.adapter_width))

    def forward(self, hidden, mask, priors, latents, lang_index=None, head_mask=None, deterministic=True):
        attended = self.attention(hidden, mask=mask, head_mask=head_mask, deterministic=deterministic)
        if self.inter_token_attention:
            bias = priors * self.prior_bias_scale if self.prior_bias else None
            blocks = self.inter_token_attention_down(latents[None], keys=attended[None, :, 0], bias=bias,
                                                      deterministic=deterministic)[0]
            attended = self.inter_token_attention_up(
                attended.reshape(1, -1, attended.shape[-1]), keys=blocks[None], deterministic=deterministic
            ).reshape_as(attended)
        hidden = self.output_norm(attended + F.dropout(self.output_dense(self.act(self.intermediate(attended))),
                                                       p=self.output_dropout.p, training=not deterministic)).to(attended.dtype)
        if self.adapter_width:
            down = torch.matmul(hidden, self.lang_adapter_down_weights[lang_index]) + self.lang_adapter_down_bias[lang_index]
            up = torch.matmul(self.act(down), self.lang_adapter_up_weights[lang_index]) + self.lang_adapter_up_bias[lang_index]
            hidden = self.lang_adapter_norm(hidden + up).to(hidden.dtype)
        return hidden


class HyperRobertaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, width)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, width)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, width)
        self.LayerNorm = nn.LayerNorm(width, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.target_priors_projection = nn.Linear(1, width)
        self.lang_embedding = nn.Embedding(max(config.n_langs or 1, 1), width)
        self.layers = nn.ModuleList(HyperLayer(config) for _ in range(config.num_hidden_layers))
        self.inter_token_blocks = nn.Parameter(torch.empty(config.n_inter_token_blocks, width))
        nn.init.normal_(self.inter_token_blocks, std=config.initializer_range)
        self.activation_checkpointing = False

    def forward(self, input_ids, inputs_embeds=None, attention_mask=None, position_ids=None,
                token_type_ids=None, target_priors=None, lang_index=None, head_mask=None,
                deterministic=True):
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device)[None].expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        # nn.Embedding does not participate in autocast. Match the dtype of the
        # projected source embeddings so BF16 training does not promote the
        # entire vocabulary-wide activation graph back to FP32.
        hidden = (inputs_embeds
                  + self.position_embeddings(position_ids).to(inputs_embeds.dtype)
                  + self.token_type_embeddings(token_type_ids).to(inputs_embeds.dtype))
        if self.config.embed_target_priors:
            if target_priors is None:
                raise ValueError("target_priors are required when embed_target_priors=True")
            hidden = torch.cat((hidden, self.target_priors_projection(target_priors[:, None, None].to(hidden.dtype))), dim=1)
            attention_mask = F.pad(attention_mask, (0, 1), value=1)
        if self.config.embed_lang_id:
            if lang_index is None:
                raise ValueError("lang_index is required when embed_lang_id=True")
            lang = self.lang_embedding(lang_index).to(hidden.dtype).reshape(1, 1, -1).expand(hidden.shape[0], -1, -1)
            hidden = torch.cat((hidden, lang), dim=1)
            attention_mask = F.pad(attention_mask, (0, 1), value=1)
        hidden = F.dropout(self.LayerNorm(hidden).to(hidden.dtype), p=self.dropout.p, training=not deterministic)
        if self.config.add_inter_token_attention and self.config.inter_token_attention_bias_by_priors and target_priors is None:
            raise ValueError("target_priors are required for prior-biased inter-token attention")
        for i, layer in enumerate(self.layers):
            layer_head_mask = None if head_mask is None else head_mask[i]
            if self.activation_checkpointing and torch.is_grad_enabled():
                hidden = checkpoint(layer, hidden, attention_mask, target_priors,
                                    self.inter_token_blocks, lang_index,
                                    layer_head_mask, deterministic,
                                    use_reentrant=False, preserve_rng_state=True)
            else:
                hidden = layer(hidden, attention_mask, target_priors,
                               self.inter_token_blocks, lang_index,
                               layer_head_mask, deterministic)
        return SimpleNamespace(last_hidden_state=hidden)
