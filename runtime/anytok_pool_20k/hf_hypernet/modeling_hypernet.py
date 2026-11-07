from .configuration_hypernet import ZettHypernetConfig
from transformers import PreTrainedModel, RobertaConfig, RobertaModel, T5Config
from functools import partial

from torch import nn as nn
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .hyper_roberta import HyperRobertaModel
from .hyper_t5 import HyperT5Model

class Rescaler(nn.Module):
    def __init__(self, dim: int, trainable: bool = False):
        super().__init__()

        self.dim = dim

        self.w = nn.Parameter(torch.ones((1, self.dim)), requires_grad=trainable)
        self.b = nn.Parameter(torch.zeros((1, self.dim)), requires_grad=trainable)

    def __call__(self, x):
        return self.w * x + self.b

    @staticmethod
    def scale_to(x, target=None, target_stds=None, target_means=None):
        if target_stds is None:
            target_stds = target.std(dim=0, correction=0)
        if target_means is None:
            target_means = target.mean(dim=0)
        w = (target_stds / (x.std(dim=0, correction=0) + 1e-8))[None]
        b = (target_means - (x * w).mean(dim=0))[None]
        return w, b


class ProjectorBlock(nn.Module):
    def __init__(self, input_dim: int, dim: int, intermediate_dim: int):
        super().__init__()

        self.input_dim = input_dim
        self.dim = dim
        self.intermediate_dim = intermediate_dim

        self.dense1 = nn.Linear(self.input_dim, self.intermediate_dim)
        self.dense2 = nn.Linear(self.intermediate_dim, self.dim)

        self.ln = nn.LayerNorm(self.dim, eps=1e-6)

    def __call__(self, x):
        h = F.gelu(
            self.dense2(F.gelu(self.dense1(x), approximate="tanh")),
            approximate="tanh",
        )
        return self.ln(h + x).to(x.dtype)


class ZettHypernet(PreTrainedModel):
    config_class = ZettHypernetConfig

    def __init__(self, config: ZettHypernetConfig):
        super().__init__(config)

        self.config = config
        self.activation_checkpoint_input = False
        self.has_separate_out_embeddings = getattr(
            self.config, "separate_out_embeddings", False
        )

        if self.config.hn_embed_lang_id:
            self.lang_embeddings = nn.Embedding(
                self.config.n_langs, self.config.hn_hidden_size
            )

        if self.has_separate_out_embeddings:
            n_in_embd = self.config.n_embd * 2
            n_out_embd = self.config.n_embd
        else:
            n_in_embd = self.config.n_embd
            n_out_embd = self.config.n_embd

        if self.config.hn_model_type == "roberta":
            backbone_config = getattr(self.config, "hn_backbone_config", None)
            config = (RobertaConfig.from_dict(backbone_config)
                      if backbone_config is not None
                      else RobertaConfig.from_pretrained(self.config.hn_model_name_or_path))
            config.num_hidden_layers = self.config.hn_n_layers
            config.hidden_size = self.config.hn_hidden_size
            config.intermediate_size = self.config.hn_intermediate_size
            if getattr(self.config, "hn_num_attention_heads", None) is None:
                self.config.hn_num_attention_heads = self.config.hn_hidden_size // 64
            config.num_attention_heads = self.config.hn_num_attention_heads
            self.embed_init_range = config.initializer_range
            module_class = partial(RobertaModel, add_pooling_layer=False)
        elif self.config.hn_model_type == "t5":
            config = T5Config.from_pretrained(self.config.hn_model_name_or_path)
            config.num_layers = self.config.hn_n_layers
            if self.config.hn_hidden_size is not None:
                config.d_model = self.config.hn_hidden_size
            else:
                self.config.hn_hidden_size = config.d_model
            if self.config.hn_intermediate_size is not None:
                config.d_ff = self.config.hn_intermediate_size
            else:
                self.config.hn_intermediate_size = config.d_ff
            self.embed_init_range = config.initializer_factor * 1.0
            module_class = HyperT5Model
            if self.config.hn_embed_using_source_embeddings:
                raise NotImplementedError("ZeTT's Flax T5 encoder does not accept source embeddings")
        else:
            raise ValueError(f"Unsupported hypernetwork model type: {self.config.hn_model_type}")

        if self.config.hn_embed_using_source_embeddings:
            # do not need to alloc embeddings since inputs_embeds is always used
            config.vocab_size = self.config.pad_token_id + 1

        self.use_hyper_roberta = self.config.hn_model_type == "roberta" and bool(
            self.config.hn_flax_compatible_roberta
            or self.config.hn_add_inter_token_attention
            or self.config.hn_embed_target_priors
            or self.config.hn_embed_lang_id
            or self.config.hn_language_adapter_bottleneck_dim
        )
        self.use_hyper_t5 = self.config.hn_model_type == "t5"
        if self.use_hyper_roberta or self.use_hyper_t5:
            config.add_inter_token_attention = self.config.hn_add_inter_token_attention
            config.embed_target_priors = self.config.hn_embed_target_priors
            config.inter_token_attention_bias_by_priors = self.config.hn_inter_token_attention_bias_by_priors
            config.inter_token_attention_bias_scaler = self.config.hn_inter_token_attention_bias_scaler
            config.n_inter_token_blocks = self.config.hn_n_inter_token_blocks
            config.language_adapter_bottleneck_dim = self.config.hn_language_adapter_bottleneck_dim
            config.embed_lang_id = self.config.hn_embed_lang_id
            config.n_langs = self.config.n_langs
            if self.use_hyper_roberta:
                module_class = HyperRobertaModel

        self.pad_token_id = self.config.pad_token_id
        assert self.pad_token_id is not None
        self.model = module_class(config)

        # need at least one embedding
        self.fallback_embeddings = nn.Embedding(
            max(self.config.hn_n_extra_tokens, 1), n_in_embd
        )

        if self.config.hn_embed_using_source_embeddings:
            self.input_projection = nn.Sequential(
                *[
                    nn.Linear(n_in_embd, self.config.hn_hidden_size),
                    ProjectorBlock(
                        self.config.hn_hidden_size,
                        self.config.hn_hidden_size,
                        self.config.hn_intermediate_size,
                    ),
                ]
            )

        if self.config.hn_single_head:
            self.output_projection = nn.Sequential(
                *[
                    ProjectorBlock(
                        self.config.hn_hidden_size,
                        self.config.hn_hidden_size,
                        self.config.hn_intermediate_size,
                    ),
                    nn.Linear(self.config.hn_hidden_size, n_in_embd),
                ]
            )
        else:
            self.output_projection = nn.Sequential(
                *[
                    ProjectorBlock(
                        self.config.hn_hidden_size,
                        self.config.hn_hidden_size,
                        self.config.hn_intermediate_size,
                    ),
                    nn.Linear(self.config.hn_hidden_size, n_out_embd),
                ]
            )
            if self.has_separate_out_embeddings:
                self.output_projection_out = nn.Sequential(
                    *[
                        ProjectorBlock(
                            self.config.hn_hidden_size,
                            self.config.hn_hidden_size,
                            self.config.hn_intermediate_size,
                        ),
                        nn.Linear(self.config.hn_hidden_size, self.config.n_embd),
                    ]
                )

        if self.config.hn_rescale_embeddings:
            self.in_scaler = Rescaler(n_in_embd)
            self.scaler = Rescaler(n_out_embd)

            if self.has_separate_out_embeddings:
                self.out_scaler = Rescaler(self.config.n_embd, trainable=True)

        if getattr(self.config, "hn_predict_bias", False):
            self.bias_projection = nn.Linear(self.config.hn_hidden_size, 1)

        # Transformers 5 uses these properties when reloading saved custom
        # PreTrainedModel instances. They are populated by post_init().
        self.post_init()

    def expand_fallback_embeddings(self, size: int) -> None:
        """Add distinct source-piece embeddings without changing existing weights.

        The row at index zero is the source-piece padding row. Extra rows are
        addressed by surface-form IDs starting at original_vocab_size + 1.
        """
        old = self.fallback_embeddings
        if size < old.num_embeddings:
            raise ValueError("Cannot remove existing fallback embeddings")
        if size == old.num_embeddings:
            return
        expanded = nn.Embedding(
            size, old.embedding_dim, device=old.weight.device,
            dtype=old.weight.dtype,
        )
        nn.init.normal_(expanded.weight, mean=0.0, std=self.embed_init_range)
        with torch.no_grad():
            expanded.weight[:old.num_embeddings].copy_(old.weight)
        self.fallback_embeddings = expanded
        self.config.hn_n_extra_tokens = size

    @torch.no_grad()
    def init_rescaler(self, target_surface_forms, target_priors, source_embeddings,
                      lang_index, target_embeddings_in, target_embeddings_out=None):
        if not self.config.hn_rescale_embeddings:
            return
        win, bin_ = Rescaler.scale_to(
            source_embeddings,
            target_stds=torch.full_like(source_embeddings[0], self.embed_init_range),
            target_means=torch.zeros_like(source_embeddings[0]),
        )
        self.in_scaler.w.copy_(win)
        self.in_scaler.b.copy_(bin_)
        pred_in, pred_out, _ = self(target_surface_forms, target_priors, source_embeddings, lang_index)
        w, b = Rescaler.scale_to(pred_in, target_embeddings_in)
        self.scaler.w.copy_(w)
        self.scaler.b.copy_(b)
        if target_embeddings_out is not None:
            w, b = Rescaler.scale_to(pred_out, target_embeddings_out)
            self.out_scaler.w.copy_(w)
            self.out_scaler.b.copy_(b)

    def _project_source_inputs(self, target_surface_forms, source_embeddings):
        use_fallback = target_surface_forms >= self.config.original_vocab_size
        main_ids = target_surface_forms.clamp(max=self.config.original_vocab_size - 1)
        fallback_ids = (target_surface_forms - self.config.original_vocab_size).clamp(min=0)
        source_embeds = F.embedding(main_ids, weight=source_embeddings)
        if self.config.hn_rescale_embeddings:
            source_embeds = self.in_scaler(source_embeds)
        inputs_embeds = torch.where(
            use_fallback[..., None], self.fallback_embeddings(fallback_ids), source_embeds
        )
        return self.input_projection(inputs_embeds)

    def __call__(
        self,
        target_surface_forms,
        target_priors=None,
        source_embeddings=None,
        lang_index=None,
        deterministic: bool = True,
    ):
        if self.config.hn_embed_using_source_embeddings:
            if source_embeddings is None:
                raise ValueError("source_embeddings are required when hn_embed_using_source_embeddings=True")
            if self.activation_checkpoint_input and torch.is_grad_enabled():
                inputs_embeds = checkpoint(self._project_source_inputs,
                                           target_surface_forms, source_embeddings,
                                           use_reentrant=False, preserve_rng_state=True)
            else:
                inputs_embeds = self._project_source_inputs(target_surface_forms,
                                                            source_embeddings)
        else:
            inputs_embeds = None
        attention_mask = target_surface_forms != self.pad_token_id

        if self.config.hn_embed_lang_id and not (self.use_hyper_roberta or self.use_hyper_t5):
            lang_embedding = self.lang_embeddings(lang_index).squeeze()
            # position embed and type embed are added afterwards only in PT version so we need to subtract them here
            lang_embedding -= self.model.embeddings.token_type_embeddings(
                torch.tensor(0, device=self.device)
            ) + self.model.embeddings.position_embeddings(
                torch.tensor(attention_mask.shape[1], device=self.device)
            )

            lang_embedding = lang_embedding[None, None, :].expand(
                inputs_embeds.shape[0], -1, -1
            )

            inputs_embeds = torch.cat(
                [
                    inputs_embeds,
                    lang_embedding,
                ],
                axis=1,
            )
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(lang_embedding.shape[:-1], dtype=torch.bool, device=self.device),
                ],
                axis=1,
            )

        position_ids = torch.broadcast_to(
            torch.arange(torch.atleast_2d(attention_mask).shape[-1], device=self.device),
            attention_mask.shape,
        )

        model_kwargs = dict(inputs_embeds=inputs_embeds, input_ids=target_surface_forms,
                            attention_mask=attention_mask, position_ids=position_ids)
        if self.use_hyper_t5:
            model_kwargs.pop("inputs_embeds")
            model_kwargs.pop("position_ids")
            model_kwargs.update(target_priors=target_priors, lang_index=lang_index,
                                deterministic=deterministic)
        elif self.use_hyper_roberta:
            model_kwargs.update(target_priors=target_priors, lang_index=lang_index,
                                deterministic=deterministic)
        elif inputs_embeds is not None:
            model_kwargs.pop("input_ids")
        else:
            model_kwargs.pop("inputs_embeds")
        hidden_states = self.model(**model_kwargs).last_hidden_state

        if self.config.hn_concat_last_hidden_state:
            hidden_states = hidden_states.reshape(target_surface_forms.shape[0], -1)
        else:
            hidden_states = hidden_states[:, 0]

        predicted_embeddings = self.output_projection(hidden_states)

        if self.config.hn_single_head:
            predicted_embeddings_in = predicted_embeddings[..., : self.config.n_embd]

            if self.has_separate_out_embeddings:
                predicted_embeddings_out = predicted_embeddings[
                    ..., self.config.n_embd :
                ]
            else:
                predicted_embeddings_out = None
        else:
            predicted_embeddings_in = predicted_embeddings
            if self.has_separate_out_embeddings:
                predicted_embeddings_out = self.output_projection_out(hidden_states)
            else:
                predicted_embeddings_out = None

        if self.config.hn_rescale_embeddings:
            predicted_embeddings_in = self.scaler(predicted_embeddings_in)

            if predicted_embeddings_out is not None:
                predicted_embeddings_out = self.out_scaler(predicted_embeddings_out)

        if getattr(self.config, "hn_predict_bias", False):
            predicted_bias = self.bias_projection(hidden_states)[..., 0]
        else:
            predicted_bias = torch.zeros_like(
                target_surface_forms[..., 0], dtype=self.dtype
            )

        return predicted_embeddings_in, predicted_embeddings_out, predicted_bias
