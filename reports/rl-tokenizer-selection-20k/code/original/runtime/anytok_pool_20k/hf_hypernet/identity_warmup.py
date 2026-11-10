"""ZeTT's original-token embedding warmup, restricted to input embeddings."""

import torch
from transformers import RobertaConfig

from .configuration_hypernet import ZettHypernetConfig
from .modeling_hypernet import ZettHypernet


def build_input_hypernet(source_vocab_size: int, embedding_width: int,
                         hidden_width: int = None, intermediate_width: int = None,
                         layers: int = 3, attention_heads: int = None,
                         max_pieces: int = 10) -> ZettHypernet:
    """Use the ported ZeTT RoBERTa network with priors and vocabulary attention."""
    hidden_width = hidden_width or embedding_width
    intermediate_width = intermediate_width or 2 * hidden_width
    attention_heads = attention_heads or max(1, min(hidden_width // 64, 32))
    if hidden_width % attention_heads:
        raise ValueError("hidden_width must be divisible by attention_heads")
    roberta = RobertaConfig(
        hidden_size=hidden_width, intermediate_size=intermediate_width,
        num_hidden_layers=layers, num_attention_heads=attention_heads,
        vocab_size=source_vocab_size + 1,
        max_position_embeddings=max_pieces + 2,
        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
    )
    config = ZettHypernetConfig(
        hn_model_type="roberta", hn_backbone_config=roberta.to_dict(),
        hn_n_layers=layers, hn_hidden_size=hidden_width,
        hn_intermediate_size=intermediate_width,
        hn_num_attention_heads=attention_heads,
        n_embd=embedding_width, hn_surface_maxlen=max_pieces,
        hn_embed_using_source_embeddings=True,
        hn_embed_target_priors=True, hn_add_inter_token_attention=True,
        hn_inter_token_attention_bias_by_priors=False,
        hn_n_inter_token_blocks=16, hn_embed_lang_id=False,
        hn_rescale_embeddings=True, hn_predict_bias=False,
        original_vocab_size=source_vocab_size,
        # A separate padding ID avoids hiding a real source token such as EOS.
        pad_token_id=source_vocab_size, hn_n_extra_tokens=1,
        separate_out_embeddings=False,
    )
    model = ZettHypernet(config)
    model.model.activation_checkpointing = True
    model.activation_checkpoint_input = True
    return model


@torch.no_grad()
def initialize_rescalers(model: ZettHypernet, source_embeddings: torch.Tensor,
                         anchor_count: int = 2048) -> None:
    """Match ZeTT's source and prediction scales using canonical anchors."""
    count = min(anchor_count, source_embeddings.shape[0])
    forms = torch.arange(count, device=source_embeddings.device)[:, None]
    priors = torch.zeros(count, device=source_embeddings.device)
    model.init_rescaler(forms, priors, source_embeddings, None,
                        source_embeddings[:count])


def identity_loss(model: ZettHypernet, source_embeddings: torch.Tensor,
                  token_ids: torch.Tensor) -> torch.Tensor:
    """ZeTT train.py identity loss: mean over tokens of squared vector error."""
    if token_ids.ndim != 1:
        raise ValueError("token_ids must be a one-dimensional selection")
    forms = token_ids[:, None]  # One original token is one source piece.
    priors = torch.zeros(len(token_ids), device=token_ids.device)
    predicted, output_embeddings, _ = model(
        forms, priors, source_embeddings, None, deterministic=False
    )
    if output_embeddings is not None:
        raise ValueError("Warmup expects an input-only hypernetwork")
    targets = source_embeddings.index_select(0, token_ids)
    return (predicted.float() - targets.float()).square().sum(-1).mean()


@torch.no_grad()
def identity_validation_loss(model: ZettHypernet, source_embeddings: torch.Tensor,
                             token_ids: torch.Tensor) -> float:
    """Measure copying error on a fixed token selection, without dropout."""
    was_training = model.training
    model.eval()
    device_type = source_embeddings.device.type
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                        enabled=device_type == "cuda"):
        predicted, output_embeddings, _ = model(
            token_ids[:, None], torch.zeros(len(token_ids), device=token_ids.device),
            source_embeddings, None, deterministic=True,
        )
    if was_training:
        model.train()
    if output_embeddings is not None:
        raise ValueError("Validation expects an input-only hypernetwork")
    targets = source_embeddings.index_select(0, token_ids)
    return float((predicted.float() - targets.float()).square().sum(-1).mean())


def make_warmup_optimizer(model: ZettHypernet, learning_rate: float = 3e-4):
    """ZeTT's AdamW settings, without decay on biases and normalization."""
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith("bias") or parameter.ndim < 2 or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.01},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=learning_rate, betas=(0.9, 0.95),
    )


def freeze_rescalers(model: ZettHypernet) -> None:
    """Restore ZeTT's fixed affine scales after Transformers reloads weights."""
    if model.config.hn_rescale_embeddings:
        model.in_scaler.requires_grad_(False)
        model.scaler.requires_grad_(False)


def identity_warmup_step(model: ZettHypernet, source_embeddings: torch.Tensor,
                         optimizer: torch.optim.Optimizer, batch_size: int,
                         step: int, total_steps: int, peak_lr: float = 3e-4):
    """Sample original tokens without replacement and update the hypernetwork."""
    if batch_size > len(source_embeddings) or batch_size < 1:
        raise ValueError("batch_size must be between 1 and source vocabulary size")
    if total_steps < 1 or step < 0 or step >= total_steps:
        raise ValueError("step must be within the warmup range")
    token_ids = torch.randperm(len(source_embeddings), device=source_embeddings.device)[:batch_size]
    learning_rate = peak_lr * step / total_steps
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    optimizer.zero_grad(set_to_none=True)
    model.train()
    device_type = source_embeddings.device.type
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                        enabled=device_type == "cuda"):
        loss = identity_loss(model, source_embeddings, token_ids)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
        raise FloatingPointError("Non-finite warmup loss or gradients")
    optimizer.step()
    return float(loss.detach()), float(gradient_norm), learning_rate
