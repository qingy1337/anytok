"""Opt-in diagnostic patches; never imported by the production trainer."""
from contextlib import nullcontext
from types import MethodType

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def recompute(function, *args):
    return checkpoint(function, *args, use_reentrant=False, preserve_rng_state=True)


def fine_layer(self, hidden, mask, priors, latents, lang_index=None, head_mask=None, deterministic=True):
    if self.adapter_width:
        raise ValueError('The diagnostic patch is restricted to the actual adapter-free configuration')
    attended = recompute(lambda h: self.attention(h, mask=mask, head_mask=head_mask,
                                                 deterministic=deterministic), hidden)
    if self.inter_token_attention:
        bias = priors * self.prior_bias_scale if self.prior_bias else None
        blocks = recompute(lambda l, a: self.inter_token_attention_down(l[None],
            keys=a[None, :, 0], bias=bias, deterministic=deterministic)[0], latents, attended)
        attended = recompute(lambda a, b: self.inter_token_attention_up(
            a.reshape(1, -1, a.shape[-1]), keys=b[None], deterministic=deterministic).reshape_as(a), attended, blocks)
    return recompute(lambda a: self.output_norm(a + F.dropout(
        self.output_dense(self.act(self.intermediate(a))), p=self.output_dropout.p,
        training=not deterministic)).to(a.dtype), attended)


def fine_projection(self, forms, source):
    def embeddings(f, s):
        fallback = f >= self.config.original_vocab_size
        main_ids = f.clamp(max=self.config.original_vocab_size - 1)
        fallback_ids = (f - self.config.original_vocab_size).clamp(min=0)
        embedded = F.embedding(main_ids, weight=s)
        if self.config.hn_rescale_embeddings:
            embedded = self.in_scaler(embedded)
        return torch.where(fallback[..., None], self.fallback_embeddings(fallback_ids), embedded)
    embedded = recompute(embeddings, forms, source)
    projected = recompute(self.input_projection[0], embedded)
    return recompute(self.input_projection[1], projected)


def apply_variant(hypernetwork, variant):
    if variant not in ['baseline', 'offload', 'fine', 'fine_offload']:
        raise ValueError('Unknown memory variant')
    names = list(hypernetwork.state_dict())
    if variant.startswith('fine'):
        hypernetwork.model.activation_checkpointing = False
        hypernetwork.activation_checkpoint_input = False
        hypernetwork._project_source_inputs = MethodType(fine_projection, hypernetwork)
        for layer in hypernetwork.model.layers:
            layer.forward = MethodType(fine_layer, layer)
        original = hypernetwork.output_projection.forward
        hypernetwork.output_projection.forward = lambda x: recompute(original, x)
    if list(hypernetwork.state_dict()) != names:
        raise ValueError('A diagnostic patch changed checkpoint parameter names')


class HostTensor:
    def __init__(self, tensor, ledger):
        self.cpu = tensor.detach().to('cpu', copy=True)
        self.device = tensor.device
        self.nbytes = self.cpu.numel() * self.cpu.element_size()
        self.ledger = ledger
        ledger.live += self.nbytes
        ledger.peak = max(ledger.peak, ledger.live)
        ledger.copies += 1

    def __del__(self):
        if hasattr(self, 'ledger'):
            self.ledger.live -= self.nbytes


class BoundedOffload:
    """Saved activations only, bounded host RAM; parameters remain resident.

    Copies are synchronous and preserve values. Numeric equivalence still has
    to be measured: restored tensor layout can affect CUDA kernel choices.
    """
    def __init__(self, parameters, maximum_bytes=64 * 2**30):
        self.parameter_storage = {p.untyped_storage().data_ptr() for p in parameters}
        self.maximum_bytes = maximum_bytes
        self.live = self.peak = self.copies = self.skipped_for_budget = 0

    def pack(self, tensor):
        size = tensor.numel() * tensor.element_size()
        if (tensor.device.type != 'cuda' or size < 2**20
                or tensor.untyped_storage().data_ptr() in self.parameter_storage):
            return tensor
        if self.live + size > self.maximum_bytes:
            self.skipped_for_budget += 1
            return tensor
        return HostTensor(tensor, self)

    @staticmethod
    def unpack(value):
        return value.cpu.to(value.device) if isinstance(value, HostTensor) else value

    def context(self, enabled):
        return torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack) if enabled else nullcontext()

    def statistics(self):
        return dict(peak_cpu_offload_bytes=self.peak, activation_copies=self.copies,
                    skipped_for_host_budget=self.skipped_for_budget)
