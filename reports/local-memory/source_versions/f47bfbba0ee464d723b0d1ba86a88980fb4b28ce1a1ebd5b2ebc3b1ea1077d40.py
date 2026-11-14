"""Opt-in diagnostic patches; never imported by the production trainer."""
from contextlib import nullcontext
from pathlib import Path
import shutil
from types import MethodType

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def recompute(function, *args):
    return checkpoint(function, *args, use_reentrant=False, preserve_rng_state=True)


def split_attention(self, hidden, keys=None, mask=None, bias=None, head_mask=None, deterministic=True):
    if keys is None:keys=hidden
    def core(h, k_input):
        batch,q_len,width=h.shape;k_len=k_input.shape[1]
        q=self.query(h).view(batch,q_len,self.n_heads,self.head_width).transpose(1,2)
        k=self.key(k_input).view(batch,k_len,self.n_heads,self.head_width).transpose(1,2)
        v=self.value(k_input).view(batch,k_len,self.n_heads,self.head_width).transpose(1,2)
        scores=torch.matmul(q,k.transpose(-2,-1))/(self.head_width**.5)
        if mask is not None:scores=scores.masked_fill(~mask[:,None,None,:].bool(),torch.finfo(scores.dtype).min)
        if bias is not None:scores=scores+bias.to(scores.dtype)[None,None,None,:]
        weights=F.dropout(torch.softmax(scores,dim=-1),p=self.attention_dropout.p,training=not deterministic)
        if head_mask is not None:weights=weights*head_mask[None,:,None,None]
        result=torch.matmul(weights,v).transpose(1,2).reshape(batch,q_len,width)
        return self.dense(result)
    dense=recompute(core,hidden,keys)
    residual=hidden+F.dropout(dense,p=self.dropout.p,training=not deterministic)
    return recompute(lambda x:self.LayerNorm(x).to(hidden.dtype),residual)


def fine_layer(self, hidden, mask, priors, latents, lang_index=None, head_mask=None, deterministic=True):
    if self.adapter_width:
        raise ValueError('The diagnostic patch is restricted to the actual adapter-free configuration')
    split=getattr(self,'_memory_split_norm',False)
    invoke=(lambda fn,*args:fn(*args)) if split else recompute
    attended = invoke(lambda h: self.attention(h, mask=mask, head_mask=head_mask,
                                              deterministic=deterministic), hidden)
    if self.inter_token_attention:
        bias = priors * self.prior_bias_scale if self.prior_bias else None
        blocks = invoke(lambda l, a: self.inter_token_attention_down(l[None],
            keys=a[None, :, 0], bias=bias, deterministic=deterministic)[0], latents, attended)
        attended = invoke(lambda a, b: self.inter_token_attention_up(
            a.reshape(1, -1, a.shape[-1]), keys=b[None], deterministic=deterministic).reshape_as(a), attended, blocks)
    if split:
        dense=recompute(lambda a:self.output_dense(self.act(self.intermediate(a))),attended)
        residual=attended+F.dropout(dense,p=self.output_dropout.p,training=not deterministic)
        return recompute(lambda x:self.output_norm(x).to(attended.dtype),residual)
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
    if getattr(self,'_memory_split_norm',False):
        block=self.input_projection[1]
        intermediate=recompute(lambda x:F.gelu(block.dense1(x),approximate='tanh'),projected)
        result=recompute(lambda x:F.gelu(block.dense2(x),approximate='tanh'),intermediate)
        return recompute(lambda x:block.ln(x).to(projected.dtype),result+projected)
    return recompute(self.input_projection[1], projected)


def apply_variant(hypernetwork, variant):
    if variant not in ['baseline', 'offload', 'fine', 'fine_offload', 'split_norm_offload', 'split_norm_disk']:
        raise ValueError('Unknown memory variant')
    names = list(hypernetwork.state_dict())
    if variant.startswith(('fine','split_norm')):
        hypernetwork.model.activation_checkpointing = False
        hypernetwork.activation_checkpoint_input = False
        hypernetwork._project_source_inputs = MethodType(fine_projection, hypernetwork)
        hypernetwork._memory_split_norm = variant.startswith('split_norm')
        for layer in hypernetwork.model.layers:
            layer.forward = MethodType(fine_layer, layer)
            layer._memory_split_norm=hypernetwork._memory_split_norm
            if layer._memory_split_norm:
                for module in [layer.attention,layer.inter_token_attention_down,layer.inter_token_attention_up]:
                    module.forward=MethodType(split_attention,module)
        original = hypernetwork.output_projection.forward
        hypernetwork.output_projection.forward = lambda x: recompute(original, x)
    if list(hypernetwork.state_dict()) != names:
        raise ValueError('A diagnostic patch changed checkpoint parameter names')


class HostTensor:
    def __init__(self, tensor, ledger):
        self.shape, self.stride = tuple(tensor.shape), tuple(tensor.stride())
        self.offset=tensor.storage_offset()
        self.elements=1+self.offset+sum((n-1)*s for n,s in zip(self.shape,self.stride))
        self.cpu = torch.empty(self.elements, dtype=tensor.dtype, device='cpu').as_strided(self.shape,self.stride,self.offset)
        self.cpu.copy_(tensor.detach())
        self.device = tensor.device
        self.nbytes = self.cpu.untyped_storage().nbytes()
        self.ledger = ledger
        ledger.live += self.nbytes
        ledger.peak = max(ledger.peak, ledger.live)
        ledger.copies += 1

    def __del__(self):
        if hasattr(self, 'ledger'):
            self.ledger.live -= self.nbytes


class DiskTensor:
    def __init__(self,tensor,ledger):
        self.shape,self.stride=tuple(tensor.shape),tuple(tensor.stride())
        self.offset=tensor.storage_offset();self.device=tensor.device;self.dtype=tensor.dtype
        self.elements=1+self.offset+sum((n-1)*s for n,s in zip(self.shape,self.stride))
        self.path=ledger.spill_directory/f'activation-{ledger.files_spilled:06d}.pt'
        ledger.files_spilled+=1
        cpu=tensor.detach().to('cpu',copy=True)
        ledger.peak=max(ledger.peak,ledger.live+cpu.untyped_storage().nbytes())
        torch.save(cpu,self.path);self.nbytes=self.path.stat().st_size
        self.ledger=ledger;ledger.disk_live+=self.nbytes
        ledger.disk_peak=max(ledger.disk_peak,ledger.disk_live)

    def __del__(self):
        if hasattr(self,'ledger'):
            self.ledger.disk_live-=self.nbytes
            self.path.unlink(missing_ok=True)


class BoundedOffload:
    """Saved activations only, bounded host RAM; parameters remain resident.

    Copies are synchronous and preserve values and strides. Numeric
    equivalence is still measured, including the baseline's reproducibility.
    """
    def __init__(self, parameters, maximum_bytes=64 * 2**30, spill_directory=None):
        self.parameter_storage = {p.untyped_storage().data_ptr() for p in parameters}
        self.maximum_bytes = maximum_bytes
        self.live = self.peak = self.copies = self.skipped_for_budget = 0
        self.spill_directory=Path(spill_directory) if spill_directory else None
        self.files_spilled=self.disk_live=self.disk_peak=0
        self.disk_capacity=144*2**30
        self.spill_reserve=min(16*2**30,maximum_bytes//2) if self.spill_directory else 0
        self.cpu_capacity=maximum_bytes-self.spill_reserve
        if self.spill_directory:self.spill_directory.mkdir(parents=True,exist_ok=True)

    def pack(self, tensor):
        logical_size = tensor.numel() * tensor.element_size()
        if (tensor.device.type != 'cuda' or logical_size < 2**20
                or tensor.untyped_storage().data_ptr() in self.parameter_storage):
            return tensor
        if any(n > 1 and s == 0 for n,s in zip(tensor.shape,tensor.stride())):
            return tensor  # Do not materialize overlapping broadcast storage.
        size = (1 + tensor.storage_offset() + sum((n-1)*s for n,s in zip(tensor.shape,tensor.stride()))) * tensor.element_size()
        if self.live + size > self.cpu_capacity:
            if (self.spill_directory and logical_size<=self.spill_reserve
                    and self.disk_live+logical_size+2**20<=self.disk_capacity
                    and shutil.disk_usage(self.spill_directory).free>logical_size+16*2**30):
                return DiskTensor(tensor,self)
            self.skipped_for_budget += 1
            return tensor
        return HostTensor(tensor, self)

    @staticmethod
    def unpack(value):
        if isinstance(value,DiskTensor):
            cpu=torch.load(value.path,map_location='cpu',weights_only=True,mmap=True)
            restored=torch.empty(value.elements,dtype=value.dtype,device=value.device).as_strided(value.shape,value.stride,value.offset)
            restored.copy_(cpu)
            return restored
        if not isinstance(value, HostTensor):return value
        restored = torch.empty(value.elements,dtype=value.cpu.dtype,device=value.device).as_strided(value.shape,value.stride,value.offset)
        restored.copy_(value.cpu)
        return restored

    def context(self, enabled):
        return torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack) if enabled else nullcontext()

    def statistics(self):
        return dict(peak_cpu_offload_bytes=self.peak, activation_copies=self.copies,
                    skipped_for_host_budget=self.skipped_for_budget,
                    peak_disk_spill_bytes=self.disk_peak,files_spilled=self.files_spilled)
