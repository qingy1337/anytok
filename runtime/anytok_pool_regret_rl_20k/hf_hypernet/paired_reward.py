"""Reversible, shared training contexts and a pre-update canonical control."""
import hashlib
import json
import math

import torch


def make_paired_prefix_example(raw, target_tokenizer, canonical, max_positions=128, min_chars=32):
    """Nearest valid canonical boundary to the midpoint, shrinking as before.

    The continuation remains variable-length, as in the original training run;
    it is not replaced by the fixed 32-token evaluation continuation.
    """
    for _ in range(32):
        if len(raw) < min_chars:
            return None
        encoding = canonical._tokenizer.encode(raw, add_special_tokens=False)
        cuts = sorted({end for _, end in encoding.offsets if 0 < end < len(raw)},
                      key=lambda end: (abs(end - len(raw) // 2), end < len(raw) // 2, end))
        for cut in cuts:
            prefix, suffix = raw[:cut], raw[cut:]
            canonical_prefix = canonical(prefix, add_special_tokens=False).input_ids
            targets = canonical(suffix, add_special_tokens=False).input_ids
            if (not canonical_prefix or not targets or 0 in canonical_prefix or 0 in targets
                    or encoding.ids != canonical_prefix + targets
                    or canonical.decode(canonical_prefix, clean_up_tokenization_spaces=False) != prefix
                    or canonical.decode(targets, clean_up_tokenization_spaces=False) != suffix):
                continue
            native_prefix = target_tokenizer.encode(prefix, add_special_tokens=False).ids
            if (not native_prefix or 0 in native_prefix
                    or target_tokenizer.decode(native_prefix, skip_special_tokens=False) != prefix):
                continue
            # Keep the nearest valid split. If oversized, shorten the window
            # rather than choosing an unusually short prefix or continuation.
            positions = 1 + max(len(native_prefix), len(canonical_prefix)) + len(targets)
            if positions > max_positions:
                break
            size = len(suffix.encode('utf-8'))
            native = (native_prefix, targets, size)
            control = (canonical_prefix, targets, size)
            evidence = dict(window_chars=len(raw), cut_chars=cut, maximum_positions=positions,
                continuation_tokens=len(targets), continuation_bytes=size,
                canonical_joint_encoding_matches=True,
                context_sha256=hashlib.sha256(json.dumps([prefix, targets], ensure_ascii=False).encode()).hexdigest())
            return native, control, evidence
        raw = raw[:int(len(raw) * .85)]
    return None


def paired_gap_bpb(transferred_nats, canonical_nats, scored_bytes):
    if (scored_bytes <= 0 or not math.isfinite(transferred_nats) or not math.isfinite(canonical_nats)
            or transferred_nats < 0 or canonical_nats < 0):
        raise ValueError('Invalid paired continuation losses')
    denominator = scored_bytes * math.log(2)
    return transferred_nats / denominator, canonical_nats / denominator, (transferred_nats - canonical_nats) / denominator


def canonical_loss_nats(lm, examples, source, microbatch_size, make_microbatch):
    """No gradients or persistent RNG changes; same LM state before its update."""
    device = source.device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    total = 0.
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        for start in range(0, len(examples), microbatch_size):
            micro = examples[start:start + microbatch_size]
            embeds, mask, labels, positions = make_microbatch(micro, source, source, device)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                loss = lm(inputs_embeds=embeds, attention_mask=mask, position_ids=positions,
                          labels=labels, use_cache=False).loss
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite paired canonical loss')
            total += float(loss) * sum(len(example[1]) for example in micro)
    return total
