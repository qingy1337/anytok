"""Fixed 128-position evaluation on the existing eleven unseen 48k BPEs."""
import gzip
import hashlib
import json
import math
from pathlib import Path

from pool_generalization import file_sha, save_json, shared_prefix

LABELS = ('Base', 'Code50', 'Papers25', 'Math50', 'QA25', 'Books25',
          'Medical40', 'Legal40', 'Spanish35', 'Chinese35', 'Japanese35')
STEPS = (2000, 6000, 12000)
EXPECTED = (256, 8192, 38534)


def read_rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def prepare_panel(corpus_path, examples_path, canonical, targets):
    """Keep all saved IDs/suffixes; trim only a shared prefix to fit every BPE."""
    corpus_rows = read_rows(corpus_path)
    corpus = {r['id']: r for r in corpus_rows}
    rows = read_rows(examples_path)
    if len(corpus) != len(corpus_rows) or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate document IDs')
    prepared = []
    for row in rows:
        text = corpus[row['id']]['text']
        prefix, suffix = text[:row['cut_chars']], text[row['cut_chars']:]
        continuation = row['canonical_continuation_ids']
        if (row['source'] != 'fineweb-edu' or len(continuation) != 32
                or canonical(prefix, add_special_tokens=False).input_ids != row['canonical_prefix_ids']
                or canonical(suffix, add_special_tokens=False).input_ids != continuation
                or canonical.decode(continuation, clean_up_tokenization_spaces=False) != suffix
                or len(suffix.encode('utf-8')) != row['continuation_bytes']):
            raise ValueError(f"Saved continuation changed: {row['id']}")
        common, source_ids, maximum = shared_prefix(prefix, continuation, canonical, targets, 128)
        prepared.append({'id': row['id'], 'prefix': common,
                         'canonical_prefix_ids': source_ids,
                         'canonical_continuation_ids': continuation,
                         'continuation_bytes': row['continuation_bytes'],
                         'maximum_positions': 1 + maximum + len(continuation)})
    counts = (len(prepared), sum(len(r['canonical_continuation_ids']) for r in prepared),
              sum(r['continuation_bytes'] for r in prepared))
    if counts != EXPECTED or any(r['maximum_positions'] > 128 for r in prepared):
        raise ValueError(f'Panel counts or position cap changed: {counts}')
    return prepared


def examples_for(panel, tokenizer=None):
    examples = []
    for row in panel:
        prefix = (row['canonical_prefix_ids'] if tokenizer is None else
                  tokenizer.encode(row['prefix'], add_special_tokens=False).ids)
        if tokenizer is not None and tokenizer.decode(prefix, skip_special_tokens=False) != row['prefix']:
            raise ValueError('Prefix does not round-trip')
        suffix = row['canonical_continuation_ids']
        if not prefix or 1 + len(prefix) + len(suffix) > 128:
            raise ValueError('Context exceeds 128 positions')
        examples.append((prefix, suffix, row['continuation_bytes']))
    return examples


def document_nll(logits, labels):
    """Causal shift: prefix/padding labels are ignored, including the BOS."""
    import torch
    import torch.nn.functional as F
    shifted = labels[:, 1:]
    losses = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                             shifted.reshape(-1), ignore_index=-100, reduction='none')
    nats = losses.reshape(shifted.shape).sum(dim=1)
    counts = (shifted != -100).sum(dim=1)
    if not torch.isfinite(nats).all():
        raise FloatingPointError('Non-finite evaluation score')
    return nats, counts


def score_documents(model, examples, inputs, canonical_table, batch_size=8):
    import torch
    from hf_hypernet.joint_training import make_lm_microbatch
    result = []
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            micro = examples[start:start + batch_size]
            embeds, mask, labels, positions = make_lm_microbatch(micro, inputs, canonical_table, 'cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(inputs_embeds=embeds, attention_mask=mask,
                               position_ids=positions, use_cache=False).logits
            nats, counts = document_nll(logits, labels)
            if counts.tolist() != [len(r[1]) for r in micro]:
                raise ValueError('Scored token count changed')
            result.extend(nats.double().cpu().tolist())
            del logits
    return result


def summarize(results, panel):
    import numpy as np
    byte_counts = np.array([r['continuation_bytes'] for r in panel])
    denominator = byte_counts.sum() * math.log(2)
    rows = []
    for step in STEPS:
        score = results[str(step)]
        canonical = float(np.sum(score['canonical_nats']) / denominator)
        tokenizers = {}
        for name in LABELS:
            nats = np.array(score['tokenizers'][name]['nats'])
            if nats.shape != (EXPECTED[0],) or not np.isfinite(nats).all():
                raise ValueError('Incomplete tokenizer score')
            bpb = float(nats.sum() / denominator)
            tokenizers[name] = {'bpb': bpb, 'gap_bpb': bpb - canonical}
        mean = float(np.mean([r['bpb'] for r in tokenizers.values()]))
        rows.append({'step': step, 'canonical_bpb': canonical, 'mean_bpb': mean,
                     'mean_gap_bpb': mean - canonical, 'tokenizers': tokenizers})
    # Fixed tokenizer panel, resample paired documents only, same draws at both steps.
    first, last = results[str(STEPS[0])], results[str(STEPS[-1])]
    def excess(r):
        return np.mean([r['tokenizers'][n]['nats'] for n in LABELS], axis=0) - r['canonical_nats']
    delta = excess(last) - excess(first)
    rng = np.random.default_rng(20261107)
    draws = rng.integers(EXPECTED[0], size=(5000, EXPECTED[0]))
    changes = delta[draws].sum(axis=1) / (byte_counts[draws].sum(axis=1) * math.log(2))
    return {'rows': rows, 'gap_change_2k_to_12k': float(delta.sum() / denominator),
            'gap_change_ci95': np.quantile(changes, [.025, .975]).tolist(),
            'uncertainty': 'Paired document bootstrap on these fixed 11 tokenizers; not uncertainty over all possible tokenizers.',
            'limitations': 'Unseen tokenizer files, all 48k BPE/line256; tokenizer-training corpora may overlap the pool source corpus. Does not test all three algorithms or prove broad generalization.'}
