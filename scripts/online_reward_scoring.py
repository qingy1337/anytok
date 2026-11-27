"""The frozen search's paired batch and held-out scoring protocol."""

from collections import defaultdict
import gzip
import hashlib
import json
import math

import torch

from natural_transfer import score_rows
from package_training_inputs import sha256
from pool_generalization import digest as panel_digest


BATCH_SIZE = 128
MICROBATCH_SIZE = 32
TRIMMED_PANEL_SHA = 'c09a49746a69f40dcf04f3cf66fe7af27631da0b3941010821085bd5a14806e5'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class SharedBatch:
    """Share every original window and the bucket-specific replacement lists."""

    def __init__(self, corpus, rng):
        self.corpus = corpus
        self.extras = defaultdict(list)
        names = rng.choices(corpus.names, weights=corpus.weights, k=BATCH_SIZE)
        self.windows = [(name, self.next_text(name)) for name in names]
        self.sha256 = digest(self.windows)

    def next_text(self, name):
        return self.corpus.next_text(name, chars=2048)[:512]

    def replacement(self, name, index):
        bank = self.extras[name]
        if index == len(bank):
            bank.append(self.next_text(name))
        return bank[index]


def paired_batch(trainer, canonical, native, shared):
    examples, controls, proofs, prefix_bytes = [], [], [], []
    replaced, next_replacement = 0, defaultdict(int)
    for name, raw in shared.windows:
        while True:
            pair = trainer.make_paired_prefix_example(raw, native, canonical)
            if pair is not None and not {0, 1}.intersection(pair[0][0]):
                break
            replaced += 1
            if replaced > BATCH_SIZE:
                raise ValueError('Paired training-example rejection limit exceeded')
            raw = shared.replacement(name, next_replacement[name])
            next_replacement[name] += 1
        example, control, proof = pair
        if example[1:] != control[1:] or proof['maximum_positions'] > 128:
            raise ValueError('The two inputs do not score the same continuation')
        prefix = canonical.decode(control[0], clean_up_tokenization_spaces=False)
        prefix_bytes.append(len(prefix.encode()))
        proofs.append(dict(proof, raw_sha256=hashlib.sha256(raw.encode()).hexdigest()))
        examples.append(example)
        controls.append(control)
    return examples, controls, proofs, replaced, prefix_bytes


def heldout_panel(inputs, canonical, replay_path):
    replay = json.loads(replay_path.read_text())
    passages = replay['passages']
    reconstructed = {}
    for domain, file in replay['files'].items():
        path = inputs / 'eval' / f'{domain}.jsonl.gz'
        if path.stat().st_size != file['bytes'] or sha256(path) != file['sha256']:
            raise ValueError(f'Held-out corpus file changed: {domain}')
        requested = [row for row in passages if row['domain'] == domain]
        wanted = {row['source_id'] for row in requested}
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            docs = {record['id']: record['text'] for record in map(json.loads, stream) if record['id'] in wanted}
        for row in requested:
            source = docs[row['source_id']]
            if hashlib.sha256(source.encode()).hexdigest() != row['document_sha256']:
                raise ValueError(f'Source document changed: {row["id"]}')
            cut = row['start'] + row['prefix_chars']
            prefix = source[row['start']:cut]
            suffix = source[cut:cut + row['continuation_chars']]
            ids = canonical(prefix, add_special_tokens=False).input_ids
            targets = canonical(suffix, add_special_tokens=False).input_ids
            if not ids or 0 in ids or len(ids) > 95 or len(targets) != 32 or len(suffix.encode()) != row['continuation_bytes']:
                raise ValueError(f'Invalid held-out input: {row["id"]}')
            reconstructed[row['id']] = dict(id=row['id'], domain=domain, prefix=prefix, canonical_prefix_ids=ids,
                                            canonical_continuation_ids=targets, continuation_bytes=row['continuation_bytes'])
    panel = [reconstructed[row['id']] for row in passages]
    if panel_digest(panel) != TRIMMED_PANEL_SHA or (len(panel), sum(r['continuation_bytes'] for r in panel)) != (156, 14429):
        raise ValueError('Held-out prefixes differ from the direct frozen run')
    return panel


def domain_bpb(rows, losses):
    domains = defaultdict(list)
    for row, scores in zip(rows, losses, strict=True):
        domains[row['domain']].append((math.fsum(scores), row['continuation_bytes']))
    return {domain: math.fsum(n for n, _ in values) / (math.log(2) * sum(b for _, b in values))
            for domain, values in sorted(domains.items())}


def confirm(scorer, panel, settings):
    sample = scorer.generator.sample(**settings)
    native, forms, priors, _ = sample.training_inputs(scorer.canonical, scorer.fallback, scorer.trainer.build_surface_forms)
    lengths = defaultdict(list)
    for row in panel:
        prefix = row['prefix']
        ids = native.encode(prefix, add_special_tokens=False).ids
        if not ids or len(ids) > 95 or {0, 1}.intersection(ids) or native.decode(ids, skip_special_tokens=False) != prefix:
            raise ValueError(f'Invalid generated-tokenizer prefix for {row["id"]}')
        lengths[row['domain']].append((len(ids), len(prefix.encode())))
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        table, output, _ = scorer.hn(torch.as_tensor(forms, device='cuda', dtype=torch.long),
                                     torch.as_tensor(priors, device='cuda'), scorer.source, None, deterministic=True)
        if output is not None or table.shape != (len(forms), scorer.source.shape[1]) or not torch.isfinite(table).all():
            raise ValueError('Invalid embedding table in held-out confirmation')
    losses = score_rows(scorer.lm, table, scorer.source, panel, native)
    ratios = {d: sum(n for n, _ in xs) / sum(b for _, b in xs) for d, xs in lengths.items()}
    return dict(domain_bpb=domain_bpb(panel, losses), tokens_per_prefix_byte=ratios,
                losses_nats=[math.fsum(row) for row in losses],
                tokenizer_sha256=hashlib.sha256(native.to_str().encode()).hexdigest())
