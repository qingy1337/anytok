"""Uniform sampling of immutable saved tokenizers, with bounded local caches."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from .tokenizer_inputs import build_surface_forms


class TokenizerPool:
    def __init__(self, directory):
        self.root = Path(directory)
        raw = (self.root / 'full-results.json').read_bytes()
        index = json.loads(raw)
        self.index_hash = hashlib.sha256(raw).hexdigest()
        self.plan_hash = index['plan_sha256']
        plan = json.loads((self.root / 'plan.json').read_text())
        self.entries = sorted(index['tokenizers'], key=lambda m: m['config']['index'])
        if (index['completed'] != 1000 or len(self.entries) != 1000
                or [m['config']['index'] for m in self.entries] != list(range(1000))
                or self.plan_hash != plan['plan_sha256']
                or any(m['status'] != 'complete' or m['config'] != plan['configs'][i]
                       or m['vocab_size_actual'] != m['config']['vocab_size']
                       for i, m in enumerate(self.entries))):
            raise ValueError('Expected all 1000 matching saved tokenizer configurations')
        self._forms = {}

    @lru_cache(maxsize=16)
    def load(self, index):
        manifest = self.entries[index]
        path = self.root / 'tokenizers' / f'{index:04d}' / 'tokenizer.json'
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest['tokenizer_sha256']:
            raise ValueError(f'Tokenizer checksum mismatch: {index}')
        tokenizer = Tokenizer.from_str(raw.decode())
        tokens = tokenizer.get_vocab()
        size = manifest['vocab_size_actual']
        if len(tokens) != size or set(tokens.values()) != set(range(size)):
            raise ValueError(f'Non-contiguous tokenizer vocabulary: {index}')
        model = json.loads(raw)['model']
        # Match existing BPE transfer evaluation. Unigram's learned scores
        # remain part of its tokenizer; BPE and WordPiece have no such scores.
        scores = [float(row[1]) for row in model['vocab']] if model['type'] == 'Unigram' else [0.] * size
        ordered = [None] * size
        for token, token_id in tokens.items():
            ordered[token_id] = (token, scores[token_id])
        if not np.isfinite(scores).all() or tokenizer.token_to_id('<|endoftext|>') is None:
            raise ValueError(f'Invalid tokenizer priors or missing EOS: {index}')
        return tokenizer, ordered

    def forms(self, index, source_tokenizer, fallback):
        # At most ~5 GiB for the full pool; reuse forms after the first visit.
        # Tokenizer parser objects use a separate bounded cache above.
        if index not in self._forms:
            _, vocab = self.load(index)
            self._forms[index] = build_surface_forms(
                [piece for piece, _ in vocab], source_tokenizer, 10, fallback,
                {'<|endoftext|>': 0})
        return self._forms[index]
