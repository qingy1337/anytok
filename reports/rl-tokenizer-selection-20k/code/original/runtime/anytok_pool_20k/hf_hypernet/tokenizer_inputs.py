"""Represent byte-level target tokens for a source-embedding hypernetwork.

ZeTT normally splits each target token using the source tokenizer's BPE model.
Some source models, including SmolLM2, lack individual byte symbols. Those
symbols get distinct trainable fallback embeddings instead of an empty form.
"""

import numpy as np
from tokenizers import pre_tokenizers


def missing_byte_symbols(source_tokenizer) -> list[str]:
    """Find byte alphabet entries the source BPE model cannot represent."""
    model = source_tokenizer._tokenizer.model
    return [
        symbol for symbol in sorted(pre_tokenizers.ByteLevel.alphabet())
        if "".join(token.value for token in model.tokenize(symbol)) != symbol
    ]


def enable_byte_fallbacks(hypernet, source_tokenizer) -> dict[str, int]:
    """Extend a warmup checkpoint once; keep byte-to-row IDs in its config."""
    source_size = len(source_tokenizer)
    if hypernet.config.original_vocab_size != source_size:
        raise ValueError("Source tokenizer size differs from the hypernetwork")
    if hypernet.config.pad_token_id != source_size:
        raise ValueError("Expected the dedicated pad ID after the source vocabulary")
    symbols = missing_byte_symbols(source_tokenizer)
    saved = hypernet.config.hn_byte_fallback_symbols
    if saved is None:
        if hypernet.fallback_embeddings.num_embeddings != 1:
            raise ValueError("Unrecorded fallback embeddings already exist")
        hypernet.expand_fallback_embeddings(1 + len(symbols))
        hypernet.config.hn_byte_fallback_symbols = symbols
    elif saved != symbols:
        raise ValueError("Source tokenizer byte alphabet differs from checkpoint")
    elif hypernet.fallback_embeddings.num_embeddings != 1 + len(symbols):
        raise ValueError("Checkpoint byte fallback rows do not match its mapping")
    return {symbol: source_size + 1 + i for i, symbol in enumerate(symbols)}


def build_surface_forms(
    target_tokens: list[str], source_tokenizer, max_pieces: int,
    byte_fallback_ids: dict[str, int], special_token_ids: dict[str, int] = None,
) -> tuple[np.ndarray, int]:
    """Make ZeTT's source-piece matrix, preserving previously missing bytes."""
    if max_pieces < 1:
        raise ValueError("max_pieces must be positive")
    source_size = len(source_tokenizer)
    pad_id = source_size
    matrix = np.full((len(target_tokens), max_pieces), pad_id, dtype=np.int32)
    truncated = 0
    model = source_tokenizer._tokenizer.model
    special_token_ids = special_token_ids or {}

    def tokenize_exact(text: str) -> list[int]:
        pieces = model.tokenize(text)
        if "".join(piece.value for piece in pieces) != text:
            raise ValueError(f"Source tokenizer cannot represent target piece {text!r}")
        return [piece.id for piece in pieces]

    for row, token in enumerate(target_tokens):
        if token in special_token_ids:
            ids = [special_token_ids[token]]
        else:
            pieces = model.tokenize(token)
            if "".join(piece.value for piece in pieces) == token:
                ids = [piece.id for piece in pieces]
            else:
                ids = []
                segment_start = 0
                for pos, symbol in enumerate(token):
                    if symbol in byte_fallback_ids:
                        if pos > segment_start:
                            ids.extend(tokenize_exact(token[segment_start:pos]))
                        ids.append(byte_fallback_ids[symbol])
                        segment_start = pos + 1
                if segment_start < len(token):
                    ids.extend(tokenize_exact(token[segment_start:]))
                if not ids:
                    raise ValueError(f"Source tokenizer cannot represent target piece {token!r}")
        if not ids:
            raise ValueError(f"Target token {token!r} has an empty source form")
        if len(ids) > max_pieces:
            truncated += 1
        matrix[row, :min(len(ids), max_pieces)] = ids[:max_pieces]
    return matrix, truncated
