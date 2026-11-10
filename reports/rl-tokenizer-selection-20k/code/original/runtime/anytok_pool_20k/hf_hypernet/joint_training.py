"""Data and loss helpers for input-only joint tokenizer training.

The tokenizer sampler is ZeTT's Rust implementation. The LM always predicts
SmolLM2 tokens; only the prefix is encoded with the sampled tokenizer.
"""

import gzip
import json
import math
from pathlib import Path

import numpy as np
import rust_utils
import tokenizers
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers


SPLIT_REGEX = (
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}\p{M}]+| ?\p{N}+| "
    r"?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)
VOCAB_SIZE = 32_768
INITIAL_QUEUE_BATCHES = 16
QUEUE_BATCH_SIZE = 128


def zett_main_learning_rate(update: int, peak: float, main_updates: int = 190_000,
                            ramp_updates: int = 10_000) -> float:
    """ZeTT's post-identity linear ramp and cosine decay (floor 10% of peak)."""
    if not 0 <= update <= main_updates or not 0 < ramp_updates < main_updates:
        raise ValueError("Update or schedule length is out of range")
    if update < ramp_updates:
        return peak * update / ramp_updates
    fraction = (update - ramp_updates) / (main_updates - ramp_updates)
    return peak * (0.1 + 0.9 * (1.0 + math.cos(math.pi * fraction)) / 2.0)


class BalancedTextStream:
    """Read the pinned corpus in recipe proportions without loading it into RAM."""

    def __init__(self, recipe_path: Path, data_dir: Path, state: dict = None):
        self.recipe = json.loads(Path(recipe_path).read_text())
        if self.recipe["name"] != "balanced-full-v3":
            raise ValueError("Expected the pinned balanced-full-v3 corpus")
        self.names = [bucket["name"] for bucket in self.recipe["buckets"]]
        self.weights = [bucket["train_target_bytes"] for bucket in self.recipe["buckets"]]
        self.quotas = dict(zip(self.names, self.weights))
        self.paths = {name: Path(data_dir) / name / "train.jsonl.gz" for name in self.names}
        missing = [str(path) for path in self.paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing training files: {missing}")
        self.handles = {name: gzip.open(path, "rt", encoding="utf-8")
                        for name, path in self.paths.items()}
        self.cursors = {name: 0 for name in self.names}
        self.used_bytes = {name: 0 for name in self.names}
        self.epochs = {name: 0 for name in self.names}
        self.pending = {name: "" for name in self.names}
        self.pending_pos = {name: 0 for name in self.names}
        if state is not None:
            if set(state["cursors"]) != set(self.names):
                raise ValueError("Stream state does not match corpus buckets")
            for key in ("cursors", "used_bytes", "epochs", "pending", "pending_pos"):
                getattr(self, key).update(state[key])
            for name, handle in self.handles.items():
                handle.seek(self.cursors[name])

    def state_dict(self) -> dict:
        return {key: dict(getattr(self, key)) for key in
                ("cursors", "used_bytes", "epochs", "pending", "pending_pos")}

    def close(self):
        for handle in self.handles.values():
            handle.close()

    def _next_document(self, name: str) -> str:
        handle = self.handles[name]
        if self.used_bytes[name] >= self.quotas[name]:
            handle.seek(0)
            self.used_bytes[name] = 0
            self.epochs[name] += 1
        line = handle.readline()
        if not line:
            handle.seek(0)
            self.used_bytes[name] = 0
            self.epochs[name] += 1
            line = handle.readline()
            if not line:
                raise ValueError(f"Empty training file for {name}")
        self.cursors[name] = handle.tell()
        text = json.loads(line)["text"]
        remaining = self.quotas[name] - self.used_bytes[name]
        encoded = text.encode("utf-8")
        if len(encoded) > remaining:
            encoded = encoded[:remaining]
            text = encoded.decode("utf-8", errors="ignore")
        self.used_bytes[name] += len(encoded)
        return text

    def next_text(self, name: str, chars: int = 512, min_chars: int = 64) -> str:
        if name not in self.handles:
            raise KeyError(name)
        for _ in range(1000):
            text = self.pending[name]
            pos = self.pending_pos[name]
            if len(text) - pos < min_chars:
                self.pending[name] = self._next_document(name)
                self.pending_pos[name] = 0
                continue
            result = text[pos:pos + chars]
            self.pending_pos[name] = pos + chars
            return result
        raise RuntimeError(f"Could not find usable text in {name}")


def prime_sampler(sampler, queue_batches: list[list[str]]) -> None:
    """Rebuild ZeTT's rolling queue from saved text batches."""
    for batch in queue_batches:
        sampled = sampler.sample_tokenizer(
            {text: 1 for text in batch}, 30_000, 16, 4, 0.0, False, True
        )
        if sampled:
            raise AssertionError("Queue priming unexpectedly emitted a tokenizer")


def draw_tokenizer(sampler, texts: list[str], noise_rng: np.random.Generator,
                   eos_token: str) -> tuple[Tokenizer, list[tuple[str, float]], float]:
    """Sample one 32k Unigram tokenizer from the Rust queue and current batch."""
    noise = float(noise_rng.lognormal(mean=math.log(1e-5), sigma=4.0))
    sampled = sampler.sample_tokenizer(
        {text: 1 for text in texts}, VOCAB_SIZE, 16, 4, noise, True, True
    )
    # ZeTT inserts the original special token after sampling. Reserve its ID
    # here so the final input vocabulary has exactly 32,768 rows.
    pieces = [row for row in sampled if row[0] != eos_token][:VOCAB_SIZE - 1]
    if len(pieces) != VOCAB_SIZE - 1 or len({row[0] for row in pieces}) != len(pieces):
        raise ValueError("Sampler did not produce a unique 32k vocabulary")
    vocab = [(eos_token, 0.0), *pieces]
    tokenizer = Tokenizer(models.Unigram(vocab))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(tokenizers.Regex(SPLIT_REGEX), "removed", invert=True),
        pre_tokenizers.ByteLevel(False, False),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer, vocab, noise


def make_split_prefix_example(raw: str, target_tokenizer: Tokenizer, source_tokenizer,
                              max_positions: int = 128, min_chars: int = 64):
    """Encode a new-tokenizer prefix followed by canonical output tokens."""
    if len(raw) < min_chars:
        return None
    for _ in range(32):
        cut = len(raw) // 2
        prefix, continuation = raw[:cut], raw[cut:]
        prefix_ids = target_tokenizer.encode(prefix, add_special_tokens=False).ids
        continuation_ids = source_tokenizer(
            continuation, add_special_tokens=False
        ).input_ids
        if 1 + len(prefix_ids) + len(continuation_ids) <= max_positions:
            if (prefix_ids and continuation_ids
                and target_tokenizer.decode(prefix_ids) == prefix
                and source_tokenizer.decode(continuation_ids,
                                            clean_up_tokenization_spaces=False) == continuation):
                return prefix_ids, continuation_ids, len(continuation.encode("utf-8"))
            return None
        raw = raw[:int(len(raw) * 0.85)]
        if len(raw) < min_chars:
            return None
    return None


def make_lm_microbatch(examples, generated, source, device):
    """Make padded inputs and labels for the fixed canonical output head."""
    sequences, masks, labels = [], [], []
    for prefix_ids, continuation_ids, _ in examples:
        prefix = torch.tensor(prefix_ids, device=device, dtype=torch.long)
        continuation = torch.tensor(continuation_ids, device=device, dtype=torch.long)
        sequence = torch.cat([
            source[0:1].to(torch.bfloat16),
            generated[prefix].to(torch.bfloat16),
            source[continuation].to(torch.bfloat16),
        ])
        sequences.append(sequence)
        masks.append(torch.ones(len(sequence), device=device, dtype=torch.long))
        labels.append(torch.cat([
            torch.full((1 + len(prefix),), -100, device=device, dtype=torch.long),
            continuation,
        ]))
    embeds = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    mask = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True)
    targets = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )
    positions = torch.arange(embeds.shape[1], device=device)[None].expand(len(examples), -1)
    return embeds, mask, targets, positions


def input_lexical_loss(generated, surface_forms, source, pad_id: int,
                       eos_id: int) -> torch.Tensor:
    """ZeTT's overlapping-token MSE penalty, adapted to input embeddings."""
    first_ids = surface_forms[:, 0]
    overlap = ((surface_forms[:, 1:] == pad_id).all(dim=1)
               & (first_ids < source.shape[0]) & (first_ids != eos_id))
    if not overlap.any():
        raise ValueError("No original tokens overlap the sampled vocabulary")
    target = source[first_ids[overlap]].float()
    predicted = generated[overlap].float()
    return (predicted - target).square().sum(-1).mean() / target.norm(dim=1).mean()
