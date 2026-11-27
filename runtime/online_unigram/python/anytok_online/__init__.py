"""A resident, corpus-conditioned generator of byte-level Unigram tokenizers."""

import gzip
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers

from .anytok_online_native import Corpus

DOMAINS = {
    "web": tuple(
        f"web-{lang}" for lang in ("en", "es", "fr", "ru", "zh", "ja", "ar", "hi")
    ),
    "wikipedia": tuple(
        f"wiki-{lang}" for lang in ("en", "es", "fr", "ru", "zh", "ja", "ar", "hi")
    ),
    "code": ("code",),
    "papers": ("papers-en",),
    "math": ("math-en",),
    "qa": ("qa-en",),
    "books": ("books-en",),
    "medical": ("medical-en",),
    "legal": ("legal-en",),
}


def allocate(total, weights):
    if any(not math.isfinite(w) or w < 0 for w in weights) or max(weights) <= 0:
        raise ValueError("Expected finite nonnegative domain weights with positive sum")
    # Follow pool-v2's exact arithmetic order for saved domain and bucket
    # quotas. Rescale extreme floating-point policy outputs before multiplying.
    if not math.isfinite(sum(weights)) or not math.isfinite(max(weights) * total):
        scale = max(weights)
        weights = [w / scale for w in weights]
    norm = sum(weights)
    shares = [total * w / norm for w in weights]
    amounts = [math.floor(x) for x in shares]
    order = sorted(
        range(len(shares)), key=lambda i: shares[i] - amounts[i], reverse=True
    )
    for i in order[: total - sum(amounts)]:
        amounts[i] += 1
    return amounts


def read_prefix(path, capacity):
    data = bytearray()
    ends = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            text = json.loads(line)["text"]
            part = text.encode("utf-8")
            room = capacity - len(data)
            if len(part) > room:
                part = part[:room].decode("utf-8", errors="ignore").encode("utf-8")
                if part:
                    data.extend(part)
                    ends.append(len(data))
                break
            if part:
                data.extend(part)
                ends.append(len(data))
            if len(data) == capacity:
                break
        else:
            # Some distributed bundles contain a shorter prefix than the
            # source recipe. Expose the quota available on this installation.
            capacity = len(data)
    return bytes(data), ends, capacity


@dataclass
class Generated:
    tokenizer: Tokenizer
    vocabulary: list
    bytes_used: int
    sampled_starts_with_two_bytes: int
    eligible_vocab_size: int
    bucket_quotas: dict[str, int]

    def training_inputs(self, source_tokenizer, fallback_ids, build_surface_forms):
        """Return the same native tokenizer, surface-form and prior layout used by the student."""
        forms, truncated = build_surface_forms(
            [piece for piece, _ in self.vocabulary],
            source_tokenizer,
            10,
            fallback_ids,
            {"<|endoftext|>": 0},
        )
        return (
            self.tokenizer,
            forms,
            np.asarray([score for _, score in self.vocabulary], dtype=np.float32),
            truncated,
        )


class Generator:
    def __init__(self, corpus_dir, recipe, *, corpus_bytes=200_000_000, progress=None):
        if not isinstance(corpus_bytes, int) or not 0 < corpus_bytes < 2**32 - 1:
            raise ValueError("Expected a positive corpus byte budget below 2**32 - 1")
        self.corpus_bytes = corpus_bytes
        self.native = Corpus()
        self.names = []
        self.within = {}
        self.capacity = {}
        sources = {r["name"]: r for r in recipe["buckets"]}
        for domain, names in DOMAINS.items():
            weights = [sources[n]["train_target_bytes"] for n in names]
            self.within[domain] = weights
            for name in names:
                capacity = min(sources[name]["train_available_bytes"], corpus_bytes)
                data, ends, self.capacity[name] = read_prefix(
                    Path(corpus_dir) / name / "train.jsonl.gz", capacity
                )
                self.native.add_bucket(data, ends)
                self.names.append(name)
                if progress:
                    progress(name, len(data))

    def sample(
        self,
        mixture,
        *,
        vocab_size,
        language_mix=None,
        pretokenizer="regex",
        max_piece_bytes=16,
        stride=32,
        length_bias=1.0,
        score_noise=0.0,
        seed=0,
    ):
        """Weight languages independently inside web and wikipedia, e.g. {"web": {"ja": 1}}."""
        if set(mixture) - DOMAINS.keys():
            raise ValueError("Unknown mixture domain")
        if language_mix is None:
            language_mix = {}
        if not isinstance(language_mix, Mapping) or language_mix.keys() - {
            "web",
            "wikipedia",
        }:
            raise ValueError("Language weights are supported for web and wikipedia")
        domain_bytes = allocate(
            self.corpus_bytes, [mixture.get(d, 0.0) for d in DOMAINS]
        )
        quotas = {}
        for domain, amount in zip(DOMAINS, domain_bytes):
            names = DOMAINS[domain]
            weights = self.within[domain]
            if domain in language_mix:
                shares = language_mix[domain]
                languages = [name.split("-")[1] for name in names]
                if not isinstance(shares, Mapping) or shares.keys() - set(languages):
                    raise ValueError(
                        f"Expected weights for these {domain} languages: {', '.join(languages)}"
                    )
                weights = [shares.get(lang, 0.0) for lang in languages]
            for name, quota in zip(names, allocate(amount, weights)):
                if quota > self.capacity[name]:
                    raise ValueError(
                        f"{name} needs {quota:,} bytes; this corpus provides {self.capacity[name]:,}"
                    )
                quotas[name] = quota
        pieces, used, starts, eligible = self.native.sample(
            [quotas[name] for name in self.names],
            pretokenizer,
            vocab_size,
            max_piece_bytes,
            stride,
            length_bias,
            score_noise,
            seed,
        )
        tokenizer = Tokenizer(models.Unigram(pieces, unk_id=0))
        bytelevel = pre_tokenizers.ByteLevel(
            add_prefix_space=False, use_regex=pretokenizer == "regex"
        )
        if pretokenizer == "line256":
            tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
                [
                    pre_tokenizers.Split(
                        Regex(r"[^\n]{1,256}\n?|\n"), behavior="isolated"
                    ),
                    bytelevel,
                ]
            )
        else:
            tokenizer.pre_tokenizer = bytelevel
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.add_special_tokens(["<|unk|>", "<|endoftext|>"])
        return Generated(tokenizer, pieces, used, starts, eligible, quotas)
