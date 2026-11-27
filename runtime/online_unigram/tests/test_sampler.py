"""Compare native candidate extraction with exhaustive counts on a small corpus."""

import gzip
import json
import math
from collections import Counter, defaultdict

import pytest
from tokenizers import Regex, pre_tokenizers

online = pytest.importorskip(
    "anytok_online",
    reason="Install ./runtime/online_unigram to test its native extension",
)
Corpus = online.Corpus


@pytest.mark.parametrize("mode", ["regex", "line256", "no-regex"])
def test_candidates_at_document_and_quota_boundaries(mode):
    docs = ["don't walk\nwalk、歩く\n", "walk.. walk 42" * 23, "tail猫tail"]
    raw = [doc.encode() for doc in docs]
    total = sum(map(len, raw))
    corpus = Corpus()
    corpus.add_bucket(b"".join(raw), [len(raw[0]), total - len(raw[2]), total])
    # End one byte inside 猫. The sampler must stop before that character.
    quota = total - len(raw[2]) + 5
    cuts = docs[:2] + ["tail"]
    limit = 8
    bytelevel = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=mode == "regex"
    )
    split = (
        pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(r"[^\n]{1,256}\n?|\n"), behavior="isolated"),
                bytelevel,
            ]
        )
        if mode == "line256"
        else bytelevel
    )
    singles = Counter()
    counts = Counter()
    followers = defaultdict(set)
    leaves = set()
    for doc in cuts:
        for part, _ in split.pre_tokenize_str(doc):
            for start, symbol in enumerate(part):
                singles[symbol] += 1
                suffix = part[start : start + limit]
                if len(suffix) >= 2:
                    leaves.add(suffix)
                for end in range(2, len(suffix) + 1):
                    piece = suffix[:end]
                    counts[piece] += 1
                    if end < len(suffix):
                        followers[piece].add(suffix[end])
    expected = leaves | {piece for piece, after in followers.items() if len(after) > 1}
    vocab, used, starts, eligible = corpus.sample(
        [quota], mode, 258 + len(expected), limit, 1, 0.0, 0.0, 39
    )
    assert used == sum(len(d.encode()) for d in cuts)
    assert starts == sum(counts[p] for p in counts if len(p) == 2)
    assert eligible == len({w for w, _ in vocab}) == 258 + len(expected)
    assert [p for p, _ in vocab[:2]] == ["<|unk|>", "<|endoftext|>"]
    assert {p for p, _ in vocab[2:258]} == set(pre_tokenizers.ByteLevel.alphabet())
    assert {p for p, _ in vocab[258:]} == expected
    weights = {p: max(1, singles[p]) for p, _ in vocab[2:258]}
    weights.update({p: counts[p] for p in expected})
    denominator = sum(weights.values())
    for piece, score in vocab[2:]:
        assert abs(score - math.log(weights[piece] / denominator)) <= 0.5 / 4096 + 1e-12
    with pytest.raises(ValueError, match="eligible pieces"):
        corpus.sample([quota], mode, eligible + 1, limit, 1, 0.0, 0.0, 39)


def test_byte_only_tokenizer_and_invalid_scores():
    corpus = Corpus()
    corpus.add_bucket(b"aaaa", [4])
    vocab, used, starts, available = corpus.sample([4], "regex", 258, 1, 1, 0.0, 0.0, 0)
    assert len(vocab) == available == 258 and used == 4 and starts == 0
    with pytest.raises(ValueError, match="float32"):
        corpus.sample([4], "regex", 259, 3, 1, 1e300, 0.0, 0)


def test_original_three_byte_quota_tolerance_with_tiny_documents():
    corpus = Corpus()
    corpus.add_bucket(b"abcdxyzw", [4, 5, 7, 8])
    assert corpus.sample([7], "regex", 258, 1, 1, 0.0, 0.0, 0)[1] == 4


def test_independent_language_quotas_and_source_limits(tmp_path):
    recipe = {"buckets": []}
    for names in online.DOMAINS.values():
        for name in names:
            path = tmp_path / name
            path.mkdir()
            with gzip.open(path / "train.jsonl.gz", "wt") as stream:
                stream.write(
                    json.dumps({"text": "x" * (64 if name == "wiki-ja" else 128)})
                    + "\n"
                )
            recipe["buckets"].append(
                {"name": name, "train_target_bytes": 1, "train_available_bytes": 200}
            )

    gen = online.Generator(tmp_path, recipe, corpus_bytes=128)
    mix = {"web": 0.5, "wikipedia": 0.5}
    default = gen.sample(mix, vocab_size=258)
    assert default.bucket_quotas["wiki-en"] == default.bucket_quotas["web-ja"] == 8
    web = gen.sample(mix, vocab_size=258, language_mix={"web": {"ja": 1}})
    assert web.bucket_quotas["web-ja"] == 64 and web.bucket_quotas["wiki-en"] == 8
    sample = gen.sample(
        mix,
        vocab_size=258,
        language_mix={"web": {"ja": 3, "es": 1}, "wikipedia": {"ja": 1}},
    )
    assert {k: v for k, v in sample.bucket_quotas.items() if v} == {
        "web-ja": 48,
        "web-es": 16,
        "wiki-ja": 64,
    }
    assert sample.bytes_used == 128
    with pytest.raises(ValueError, match="wiki-ja needs 128 bytes"):
        gen.sample(
            {"wikipedia": 1}, vocab_size=258, language_mix={"wikipedia": {"ja": 1}}
        )
    with pytest.raises(ValueError, match="languages"):
        gen.sample(mix, vocab_size=258, language_mix={"web": {"xx": 1}})
    with pytest.raises(ValueError, match="positive"):
        gen.sample(mix, vocab_size=258, language_mix={"web": {"ja": 0}})
