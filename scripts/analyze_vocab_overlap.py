"""Measure 200 MB tokenizer vocabulary overlap and use of unique domain tokens."""

from __future__ import annotations

from collections import Counter
import gzip
import json
from pathlib import Path

from tokenizers import Tokenizer

from generate_mixture_examples import DOMAIN_BUCKETS


ROOT = Path(__file__).resolve().parent.parent
FOCI = {
    "papers": "Papers25", "math": "Math50", "qa": "QA25",
    "books": "Books25", "medical": "Medical40", "legal": "Legal40",
}
LABELS = ("Base", *FOCI.values())
COLLECTIONS = {"regex": "tokenizers-regex200", "cross_word": "tokenizers-line200"}


def model_path(rule: str, label: str) -> Path:
    return ROOT / "modal_results" / COLLECTIONS[rule] / label.lower() / "tokenizer.json"


def overlap(first: dict[str, int], second: dict[str, int]) -> dict:
    shared = first.keys() & second.keys()
    size = len(first)
    return {
        "shared": len(shared),
        "first_only": size - len(shared),
        "second_only": len(second) - len(shared),
        "shared_percent_of_vocab": 100 * len(shared) / size,
        "jaccard_percent": 100 * len(shared) / len(first.keys() | second.keys()),
        "first_top_10000_in_second_percent": 100 * sum(
            token in second for token, rank in first.items() if rank < 10_000
        ) / 10_000,
    }


def visible_examples(model: Tokenizer, counts: Counter[int], other_vocab: dict[str, int]) -> list[dict]:
    examples = []
    for token_id, count in counts.most_common():
        token = model.id_to_token(token_id)
        if token in other_vocab:
            continue
        decoded = model.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in decoded or len(decoded.strip()) < 3 or not decoded.isprintable():
            continue
        examples.append({"text": decoded, "held_out_occurrences": count, "rank": token_id})
        if len(examples) == 12:
            break
    return examples


def main() -> None:
    vocab = {}
    for rule in COLLECTIONS:
        for label in LABELS:
            model = json.loads(model_path(rule, label).read_text())
            vocab[(rule, label)] = model["model"]["vocab"]
            if len(vocab[(rule, label)]) != 150_000:
                raise ValueError(f"Unexpected vocabulary size: {rule} {label}")

    base_rule = overlap(vocab[("regex", "Base")], vocab[("cross_word", "Base")])
    pairs = {
        rule: {label: overlap(vocab[(rule, "Base")], vocab[(rule, label)])
               for label in FOCI.values()}
        for rule in COLLECTIONS
    }
    cross_rule = {
        label: overlap(vocab[("regex", label)], vocab[("cross_word", label)])
        for label in LABELS
    }
    common_cross = set.intersection(*(set(vocab[("cross_word", label)]) for label in LABELS))
    common_regex = set.intersection(*(set(vocab[("regex", label)]) for label in LABELS))

    domain_use = {}
    for domain, label in FOCI.items():
        model = Tokenizer.from_file(str(model_path("cross_word", label)))
        counts: Counter[int] = Counter()
        for bucket in DOMAIN_BUCKETS[domain]:
            with gzip.open(ROOT / "modal_results/eval" / f"{bucket}.jsonl.gz", "rt",
                           encoding="utf-8") as handle:
                texts = [json.loads(line)["text"] for line in handle]
            for encoded in model.encode_batch(texts, add_special_tokens=False):
                counts.update(encoded.ids)
        base_vocab = vocab[("cross_word", "Base")]
        unique_count = sum(count for token_id, count in counts.items()
                           if model.id_to_token(token_id) not in base_vocab)
        domain_use[domain] = {
            "model": label,
            "held_out_token_occurrences": counts.total(),
            "occurrences_from_tokens_absent_in_cross_word_base": unique_count,
            "unique_occurrence_percent": 100 * unique_count / counts.total(),
            "examples": visible_examples(model, counts, base_vocab),
        }
        print(f"{domain:8} {pairs['cross_word'][label]['shared_percent_of_vocab']:5.1f}% "
              f"vocab shared with cross-word Base; "
              f"{domain_use[domain]['unique_occurrence_percent']:5.1f}% "
              "of held-out token occurrences use tokens absent from Base")

    output = {
        "method": "Exact token-string overlap in matched 200 MB / 150k BPE models. "
                  "Domain use is counted on held-out documents encoded by each specialist.",
        "vocab_size": 150_000,
        "regex_base_vs_cross_word_base": base_rule,
        "base_vs_specialist": pairs,
        "regex_vs_cross_word_same_mixture": cross_rule,
        "shared_across_all_seven": {"regex": len(common_regex), "cross_word": len(common_cross)},
        "focused_domain_use": domain_use,
    }
    path = ROOT / "results/vocab_overlap_200m.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
