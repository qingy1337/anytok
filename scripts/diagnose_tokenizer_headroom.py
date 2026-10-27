"""Quantify the token-count floor imposed by the shared BPE pretokenizer."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parent.parent
FOCI = {
    "code": ("code", "modal_results/tokenizers-code50/c/tokenizer.json"),
    "papers": ("papers-en", "modal_results/tokenizers-heavy/papers25/tokenizer.json"),
    "math": ("math-en", "modal_results/tokenizers-heavy/math50/tokenizer.json"),
    "qa": ("qa-en", "modal_results/tokenizers-heavy/qa25/tokenizer.json"),
    "books": ("books-en", "modal_results/tokenizers-heavy/books25/tokenizer.json"),
    "medical": ("medical-en", "modal_results/tokenizers-heavy/medical40/tokenizer.json"),
    "legal": ("legal-en", "modal_results/tokenizers-heavy/legal40/tokenizer.json"),
}


def main() -> None:
    baseline = Tokenizer.from_file(str(ROOT / "modal_results/tokenizers-150k/base/tokenizer.json"))
    baseline_vocab = set(baseline.get_vocab())
    domains = {}
    for domain, (bucket, model_path) in FOCI.items():
        focused = Tokenizer.from_file(str(ROOT / model_path))
        with gzip.open(ROOT / "modal_results/eval" / f"{bucket}.jsonl.gz", "rt",
                       encoding="utf-8") as handle:
            texts = [json.loads(line)["text"] for line in handle]
        pieces = sum(len(baseline.pre_tokenizer.pre_tokenize_str(text)) for text in texts)
        base_count = sum(len(encoded.ids) for encoded in baseline.encode_batch(
            texts, add_special_tokens=False))
        focus_count = sum(len(encoded.ids) for encoded in focused.encode_batch(
            texts, add_special_tokens=False))
        if not pieces <= min(base_count, focus_count):
            raise AssertionError(f"Pretokenizer floor exceeds BPE count for {domain}")
        shared = len(baseline_vocab & set(focused.get_vocab()))
        domains[domain] = {
            "bucket": bucket, "documents": len(texts),
            "pretokenizer_pieces": pieces,
            "base_tokens": base_count, "focused_tokens": focus_count,
            "base_above_floor_tokens": base_count - pieces,
            "max_possible_reduction_percent_with_current_pretokenizer":
                (base_count - pieces) / base_count * 100,
            "actual_reduction_percent": (base_count - focus_count) / base_count * 100,
            "shared_vocab_tokens": shared,
            "shared_vocab_percent": shared / focused.get_vocab_size() * 100,
        }
        print(f"{domain:8s} floor {pieces:,} / Base {base_count:,} "
              f"= {pieces / base_count * 100:.1f}% fixed; "
              f"actual gain {domains[domain]['actual_reduction_percent']:.2f}%")
    result = {"method": "ByteLevel(use_regex=True) pretoken pieces form a lower bound "
                        "on BPE token count because merges cannot cross their boundaries",
              "domains": domains}
    path = ROOT / "results/tokenizer_headroom.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
