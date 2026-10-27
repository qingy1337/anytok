"""Measure six heavily focused 150k BPE models on held-out domain text."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parent.parent
FOCI = {
    "papers": ("papers-en", "Papers25", None),
    "math": ("math-en", "Math50", "D30"),
    "qa": ("qa-en", "QA25", None),
    "books": ("books-en", "Books25", None),
    "medical": ("medical-en", "Medical40", "A30"),
    "legal": ("legal-en", "Legal40", "B30"),
}
MODEL_PATHS = {
    "Base": ROOT / "modal_results/tokenizers-150k/base/tokenizer.json",
    "A30": ROOT / "modal_results/tokenizers-150k/a/tokenizer.json",
    "B30": ROOT / "modal_results/tokenizers-150k/b/tokenizer.json",
    "D30": ROOT / "modal_results/tokenizers-150k/d/tokenizer.json",
    **{label: ROOT / "modal_results/tokenizers-heavy" / label.lower() / "tokenizer.json"
       for _, label, _ in FOCI.values()},
}


def main() -> None:
    models = {label: Tokenizer.from_file(str(path)) for label, path in MODEL_PATHS.items()}
    if any(model.get_vocab_size() != 150_000 for model in models.values()):
        raise ValueError("All comparison models must have 150k vocabulary")
    results = {}
    for domain, (bucket, focus_label, prior_label) in FOCI.items():
        path = ROOT / "modal_results/eval" / f"{bucket}.jsonl.gz"
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            texts = [json.loads(line)["text"] for line in handle]
        byte_count = sum(len(text.encode("utf-8")) for text in texts)
        counts = {label: sum(len(encoded.ids) for encoded in model.encode_batch(
                      texts, add_special_tokens=False)) for label, model in models.items()}
        result = {
            "bucket": bucket, "documents": len(texts), "bytes": byte_count,
            "token_counts": counts,
            "bytes_per_token": {label: byte_count / count for label, count in counts.items()},
            "focused_model": focus_label, "prior_model": prior_label,
            "focused_vs_base_percent": (counts[focus_label] / counts["Base"] - 1) * 100,
            "focused_vs_prior_percent":
                (counts[focus_label] / counts[prior_label] - 1) * 100 if prior_label else None,
        }
        results[domain] = result
        delta = result["focused_vs_base_percent"]
        extra = (f", vs {prior_label} {result['focused_vs_prior_percent']:+.2f}%"
                 if prior_label else "")
        print(f"{domain:8s} {focus_label}: {counts[focus_label]:,} tokens, "
              f"vs Base {delta:+.2f}%{extra}")
    output = {"method": "complete held-out source documents; raw UTF-8 bytes per BPE token",
              "vocab_size": 150_000, "train_bytes_per_model": 2_000_000_000,
              "domains": results}
    path = ROOT / "results/heavy_domain_eval.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
