"""Compare the 30% and 50% code mixtures on every held-out source bucket."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from tokenizers import Tokenizer

from generate_mixture_examples import DOMAIN_BUCKETS


ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "Base": ROOT / "modal_results/tokenizers-150k/base/tokenizer.json",
    "C30": ROOT / "modal_results/tokenizers-150k/c/tokenizer.json",
    "C50": ROOT / "modal_results/tokenizers-code50/c/tokenizer.json",
}


def main() -> None:
    tokenizers = {label: Tokenizer.from_file(str(path)) for label, path in MODELS.items()}
    buckets = {}
    for domain, names in DOMAIN_BUCKETS.items():
        for name in names:
            path = ROOT / "modal_results/eval" / f"{name}.jsonl.gz"
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                texts = [json.loads(line)["text"] for line in handle]
            counts = {label: sum(len(encoding.ids) for encoding in tokenizer.encode_batch(
                          texts, add_special_tokens=False))
                      for label, tokenizer in tokenizers.items()}
            buckets[name] = {"domain": domain, "documents": len(texts),
                             "bytes": sum(len(text.encode("utf-8")) for text in texts),
                             "tokens": counts}

    domains = {}
    for domain, names in DOMAIN_BUCKETS.items():
        totals = {label: sum(buckets[name]["tokens"][label] for name in names)
                  for label in MODELS}
        domains[domain] = {
            "bytes": sum(buckets[name]["bytes"] for name in names),
            "tokens": totals,
            "c50_vs_c30_percent": (totals["C50"] / totals["C30"] - 1) * 100,
            "c30_vs_base_percent": (totals["C30"] / totals["Base"] - 1) * 100,
            "c50_vs_base_percent": (totals["C50"] / totals["Base"] - 1) * 100,
        }
        print(f"{domain:10s} C50 vs C30 {domains[domain]['c50_vs_c30_percent']:+.2f}% "
              f"| C50 vs Base {domains[domain]['c50_vs_base_percent']:+.2f}%")

    result = {"method": "full held-out source documents; same 2 GB total and 150k settings",
              "code_train_share": {"Base": 10, "C30": 30, "C50": 50},
              "buckets": buckets, "domains": domains}
    path = ROOT / "results/code_focus_eval.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
