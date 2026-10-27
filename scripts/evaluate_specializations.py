"""Compare Base and each specialist on held-out text from its focus domain.

Download the four eval files from the Modal volume into modal_results/eval first.
The report counts BPE tokens on complete records and reports bytes per token.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parent.parent
FOCI = {"A": ("medical", "medical-en"), "B": ("legal", "legal-en"),
        "C": ("code", "code"), "D": ("math", "math-en")}


def main() -> None:
    plan = json.loads((ROOT / "results" / "mixture_examples.json").read_text())
    models = {label: Tokenizer.from_file(str(ROOT / "modal_results" /
                                              ("tokenizers-code50" if label == "C" else "tokenizers-150k") /
                                              label.lower() / "tokenizer.json"))
              for label in ("Base", *FOCI)}
    domains = {}
    for label, (domain, bucket) in FOCI.items():
        path = ROOT / "modal_results" / "eval" / f"{domain}.jsonl.gz"
        texts = []
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                texts.append(json.loads(line)["text"])
        raw_bytes = sum(len(item.encode("utf-8")) for item in texts)
        counts = {name: sum(len(encoding.ids) for encoding in tokenizer.encode_batch(texts,
                                add_special_tokens=False)) for name, tokenizer in
                  (("Base", models["Base"]), (label, models[label]))}
        domains[domain] = {"bucket": bucket, "documents": len(texts),
                           "bytes": raw_bytes, "token_counts": counts,
                           "bytes_per_token": {name: raw_bytes / count for name, count in counts.items()},
                           "specialist_token_change_percent":
                           (counts[label] / counts["Base"] - 1) * 100}
        print(f"{label} {domain}: {len(texts):,} docs, {raw_bytes:,} bytes, "
              f"Base {counts['Base']:,} tokens, {label} {counts[label]:,} tokens "
              f"({domains[domain]['specialist_token_change_percent']:+.2f}%)")
    output = {"recipe": plan["source_recipe"], "method": "full held-out source documents",
              "domains": domains}
    path = ROOT / "results" / "specialization_eval.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
