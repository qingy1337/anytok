"""Refresh dashboard samples using the five locally mirrored trained models.

Run after editing dashboard/sample_passages.json. This does not retrain models.
"""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parent.parent
LABELS = ("Base", "A", "B", "C", "D")


def main() -> None:
    plan = json.loads((ROOT / "results" / "mixture_examples.json").read_text())
    samples = json.loads((ROOT / "dashboard" / "sample_passages.json").read_text())
    expected = set(plan["baseline"]["bucket_bytes"])
    sample_ids = [sample["id"] for sample in samples]
    if len(sample_ids) != len(expected) or set(sample_ids) != expected:
        raise ValueError("Provide exactly one sample for every training source bucket")

    entries = {"Base": plan["baseline"],
               **{mixture["label"]: mixture for mixture in plan["mixtures"]}}
    models = {}
    segmentations = {}
    settings = None
    for label in LABELS:
        collection = "tokenizers-code50" if label == "C" else "tokenizers-150k"
        directory = ROOT / "modal_results" / collection / label.lower()
        manifest = json.loads((directory / "manifest.json").read_text())
        if {name: item["quota_bytes"] for name, item in manifest["source_stats"].items()} != entries[label]["bucket_bytes"]:
            raise ValueError(f"Training quota mismatch for {label}")
        if settings is not None and manifest["settings"] != settings:
            raise ValueError(f"Tokenizer settings differ for {label}")
        settings = manifest["settings"]
        tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        if tokenizer.get_vocab_size() != settings["vocab_size"]:
            raise ValueError(f"Vocabulary size mismatch for {label}")
        results = {}
        for sample in samples:
            encoding = tokenizer.encode(sample["text"], add_special_tokens=False)
            if tokenizer.decode(encoding.ids, skip_special_tokens=False) != sample["text"]:
                raise ValueError(f"Round-trip mismatch for {label} on {sample['id']}")
            results[sample["id"]] = {
                "count": len(encoding.ids),
                "pieces": [
                    {"id": token_id, "token": token,
                     "text": tokenizer.decode([token_id], skip_special_tokens=False)}
                    for token_id, token in zip(encoding.ids, encoding.tokens)
                ],
            }
        models[label] = manifest
        segmentations[label] = results
        print(f"{label}: {len(results)} samples")

    output = {"recipe": plan["source_recipe"], "settings": settings,
              "samples": samples, "models": models, "segmentations": segmentations}
    path = ROOT / "results" / "tokenizer_segmentations.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
