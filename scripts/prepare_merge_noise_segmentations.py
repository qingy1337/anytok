"""Precompute held-out token cuts for the merge-noise dashboard."""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parent.parent
RUN_FILES = (
    "merge_noise_runs_200m.json",
    "merge_noise_runs_high_200m.json",
    "merge_noise_runs_mid_200m.json",
)


def main() -> None:
    samples = json.loads((ROOT / "dashboard/sample_passages.json").read_text())
    runs = [item for name in RUN_FILES
            for item in json.loads((ROOT / "results" / name).read_text())]
    if len(samples) != 23 or len(runs) != 15:
        raise ValueError("Expected 23 held-out samples and 15 matched models")
    evaluation = json.loads((ROOT / "results/merge_noise_eval_200m.json").read_text())
    segmentations = {}
    for run in runs:
        rate = run["settings"]["noise_per_million"]
        seed = run["settings"]["noise_seed"]
        key = f"{rate}:{seed}"
        path = ROOT / "modal_results/tokenizers-merge-noise" / run["job_id"] / "base/tokenizer.json"
        tokenizer = Tokenizer.from_file(str(path))
        if tokenizer.get_vocab_size() != 150_000:
            raise ValueError(f"Unexpected vocabulary size: {key}")
        segmentations[key] = {}
        for sample in samples:
            encoded = tokenizer.encode(sample["text"], add_special_tokens=False)
            pieces = []
            byte_offset = 0
            for token_id, token in zip(encoded.ids, encoded.tokens):
                next_offset = byte_offset + len(token)
                pieces.append({"id": token_id, "token": token,
                               "text": tokenizer.decode([token_id], skip_special_tokens=False),
                               "start": byte_offset, "end": next_offset})
                byte_offset = next_offset
            if byte_offset != len(sample["text"].encode("utf-8")) or \
                    tokenizer.decode(encoded.ids, skip_special_tokens=False) != sample["text"]:
                raise ValueError(f"Byte span or round-trip mismatch: {key} {sample['id']}")
            segmentations[key][sample["id"]] = {"count": len(pieces), "pieces": pieces}

    output = {
        "method": "Matched 200 MB balanced line-chunk BPE, 150k vocabulary; "
                  "non-greedy steps choose merge ranks 2–32",
        "samples": samples,
        "rates_per_million": evaluation["rates_per_million"],
        "seeds": evaluation["seeds"],
        "segmentations": segmentations,
    }
    path = ROOT / "results/merge_noise_segmentations_200m.json"
    path.write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"Saved {path} ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
