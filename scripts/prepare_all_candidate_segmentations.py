"""Precompute held-out cuts for the all-candidate merge-noise dashboard."""

from __future__ import annotations

import json
from pathlib import Path
import argparse

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parent.parent
RUN_FILES = ("merge_noise_all_smoke_40m_full.json", "merge_noise_all_mid_40m.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretokenizer", choices=("line256", "regex"), default="line256")
    args = parser.parse_args()
    run_files = (("merge_noise_all_regex_40m.json",) if args.pretokenizer == "regex"
                 else RUN_FILES)
    samples = json.loads((ROOT / "dashboard/sample_passages.json").read_text())
    runs = [item for name in run_files
            for item in json.loads((ROOT / "results" / name).read_text())]
    if len(samples) != 23 or len(runs) != 21:
        raise ValueError("Expected 23 held-out samples and 21 matched models")
    suffix = "_regex" if args.pretokenizer == "regex" else ""
    evaluation = json.loads((ROOT / f"results/merge_noise_all_eval{suffix}_40m.json").read_text())
    segmentations = {}
    for run in runs:
        rate = run["settings"]["noise_per_million"]
        seed = run["settings"]["noise_seed"]
        key = f"{rate}:{seed}"
        path = ROOT / "modal_results/tokenizers-merge-noise-all" / run["job_id"] / "base/tokenizer.json"
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
        "method": f"Matched 40 MB balanced {args.pretokenizer} BPE, 150k vocabulary; "
                  "noisy steps sample uniformly from all eligible pair types",
        "samples": samples,
        "rates_per_million": evaluation["rates_per_million"],
        "seeds": evaluation["seeds"],
        "segmentations": segmentations,
    }
    path = ROOT / f"results/merge_noise_all_segmentations{suffix}_40m.json"
    path.write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"Saved {path} ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
