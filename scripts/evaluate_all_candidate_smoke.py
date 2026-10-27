"""Evaluate the matched 40 MB all-candidate merge-noise sweep."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import argparse

from tokenizers import Tokenizer

from evaluate_merge_noise import compare
from generate_mixture_examples import DOMAIN_BUCKETS


ROOT = Path(__file__).resolve().parent.parent
RATES = (0, 1000, 10000, 50000, 200000, 250000, 350000, 500000,
         650000, 750000, 1000000)


def model_path(item: dict) -> Path:
    return ROOT / "modal_results/tokenizers-merge-noise-all" / item["job_id"] / "base/tokenizer.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretokenizer", choices=("line256", "regex"), default="line256")
    args = parser.parse_args()
    run_files = (("merge_noise_all_regex_40m.json",) if args.pretokenizer == "regex"
                 else ("merge_noise_all_smoke_40m_full.json", "merge_noise_all_mid_40m.json"))
    runs = [item for name in run_files
            for item in json.loads((ROOT / "results" / name).read_text())]
    key = lambda item: (item["settings"]["noise_per_million"], item["settings"]["noise_seed"])
    runs = {key(item): item for item in runs}
    expected = {(0, 1)} | {(rate, seed) for rate in RATES[1:] for seed in (1, 2)}
    if set(runs) != expected:
        raise ValueError("Expected 21 matched 40 MB models")
    baseline_stats = runs[(0, 1)]["source_stats"]
    if args.pretokenizer == "regex":
        line = json.loads((ROOT / "results/merge_noise_all_eval_40m.json").read_text())
        if baseline_stats != next(item for item in line["manifests"]
                                  if item["settings"]["noise_per_million"] == 0)["source_stats"]:
            raise ValueError("Regex and cross-word training samples differ")
    for item in runs.values():
        if item["vocab_size_actual"] != 150_000 or item["source_stats"] != baseline_stats or \
                item["settings"]["noise_candidates"] != "all_eligible" or \
                item["settings"]["pretokenizer"] != args.pretokenizer:
            raise ValueError("Models do not share the 40 MB training plan and settings")
    model_json = {k: json.loads(model_path(item).read_text()) for k, item in runs.items()}
    base = model_json[(0, 1)]
    vs_base = {f"{rate}:{seed}": compare(base, model_json[(rate, seed)])
               for rate, seed in sorted(expected) if rate > 0}
    between_seeds = {str(rate): compare(model_json[(rate, 1)], model_json[(rate, 2)])
                     for rate in RATES[1:]}
    for name, stats in vs_base.items():
        print(f"rate:seed {name:9} changed {stats['different_tokens_each']:,} "
              f"({100 - stats['shared_percent']:.2f}%)")

    tokenizers = {k: Tokenizer.from_file(str(model_path(item))) for k, item in runs.items()}
    held_out = {f"{rate}:{seed}": {} for rate, seed in expected}
    for domain, buckets in DOMAIN_BUCKETS.items():
        texts = []
        for bucket in buckets:
            with gzip.open(ROOT / "modal_results/eval" / f"{bucket}.jsonl.gz", "rt",
                           encoding="utf-8") as handle:
                texts.extend(json.loads(line)["text"] for line in handle)
        for (rate, seed), model in tokenizers.items():
            held_out[f"{rate}:{seed}"][domain] = sum(
                len(encoded.ids) for encoded in model.encode_batch(texts, add_special_tokens=False)
            )

    output = {
        "method": f"Same 40 MB balanced mixture, 150k {args.pretokenizer} BPE. "
                  "At a noisy step, uniformly sample any currently eligible pair "
                  "with frequency at least 2; otherwise choose the most frequent pair.",
        "rates_per_million": list(RATES),
        "seeds": [1, 2],
        "vocab_vs_base": vs_base,
        "vocab_between_seeds": between_seeds,
        "held_out_tokens": held_out,
        "manifests": list(runs.values()),
    }
    suffix = "_regex" if args.pretokenizer == "regex" else ""
    path = ROOT / f"results/merge_noise_all_eval{suffix}_40m.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
