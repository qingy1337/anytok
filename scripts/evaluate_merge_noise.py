"""Evaluate vocabulary divergence and held-out token counts for merge-noise runs."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from tokenizers import Tokenizer

from generate_mixture_examples import DOMAIN_BUCKETS


ROOT = Path(__file__).resolve().parent.parent
MIRROR = ROOT / "modal_results/tokenizers-merge-noise"


def model_file(manifest: dict) -> Path:
    return MIRROR / manifest["job_id"] / "base/tokenizer.json"


def compare(a: dict, b: dict) -> dict:
    left = a["model"]["vocab"]
    right = b["model"]["vocab"]
    shared = left.keys() & right.keys()
    merges_a = a["model"]["merges"]
    merges_b = b["model"]["merges"]
    first_difference = next((index for index, (x, y) in enumerate(zip(merges_a, merges_b))
                             if x != y), min(len(merges_a), len(merges_b)))
    return {
        "shared_tokens": len(shared),
        "shared_percent": 100 * len(shared) / len(left),
        "different_tokens_each": len(left) - len(shared),
        "shared_tokens_at_same_id": sum(left[token] == right[token] for token in shared),
        "first_different_merge_index": first_difference,
        "merges_at_same_rank": sum(x == y for x, y in zip(merges_a, merges_b)),
        "merges_at_same_rank_percent": 100 * sum(x == y for x, y in zip(merges_a, merges_b)) / len(merges_a),
        "base_top_10000_present_percent": 100 * sum(
            token in right for token, rank in left.items() if rank < 10_000
        ) / 10_000,
    }


def main() -> None:
    manifests = json.loads((ROOT / "results/merge_noise_runs_200m.json").read_text())
    high_path = ROOT / "results/merge_noise_runs_high_200m.json"
    if high_path.exists():
        manifests += json.loads(high_path.read_text())
    mid_path = ROOT / "results/merge_noise_runs_mid_200m.json"
    if mid_path.exists():
        manifests += json.loads(mid_path.read_text())
    if any(item["vocab_size_actual"] != 150_000 for item in manifests):
        raise ValueError("Expected 150k runs")
    key = lambda item: (item["settings"]["noise_per_million"],
                        item["settings"]["noise_seed"])
    manifests = sorted(manifests, key=key)
    expected = {(0, 1), (1000, 1), (1000, 2), (10000, 1), (10000, 2),
                (50000, 1), (50000, 2)}
    if high_path.exists():
        expected |= {(200000, 1), (200000, 2), (1000000, 1), (1000000, 2)}
    if mid_path.exists():
        expected |= {(500000, 1), (500000, 2), (750000, 1), (750000, 2)}
    if {key(item) for item in manifests} != expected:
        raise ValueError("Missing a rate or seed")
    source_stats = manifests[0]["source_stats"]
    if any(item["source_stats"] != source_stats for item in manifests):
        raise ValueError("Training data differs between runs")
    if any(item["total_used_bytes"] != manifests[0]["total_used_bytes"]
           or item["settings"]["noise_candidates"] != 32
           or item["settings"]["pretokenizer"] != "line256"
           for item in manifests):
        raise ValueError("Training settings or data volume differ between runs")

    saved = {key(item): json.loads(model_file(item).read_text()) for item in manifests}
    baseline = saved[(0, 1)]
    original = json.loads((ROOT / "modal_results/tokenizers-line200/base/tokenizer.json").read_text())
    zero_check = compare(original, baseline)
    if zero_check["shared_tokens"] != 150_000 or zero_check["first_different_merge_index"] != len(original["model"]["merges"]):
        raise ValueError("Patched zero-noise trainer does not reproduce cross-word Base")

    vs_base = {f"{rate}:{seed}": compare(baseline, model)
               for (rate, seed), model in saved.items() if rate > 0}
    between_seeds = {str(rate): compare(saved[(rate, 1)], saved[(rate, 2)])
                     for rate in sorted({rate for rate, _ in saved if rate > 0})}
    for label, stats in vs_base.items():
        print(f"rate:seed {label:8} shared {stats['shared_percent']:6.2f}% "
              f"({stats['different_tokens_each']:,} changed tokens each), "
              f"first changed merge {stats['first_different_merge_index']:,}")

    tokenizers = {k: Tokenizer.from_file(str(model_file(item)))
                  for item in manifests for k in [key(item)]}
    eval_counts = {k: {} for k in saved}
    for domain, buckets in DOMAIN_BUCKETS.items():
        texts = []
        for bucket in buckets:
            with gzip.open(ROOT / "modal_results/eval" / f"{bucket}.jsonl.gz", "rt",
                           encoding="utf-8") as handle:
                texts.extend(json.loads(line)["text"] for line in handle)
        for k, model in tokenizers.items():
            eval_counts[k][domain] = sum(len(encoded.ids) for encoded in
                                         model.encode_batch(texts, add_special_tokens=False))

    output = {
        "method": "Same 200 MB balanced mixture, 150k vocabulary, line-chunk cross-word BPE. "
                  "With probability p, choose uniformly from valid merge ranks 2–32. "
                  "Only rate and seed vary.",
        "rates_per_million": sorted({rate for rate, _ in saved}),
        "seeds": [1, 2],
        "zero_noise_matches_original": zero_check,
        "vocab_vs_zero_noise": vs_base,
        "vocab_between_seeds": between_seeds,
        "held_out_tokens": {f"{rate}:{seed}": counts for (rate, seed), counts in eval_counts.items()},
        "manifests": manifests,
    }
    path = ROOT / "results/merge_noise_eval_200m.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
