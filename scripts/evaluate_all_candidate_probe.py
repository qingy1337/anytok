"""Evaluate the matched 200 MB all-candidate 1% probe."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from tokenizers import Tokenizer

from evaluate_merge_noise import compare
from generate_mixture_examples import DOMAIN_BUCKETS


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    run = json.loads((ROOT / "results/merge_noise_all_probe_200m.json").read_text())[0]
    settings = run["settings"]
    if settings["noise_per_million"] != 10_000 or settings["noise_seed"] != 1 or \
            settings["noise_candidates"] != "all_eligible" or run["vocab_size_actual"] != 150_000:
        raise ValueError("Expected the 200 MB all-candidate 1% probe")
    base_path = ROOT / "modal_results/tokenizers-line200/base/tokenizer.json"
    base_manifest = json.loads(base_path.with_name("manifest.json").read_text())
    if run["source_stats"] != base_manifest["source_stats"]:
        raise ValueError("Probe training data differ from Base")
    top_run = next(item for item in json.loads((ROOT / "results/merge_noise_runs_200m.json").read_text())
                   if item["settings"]["noise_per_million"] == 10_000
                   and item["settings"]["noise_seed"] == 1)
    top_path = ROOT / "modal_results/tokenizers-merge-noise" / top_run["job_id"] / "base/tokenizer.json"
    all_path = ROOT / "modal_results/tokenizers-merge-noise-all" / run["job_id"] / "base/tokenizer.json"
    base = json.loads(base_path.read_text())
    top = json.loads(top_path.read_text())
    all_model = json.loads(all_path.read_text())
    model = Tokenizer.from_file(str(all_path))
    prior_eval = json.loads((ROOT / "results/pretokenization_eval_200m.json").read_text())
    held_out = {}
    for domain, buckets in DOMAIN_BUCKETS.items():
        texts = []
        for bucket in buckets:
            with gzip.open(ROOT / "modal_results/eval" / f"{bucket}.jsonl.gz", "rt",
                           encoding="utf-8") as handle:
                texts.extend(json.loads(line)["text"] for line in handle)
        count = sum(len(encoded.ids) for encoded in model.encode_batch(texts, add_special_tokens=False))
        base_count = prior_eval["domains"][domain]["tokens"]["line256:Base"]
        held_out[domain] = {"all_candidate_tokens": count, "base_tokens": base_count,
                            "change_percent": 100 * (count / base_count - 1)}
    output = {
        "method": "Same 200 MB balanced mixture, 150k vocabulary, cross-word line chunks, 1% noise, seed 1",
        "all_vs_base": compare(base, all_model),
        "top32_vs_base": compare(base, top),
        "all_vs_top32": compare(top, all_model),
        "held_out": held_out,
        "run": run,
    }
    path = ROOT / "results/merge_noise_all_probe_eval_200m.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
