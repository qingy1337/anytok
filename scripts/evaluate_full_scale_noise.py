"""Compare matched 2 GB, all-candidate merge-noise seeds."""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
from pathlib import Path

from tokenizers import Tokenizer

from evaluate_merge_noise import compare
from generate_mixture_examples import DOMAIN_BUCKETS


ROOT = Path(__file__).resolve().parent.parent
MIRROR = ROOT / "modal_results/tokenizers-merge-noise-all"


def model_path(run: dict) -> Path:
    return MIRROR / run["job_id"] / "base/tokenizer.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretokenizer", choices=("line256", "regex"), required=True)
    parser.add_argument("--noise-rate", type=int, choices=(500_000, 750_000),
                        default=500_000, help="Noise rate in millionths")
    args = parser.parse_args()
    rate = args.noise_rate
    suffix = "_75" if rate == 750_000 else ""
    run_file = ROOT / f"results/merge_noise_all_{args.pretokenizer}{suffix}_2gb.json"
    runs = json.loads(run_file.read_text())
    by_key = {(run["settings"]["noise_per_million"], run["settings"]["noise_seed"]): run
              for run in runs}
    expected = {(0, 1)} | {(rate, seed) for seed in (1, 2, 3)}
    if set(by_key) != expected or len(runs) != len(expected):
        raise ValueError("Expected one control and three noise seeds")
    control = by_key[(0, 1)]
    for run in runs:
        settings = run["settings"]
        if (not 2_000_000_000 - 3 * 23 <= run["total_used_bytes"] <= 2_000_000_000 or
                run["vocab_size_actual"] != 150_000 or
                run["source_stats"] != control["source_stats"] or
                settings["pretokenizer"] != args.pretokenizer or
                settings["noise_candidates"] != "all_eligible"):
            raise ValueError("Training data or settings differ between runs")

    saved = {key: json.loads(model_path(run).read_text()) for key, run in by_key.items()}
    tokenizers = {key: Tokenizer.from_file(str(model_path(run)))
                  for key, run in by_key.items()}
    base = saved[(0, 1)]
    stock_check = None
    if args.pretokenizer == "regex":
        stock_path = ROOT / "modal_results/tokenizers-150k/base/tokenizer.json"
        stock_manifest_path = stock_path.with_name("manifest.json")
        if stock_path.exists() and stock_manifest_path.exists():
            stock_manifest = json.loads(stock_manifest_path.read_text())
            if stock_manifest["source_stats"] != control["source_stats"]:
                raise ValueError("Stock Base and noise control read different training data")
            stock_check = compare(json.loads(stock_path.read_text()), base)
            if stock_check["shared_tokens"] != 150_000 or stock_check["first_different_merge_index"] != len(base["model"]["merges"]):
                raise ValueError("Zero-noise control differs from stock regex Base")
    vs_base = {str(seed): compare(base, saved[(rate, seed)]) for seed in (1, 2, 3)}
    pairwise = {f"{a}-{b}": compare(saved[(rate, a)], saved[(rate, b)])
                for a, b in itertools.combinations((1, 2, 3), 2)}

    held_out = {f"{rate}:{seed}": {} for rate, seed in expected}
    vocab_sets = {key: set(model["model"]["vocab"]) for key, model in saved.items()}
    token_use_absent = {f"{rate}:{seed}": {} for rate, seed in expected}
    for domain, buckets in DOMAIN_BUCKETS.items():
        texts = []
        for bucket in buckets:
            with gzip.open(ROOT / "modal_results/eval" / f"{bucket}.jsonl.gz", "rt",
                           encoding="utf-8") as handle:
                texts.extend(json.loads(line)["text"] for line in handle)
        for (item_rate, seed), model in tokenizers.items():
            encodings = model.encode_batch(texts, add_special_tokens=False)
            label = f"{item_rate}:{seed}"
            held_out[label][domain] = sum(len(encoded.ids) for encoded in encodings)
            token_use_absent[label][domain] = {
                f"{other_rate}:{other_seed}": sum(
                    token not in vocab_sets[(other_rate, other_seed)]
                    for encoded in encodings for token in encoded.tokens
                )
                for other_rate, other_seed in expected
                if (other_rate, other_seed) != (item_rate, seed)
            }

    samples = json.loads((ROOT / "dashboard/sample_passages.json").read_text())
    sample_cuts = {seed: {} for seed in (0, 1, 2, 3)}
    for seed in (0, 1, 2, 3):
        model = tokenizers[(0, 1) if seed == 0 else (rate, seed)]
        for sample in samples:
            encoded = model.encode(sample["text"], add_special_tokens=False)
            if model.decode(encoded.ids, skip_special_tokens=False) != sample["text"]:
                raise ValueError(f"Round-trip failed for seed {seed}, {sample['id']}")
            pieces = []
            byte_offset = 0
            for token_id, token in zip(encoded.ids, encoded.tokens):
                next_offset = byte_offset + len(token)
                pieces.append({
                    "token": token,
                    "text": model.decode([token_id], skip_special_tokens=False),
                    "start": byte_offset,
                    "end": next_offset,
                })
                byte_offset = next_offset
            if byte_offset != len(sample["text"].encode("utf-8")):
                raise ValueError(f"Byte spans failed for seed {seed}, {sample['id']}")
            sample_cuts[seed][sample["id"]] = {
                "count": len(encoded.ids),
                "boundaries": [piece["end"] for piece in pieces[:-1]],
                "pieces": pieces,
            }
    cut_overlap = {}
    for a, b in itertools.combinations((1, 2, 3), 2):
        per_sample = {}
        for sample in samples:
            sid = sample["id"]
            left = set(sample_cuts[a][sid]["boundaries"])
            right = set(sample_cuts[b][sid]["boundaries"])
            per_sample[sid] = len(left & right) / len(left | right) if left | right else 1.0
        cut_overlap[f"{a}-{b}"] = {
            "mean_boundary_jaccard_percent": 100 * sum(per_sample.values()) / len(per_sample),
            "by_sample_percent": {sid: 100 * value for sid, value in per_sample.items()},
        }

    result = {
        "method": f"Matched 2 GB balanced mixture, 150k {args.pretokenizer} BPE; "
                  f"{rate / 10_000:.0f}% of steps uniformly sample an eligible pair type with frequency >=2.",
        "noise_per_million": rate,
        "vocab_vs_base": vs_base,
        "stock_base_check": stock_check,
        "vocab_between_seeds": pairwise,
        "held_out_tokens": held_out,
        "held_out_token_use_absent_from_other_vocab": token_use_absent,
        "sample_cuts": sample_cuts,
        "samples": samples,
        "sample_cut_overlap": cut_overlap,
        "manifests": runs,
    }
    output = ROOT / f"results/merge_noise_all_{args.pretokenizer}{suffix}_2gb_eval.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {output}")
    for label, stats in pairwise.items():
        print(f"Seeds {label}: {stats['shared_percent']:.2f}% shared vocabulary; "
              f"{cut_overlap[label]['mean_boundary_jaccard_percent']:.1f}% shared sample cuts")
    baseline = sum(held_out["0:1"].values())
    for seed in (1, 2, 3):
        count = sum(held_out[f"{rate}:{seed}"].values())
        absent = sum(domain["0:1"] for domain in token_use_absent[f"{rate}:{seed}"].values())
        print(f"Seed {seed}: {count:,} held-out tokens, {100 * (count / baseline - 1):+.2f}% vs control; "
              f"{100 * absent / count:.2f}% of uses absent from Base vocabulary")


if __name__ == "__main__":
    main()
