"""Compare matched Base tokenizers on the 23 held-out source buckets.

Run mirror_algorithm_pilot.py first. Use --full for the 2 GB BPE and WordPiece
comparison; the default compares all three algorithms at 200 MB.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from tokenizers import Tokenizer

from generate_mixture_examples import DOMAIN_BUCKETS


ROOT = Path(__file__).resolve().parent.parent


def model_dir(algorithm: str, scale: str) -> Path:
    if algorithm == "bpe":
        collection = "tokenizers-150k" if scale == "2gb" else "tokenizers-regex200"
        return ROOT / "modal_results" / collection / "base"
    return ROOT / f"modal_results/tokenizers-algorithm-{scale}" / algorithm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Compare the 2 GB BPE and WordPiece models")
    args = parser.parse_args()
    scale = "2gb" if args.full else "200m"
    algorithms = ("bpe", "wordpiece") if args.full else ("bpe", "wordpiece", "unigram")
    plan_name = "mixture_examples.json" if args.full else "pretoken_pilot_mixtures.json"
    plan = json.loads((ROOT / "results" / plan_name).read_text())
    expected_quota = plan["baseline"]["bucket_bytes"]
    models = {}
    manifests = {}
    for algorithm in algorithms:
        directory = model_dir(algorithm, scale)
        manifest = json.loads((directory / "manifest.json").read_text())
        quota = {name: entry["quota_bytes"] for name, entry in manifest["source_stats"].items()}
        if quota != expected_quota:
            raise ValueError(f"{algorithm}: training quotas differ from the {scale} Base plan")
        if manifest["settings"]["pre_tokenizer"] != "ByteLevel(add_prefix_space=False, use_regex=True)":
            raise ValueError(f"{algorithm}: pretokenization is not the matched regex rule")
        model = Tokenizer.from_file(str(directory / "tokenizer.json"))
        if model.get_vocab_size() != manifest["vocab_size_actual"]:
            raise ValueError(f"{algorithm}: model and manifest vocabulary sizes differ")
        if model.get_vocab_size() != 150_000:
            raise ValueError(f"{algorithm}: model did not reach the 150k target")
        models[algorithm] = model
        manifests[algorithm] = manifest
    source_stats = manifests["bpe"]["source_stats"]
    if any(manifests[name]["source_stats"] != source_stats for name in algorithms[1:]):
        raise ValueError("Models did not consume identical training documents")

    buckets = {}
    for domain, names in DOMAIN_BUCKETS.items():
        for name in names:
            totals = {algorithm: 0 for algorithm in algorithms}
            unknowns = {algorithm: 0 for algorithm in algorithms}
            raw_bytes = documents = 0
            with gzip.open(ROOT / "modal_results/eval" / f"{name}.jsonl.gz", "rt",
                           encoding="utf-8") as handle:
                for line in handle:
                    content = json.loads(line)["text"]
                    raw_bytes += len(content.encode("utf-8"))
                    documents += 1
                    for algorithm, model in models.items():
                        encoding = model.encode(content, add_special_tokens=False)
                        if model.decode(encoding.ids, skip_special_tokens=False) != content:
                            raise ValueError(f"{algorithm}: round-trip failed in {name}, document {documents}")
                        totals[algorithm] += len(encoding.ids)
                        unknown_id = model.token_to_id("<|unk|>")
                        if unknown_id is not None:
                            unknowns[algorithm] += encoding.ids.count(unknown_id)
            buckets[name] = {"domain": domain, "documents": documents, "bytes": raw_bytes,
                             "tokens": totals, "unknown_tokens": unknowns}
        print(f"Checked {domain}", flush=True)

    domains = {}
    for domain, names in DOMAIN_BUCKETS.items():
        tokens = {algorithm: sum(buckets[name]["tokens"][algorithm] for name in names)
                  for algorithm in algorithms}
        domains[domain] = {
            "bytes": sum(buckets[name]["bytes"] for name in names),
            "tokens": tokens,
            "relative_to_bpe_percent": {
                algorithm: 100 * (tokens[algorithm] / tokens["bpe"] - 1)
                for algorithm in algorithms[1:]
            },
            "unknown_tokens": {
                algorithm: sum(buckets[name]["unknown_tokens"][algorithm] for name in names)
                for algorithm in algorithms
            },
        }
    totals = {algorithm: sum(item["tokens"][algorithm] for item in buckets.values())
              for algorithm in algorithms}
    output = {
        "method": f"Same {scale} Base training documents, 150k target vocabulary, byte-level regex pretokenization, and held-out documents; algorithm varies.",
        "vocab_size_actual": {name: manifests[name]["vocab_size_actual"] for name in algorithms},
        "train_seconds": {name: manifests[name]["train_seconds"] for name in algorithms},
        "total_held_out_tokens": totals,
        "relative_to_bpe_percent": {name: 100 * (totals[name] / totals["bpe"] - 1)
                                    for name in algorithms[1:]},
        "buckets": buckets,
        "domains": domains,
    }
    path = ROOT / f"results/algorithm_eval_{scale}.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: output[key] for key in
                      ("vocab_size_actual", "train_seconds", "total_held_out_tokens",
                       "relative_to_bpe_percent")}, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
