"""Compare regex and line-chunk BPE using matched mixtures and held-out text."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from tokenizers import Tokenizer

from generate_mixture_examples import DOMAIN_BUCKETS


ROOT = Path(__file__).resolve().parent.parent
FOCI = {
    "papers": "Papers25", "math": "Math50", "qa": "QA25",
    "books": "Books25", "medical": "Medical40", "legal": "Legal40",
}


def model_path(rule: str, label: str) -> Path:
    collection = "tokenizers-line200" if rule == "line256" else "tokenizers-regex200"
    return ROOT / "modal_results" / collection / label.lower() / "tokenizer.json"


def main() -> None:
    labels = ("Base", *FOCI.values())
    plan = json.loads((ROOT / "results/pretoken_pilot_mixtures.json").read_text())
    entries = {"Base": plan["baseline"],
               **{mixture["label"]: mixture for mixture in plan["mixtures"]}}
    if set(entries) != set(labels) or plan["total_train_bytes_per_mixture"] != 200_000_000:
        raise ValueError("Expected the seven matched 200 MB mixtures")
    for label in labels:
        manifests = {}
        for rule in ("regex", "line256"):
            path = model_path(rule, label).with_name("manifest.json")
            manifest = json.loads(path.read_text())
            quotas = {name: item["quota_bytes"] for name, item in manifest["source_stats"].items()}
            if quotas != entries[label]["bucket_bytes"] or manifest["vocab_size_actual"] != 150_000:
                raise ValueError(f"Invalid {rule} manifest for {label}")
            manifests[rule] = manifest
        if manifests["regex"]["source_stats"] != manifests["line256"]["source_stats"]:
            raise ValueError(f"Training source statistics differ for {label}")
        for key in manifests["regex"]["settings"]:
            if key != "pre_tokenizer" and (manifests["regex"]["settings"][key] !=
                                           manifests["line256"]["settings"][key]):
                raise ValueError(f"Training settings differ beyond pretokenization for {label}")
    models = {(rule, label): Tokenizer.from_file(str(model_path(rule, label)))
              for rule in ("regex", "line256") for label in labels}
    if any(model.get_vocab_size() != 150_000 for model in models.values()):
        raise ValueError("Every model must have a 150k vocabulary")

    buckets = {}
    for domain, names in DOMAIN_BUCKETS.items():
        for name in names:
            with gzip.open(ROOT / "modal_results/eval" / f"{name}.jsonl.gz", "rt",
                           encoding="utf-8") as handle:
                texts = [json.loads(line)["text"] for line in handle]
            counts = {f"{rule}:{label}": sum(len(encoded.ids) for encoded in model.encode_batch(
                          texts, add_special_tokens=False))
                      for (rule, label), model in models.items()}
            buckets[name] = {"domain": domain, "documents": len(texts),
                             "bytes": sum(len(text.encode("utf-8")) for text in texts),
                             "tokens": counts}

    domains = {}
    for domain, names in DOMAIN_BUCKETS.items():
        totals = {}
        for rule in ("regex", "line256"):
            for label in labels:
                key = f"{rule}:{label}"
                totals[key] = sum(buckets[name]["tokens"][key] for name in names)
        base_regex = totals["regex:Base"]
        base_line = totals["line256:Base"]
        result = {"bytes": sum(buckets[name]["bytes"] for name in names),
                  "tokens": totals,
                  "base_line_vs_regex_percent": (base_line / base_regex - 1) * 100}
        if domain in FOCI:
            label = FOCI[domain]
            result.update({
                "focused_label": label,
                "focused_line_vs_regex_percent":
                    (totals[f"line256:{label}"] / totals[f"regex:{label}"] - 1) * 100,
                "regex_focus_vs_base_percent":
                    (totals[f"regex:{label}"] / base_regex - 1) * 100,
                "line_focus_vs_base_percent":
                    (totals[f"line256:{label}"] / base_line - 1) * 100,
            })
        domains[domain] = result
        if domain in FOCI:
            print(f"{domain:10s} Base line vs regex {result['base_line_vs_regex_percent']:+.2f}% "
                  f"| specialization regex {result['regex_focus_vs_base_percent']:+.2f}% "
                  f"line {result['line_focus_vs_base_percent']:+.2f}%")
        else:
            print(f"{domain:10s} Base line vs regex {result['base_line_vs_regex_percent']:+.2f}%")

    output = {"method": "identical 200 MB training mixtures, 150k BPE settings, and held-out documents; "
                        "only the pretokenization rule changes",
              "rules": {"regex": "ByteLevel(use_regex=True)",
                        "line256": "split at line breaks and at 256 characters, then ByteLevel(use_regex=False)"},
              "buckets": buckets, "domains": domains}
    path = ROOT / "results/pretokenization_eval_200m.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
