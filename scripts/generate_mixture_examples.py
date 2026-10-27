"""Generate Base and four single-domain specialty mixtures from a Modal recipe.

    python scripts/generate_mixture_examples.py \
        --recipe data/balanced-full-v3.json --output results/mixture_examples.json

All five plans contain exact per-source raw UTF-8 byte quotas for 2 GB of text.
A–D each put 30% of the text into one domain by default. Use
--code-focus-percent 50 for the active C mixture. Remaining text is allocated
proportionally to Base, including its within-domain language shares.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DOMAIN_BUCKETS = {
    "web": ("web-en", "web-es", "web-fr", "web-ru", "web-zh", "web-ja", "web-ar", "web-hi"),
    "wikipedia": ("wiki-en", "wiki-es", "wiki-fr", "wiki-ru", "wiki-zh", "wiki-ja", "wiki-ar", "wiki-hi"),
    "code": ("code",),
    "papers": ("papers-en",),
    "math": ("math-en",),
    "qa": ("qa-en",),
    "books": ("books-en",),
    "medical": ("medical-en",),
    "legal": ("legal-en",),
}
FOCI = {"A": "medical", "B": "legal", "C": "code", "D": "math"}
FOCUS_PERCENT = 30


def allocate_exact(total: int, weights: list[int]) -> list[int]:
    denominator = sum(weights)
    if total < 0 or denominator <= 0:
        raise ValueError("Allocation needs a nonnegative total and positive weights")
    raw = [total * weight / denominator for weight in weights]
    result = [int(value) for value in raw]
    for index in sorted(range(len(raw)), key=lambda i: raw[i] - result[i], reverse=True)[:total - sum(result)]:
        result[index] += 1
    assert sum(result) == total
    return result


def summarise(buckets: dict[str, int], total: int, baseline: dict[str, int] | None = None,
              label: str | None = None, focus: str | None = None) -> dict:
    domains = {domain: sum(buckets[name] for name in names)
               for domain, names in DOMAIN_BUCKETS.items()}
    result = {"domain_bytes": domains, "bucket_bytes": buckets,
              "domain_percent": {name: value / total * 100 for name, value in domains.items()},
              "bucket_percent": {name: value / total * 100 for name, value in buckets.items()}}
    if label is not None:
        result.update({"label": label, "focus_domain": focus,
                       "moved_bytes": sum(abs(buckets[name] - baseline[name]) for name in baseline) // 2,
                       "macro_moved_bytes": sum(abs(domains[domain] - sum(baseline[name] for name in names))
                                                for domain, names in DOMAIN_BUCKETS.items()) // 2})
    return result


def generate(recipe: dict, total_bytes: int, focus_percent: int = FOCUS_PERCENT,
             code_focus_percent: int | None = None) -> dict:
    entries = {bucket["name"]: bucket for bucket in recipe["buckets"]}
    expected = {name for names in DOMAIN_BUCKETS.values() for name in names}
    if len(entries) != len(recipe["buckets"]) or set(entries) != expected:
        raise ValueError("Recipe bucket list must contain exactly the 23 expected sources")
    if not 1 <= focus_percent < 100 or total_bytes <= 0:
        raise ValueError("Focus share must be 1–99% and total bytes must be positive")
    if code_focus_percent is not None and not 1 <= code_focus_percent < 100:
        raise ValueError("Code focus share must be 1–99%")
    if total_bytes > sum(entry["train_available_bytes"] for entry in entries.values()):
        raise ValueError("Requested total exceeds all available training text")
    names = list(entries)
    baseline = dict(zip(names, allocate_exact(total_bytes, [entries[name]["train_target_bytes"] for name in names])))
    plans = {"Base": baseline}
    for label, focus in FOCI.items():
        focus_names = set(DOMAIN_BUCKETS[focus])
        percent = code_focus_percent if label == "C" and code_focus_percent is not None else focus_percent
        focus_total = total_bytes * percent // 100
        others = [name for name in names if name not in focus_names]
        other_quota = allocate_exact(total_bytes - focus_total, [baseline[name] for name in others])
        focused = allocate_exact(focus_total, [baseline[name] for name in names if name in focus_names])
        mixture = dict(zip(others, other_quota))
        mixture.update(zip((name for name in names if name in focus_names), focused))
        plans[label] = {name: mixture[name] for name in names}
    for label, quota in plans.items():
        for name, amount in quota.items():
            if amount > entries[name]["train_available_bytes"]:
                raise ValueError(f"{label} needs {amount:,} bytes of {name}; only "
                                 f"{entries[name]['train_available_bytes']:,} available")
        if sum(quota.values()) != total_bytes:
            raise AssertionError(f"{label} does not sum to {total_bytes}")
    return {"source_recipe": recipe["name"], "total_train_bytes_per_mixture": total_bytes,
            "focus_percent": focus_percent, "code_focus_percent": code_focus_percent or focus_percent,
            "focus_domains": FOCI,
            "baseline": summarise(baseline, total_bytes),
            "mixtures": [summarise(plans[label], total_bytes, baseline, label, focus)
                         for label, focus in FOCI.items()]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-bytes", type=int, default=2_000_000_000)
    parser.add_argument("--focus-percent", type=int, default=FOCUS_PERCENT)
    parser.add_argument("--code-focus-percent", type=int)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    result = generate(recipe, args.total_bytes, args.focus_percent, args.code_focus_percent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for mixture in result["mixtures"]:
        focus = mixture["focus_domain"]
        print(f"{mixture['label']} · {focus}: {mixture['domain_percent'][focus]:.2f}% "
              f"(moved {mixture['moved_bytes'] / args.total_bytes * 100:.2f}% from Base)")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
