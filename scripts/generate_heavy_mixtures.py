"""Plan six maximum whole-percent focus mixtures using saved corpus pools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_mixture_examples import DOMAIN_BUCKETS, allocate_exact, summarise


ROOT = Path(__file__).resolve().parent.parent
FOCI = {
    "Papers25": ("papers", 25),
    "Math50": ("math", 50),
    "QA25": ("qa", 25),
    "Books25": ("books", 25),
    "Medical40": ("medical", 40),
    "Legal40": ("legal", 40),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-bytes", type=int, default=2_000_000_000)
    parser.add_argument("--output", type=Path, default=ROOT / "results/heavy_domain_mixtures.json")
    args = parser.parse_args()
    recipe = json.loads((ROOT / "data/balanced-full-v3.json").read_text())
    current = json.loads((ROOT / "results/mixture_examples.json").read_text())
    total = args.total_bytes
    if total <= 0:
        raise ValueError("Total bytes must be positive")
    current_baseline = current["baseline"]["bucket_bytes"]
    scaled = allocate_exact(total, list(current_baseline.values()))
    baseline = dict(zip(current_baseline, scaled))
    available = {bucket["name"]: bucket["train_available_bytes"]
                 for bucket in recipe["buckets"]}
    if current["source_recipe"] != recipe["name"] or set(baseline) != set(available):
        raise ValueError("Baseline and corpus recipe disagree")

    mixtures = []
    for label, (focus, percent) in FOCI.items():
        focused_names = [name for name in baseline if name in DOMAIN_BUCKETS[focus]]
        other_names = [name for name in baseline if name not in DOMAIN_BUCKETS[focus]]
        focused = allocate_exact(total * percent // 100,
                                 [baseline[name] for name in focused_names])
        others = allocate_exact(total * (100 - percent) // 100,
                                [baseline[name] for name in other_names])
        quota = {**dict(zip(focused_names, focused)), **dict(zip(other_names, others))}
        quota = {name: quota[name] for name in baseline}
        if sum(quota.values()) != total:
            raise AssertionError(f"{label} quota does not total {total}")
        for name, amount in quota.items():
            if amount > available[name]:
                raise ValueError(f"{label}: {name} needs {amount:,}; available {available[name]:,}")
        mixtures.append(summarise(quota, total, baseline, label, focus))
        print(f"{label}: {percent}% {focus}, {total:,} bytes total")

    result = {"source_recipe": recipe["name"],
              "total_train_bytes_per_mixture": total,
              "baseline": summarise(baseline, total), "mixtures": mixtures}
    path = args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
