"""Download the completed 2 GB merge-noise models from the Modal volume."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretokenizer", choices=("regex", "line256"), required=True)
    parser.add_argument("--noise-rate", type=int, choices=(500_000, 750_000),
                        default=500_000, help="Noise rate in millionths")
    args = parser.parse_args()
    suffix = "_75" if args.noise_rate == 750_000 else ""
    manifests = json.loads((ROOT / f"results/merge_noise_all_{args.pretokenizer}{suffix}_2gb.json").read_text())
    if len(manifests) != 4:
        raise ValueError("Expected four completed models")
    env = {**os.environ, "MODAL_PROFILE": "group-cron"}
    for run in manifests:
        job_id = run["job_id"]
        destination = ROOT / "modal_results/tokenizers-merge-noise-all" / job_id / "base"
        destination.mkdir(parents=True, exist_ok=True)
        for filename in ("manifest.json", "tokenizer.json"):
            local = destination / filename
            if local.exists():
                continue
            remote = f"/tokenizers/merge-noise-all-v1/{job_id}/base/{filename}"
            subprocess.run(["modal", "volume", "get", "anytok-corpus", remote, str(local)],
                           check=True, env=env)
        saved = json.loads((destination / "manifest.json").read_text())
        if saved["job_id"] != job_id:
            raise ValueError(f"Downloaded the wrong manifest for {job_id}")
        print(f"Ready {args.pretokenizer} {run['settings']['noise_per_million']} "
              f"seed {run['settings']['noise_seed']}: {job_id}")


if __name__ == "__main__":
    main()
