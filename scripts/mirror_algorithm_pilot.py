"""Download the matched WordPiece and Unigram models from Modal."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Download the 2 GB WordPiece and Unigram models")
    args = parser.parse_args()
    environment = {**os.environ, "MODAL_PROFILE": "group-cron"}
    scale = "2gb" if args.full else "200m"
    algorithms = ("wordpiece", "unigram")
    for algorithm in algorithms:
        report = json.loads((ROOT / f"results/algorithm_{algorithm}_regex_{scale}.json").read_text())
        manifest = report["models"]["Base"]
        destination = ROOT / f"modal_results/tokenizers-algorithm-{scale}" / algorithm
        destination.mkdir(parents=True, exist_ok=True)
        for filename in ("manifest.json", "tokenizer.json"):
            local = destination / filename
            if not local.exists():
                remote = str(Path(manifest["model_path"]).with_name(filename)).removeprefix("/corpus")
                subprocess.run(["modal", "volume", "get", "anytok-corpus", remote, str(local)],
                               check=True, env=environment)
        saved = json.loads((destination / "manifest.json").read_text())
        if saved["run_id"] != manifest["run_id"]:
            raise ValueError(f"Wrong model downloaded for {algorithm}")
        print(f"Ready {algorithm}: {saved['run_id']}")


if __name__ == "__main__":
    main()
