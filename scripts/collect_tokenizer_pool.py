"""Wait for recorded pool shards, then gather them into group-cron's Volume.

    python scripts/collect_tokenizer_pool.py

Only metadata is read locally; the transfer helper copies model bytes between
Modal workers and Volume APIs. A failed shard is reported without being retried.
"""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "modal_results/tokenizer_pool"


def main():
    assignments = json.loads((STATE / "regex-shards.json").read_text())["assignments"]
    saved_path = STATE / "collected-shards.json"
    saved = json.loads(saved_path.read_text()) if saved_path.exists() else {}
    collected = set(saved.get("collected_profiles", [])) & assignments.keys()
    failed = set(saved.get("failed_profiles", [])) & assignments.keys()
    deadline = time.monotonic() + 8 * 3600
    while len(collected | failed) < len(assignments):
        if time.monotonic() > deadline:
            raise TimeoutError("Timed out waiting for pool shards")
        for profile, indices in assignments.items():
            if profile in collected | failed:
                continue
            path = STATE / profile / "status.json"
            if not path.exists():
                continue
            status = json.loads(path.read_text())
            if status["stage"] in {"failed", "budget_stop"}:
                print(json.dumps({"stage": "shard_not_collected", "profile": profile,
                                  "status": status}), flush=True)
                failed.add(profile)
            elif status["stage"] == "complete":
                manifests = [json.loads(p.read_text()) for p in (STATE / profile).glob("[0-9][0-9][0-9][0-9].json")]
                actual = {m["config"]["index"] for m in manifests if m["status"] == "complete"}
                if not set(indices) <= actual:
                    raise RuntimeError(f"Incomplete local shard metadata for {profile}")
                print(json.dumps({"stage": "gathering", "profile": profile,
                                  "tokenizers": len(indices)}), flush=True)
                for attempt in range(4):
                    try:
                        subprocess.run(["modal", "run", "--detach", "scripts/modal_tokenizer_pool_transfer.py",
                            "--action", "gather", "--source-profile", profile],
                            env={**os.environ, "MODAL_PROFILE": "group-cron"}, cwd=ROOT, check=True)
                        break
                    except subprocess.CalledProcessError:
                        if attempt == 3:
                            raise
                        time.sleep(2 ** attempt)
                collected.add(profile)
                temporary = saved_path.with_suffix(".tmp")
                temporary.write_text(json.dumps({
                    "collected_profiles": sorted(collected), "failed_profiles": sorted(failed)}, indent=2) + "\n")
                temporary.replace(saved_path)
        if len(collected | failed) < len(assignments):
            time.sleep(30)
    if failed:
        raise RuntimeError(f"Some shards failed: {sorted(failed)}")


if __name__ == "__main__":
    main()
