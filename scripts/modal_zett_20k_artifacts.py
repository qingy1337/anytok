"""Move the fixed ZeTT 20k inputs and full checkpoints between Modal profiles.

Examples (from the repo root):
    python scripts/modal_zett_20k_artifacts.py upload projects
    python scripts/modal_zett_20k_artifacts.py status projects
    python scripts/modal_zett_20k_artifacts.py handoff projects grading
    python scripts/modal_zett_20k_artifacts.py collect

Each command selects MODAL_PROFILE for its own subprocess. Downloads use one
checkpoint at a time, so local staging needs about 25 GiB free. ``collect``
leaves every 500-step full checkpoint in test-joining's volume. This script
does not launch GPU work or request CPU or memory resources.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
VOLUME = "anytok-zett-20k"
PROFILES = ("projects", "grading", "test-joining")
INPUT_MANIFEST = ROOT / "plans/zett-modal-20k-inputs.json"
MODEL = (Path.home() / ".cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-1.7B"
         / "snapshots/effd688a12921b4cc83e3312b6feb579f70f9c71")
WARMUP = ROOT / "zett/output_smollm2_input_warmup_2048_retry/step-010000"


def modal(profile, *args):
    if profile not in PROFILES:
        raise ValueError(f"Unknown Modal profile: {profile}")
    env = {**os.environ, "MODAL_PROFILE": profile}
    result = subprocess.run(["modal", "volume", *args], env=env,
                            text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"{profile}: modal volume {' '.join(args)} failed:\n"
                           f"{result.stdout}\n{result.stderr}")
    return result.stdout


def ls(profile, remote):
    return json.loads(modal(profile, "ls", VOLUME, remote, "--json"))


def get(profile, remote, destination):
    modal(profile, "get", VOLUME, remote, str(destination))


def put(profile, local, remote, force=False):
    args = ["put"]
    if force:
        args.append("--force")
    modal(profile, *args, VOLUME, str(local), remote)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_steps(profile):
    try:
        entries = ls(profile, "run")
    except RuntimeError as exc:
        if any(phrase in str(exc).lower() for phrase in ("not found", "does not exist")):
            return []
        raise
    return sorted(int(Path(entry["filename"].rstrip("/")).name.removeprefix("step-"))
                  for entry in entries
                  if Path(entry["filename"].rstrip("/")).name.startswith("step-"))


def verify_checkpoint(local):
    manifest_path = local / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("kind") != "full_restart_checkpoint":
        raise ValueError(f"Not a full restart checkpoint: {local}")
    if local.name != f"step-{manifest['step']:06d}":
        raise ValueError(f"Step mismatch in {local}")
    for name, expected in manifest["files"].items():
        path = local / name
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Checkpoint file mismatch: {path}")
    return manifest


def transfer_checkpoint(source, target, step):
    name = f"step-{step:06d}"
    with tempfile.TemporaryDirectory(prefix="zett-modal-transfer-") as temporary:
        staging = Path(temporary)
        get(source, f"run/{name}", staging)
        local = staging / "run" / name
        manifest = verify_checkpoint(local)
        put(target, local, f"run/{name}")
        print(json.dumps({"stage": "transferred", "source": source,
                          "target": target, "step": step,
                          "bytes": sum(item["bytes"] for item in manifest["files"].values())}),
              flush=True)


def upload(profile):
    if not INPUT_MANIFEST.is_file():
        raise FileNotFoundError("First run: MODAL_PROFILE=projects modal run "
                                "scripts/modal_zett_20k.py --action manifest")
    manifest = json.loads(INPUT_MANIFEST.read_text())
    sources = [(ROOT / "data/balanced-full-v3.json", "input/recipe.json"),
               (ROOT / "zett/.joint_corpus", "input/corpus"),
               (ROOT / "modal_results/eval", "input/eval"),
               (WARMUP, "input/warmup")]
    for local, remote in sources:
        put(profile, local, remote, force=True)
        print(json.dumps({"stage": "uploaded", "profile": profile,
                          "remote": remote}), flush=True)
    for name in ("config.json", "model.safetensors", "tokenizer.json"):
        put(profile, MODEL / name, f"input/model/{name}", force=True)
        print(json.dumps({"stage": "uploaded", "profile": profile,
                          "remote": f"input/model/{name}"}), flush=True)
    put(profile, INPUT_MANIFEST, "input/manifest.json", force=True)
    print(json.dumps({"stage": "input_upload_complete", "profile": profile,
                      "files": len(manifest["files"]),
                      "bytes": sum(item["bytes"] for item in manifest["files"].values())}),
          flush=True)


def handoff(source, target):
    if PROFILES.index(target) != PROFILES.index(source) + 1:
        raise ValueError("Handoffs must follow projects -> grading -> test-joining")
    steps = checkpoint_steps(source)
    if not steps:
        raise ValueError(f"No checkpoint in {source}")
    if checkpoint_steps(target):
        raise ValueError(f"{target} already has checkpoints; inspect before handoff")
    with tempfile.TemporaryDirectory(prefix="zett-modal-handoff-") as temporary:
        staging = Path(temporary)
        for name in ("run_config.json", "heldout_tokenizers.json", "initial_eval.json",
                     "events.jsonl"):
            local = staging / "run" / name
            local.parent.mkdir(parents=True, exist_ok=True)
            get(source, f"run/{name}", local)
            put(target, local, f"run/{name}")
        transfer_checkpoint(source, target, steps[-1])
    print(json.dumps({"stage": "handoff_complete", "source": source,
                      "target": target, "resume_step": steps[-1]}), flush=True)


def collect():
    target = "test-joining"
    existing = set(checkpoint_steps(target))
    for source in PROFILES[:-1]:
        for step in checkpoint_steps(source):
            if step % 500:
                continue  # An off-grid handoff restart is not part of the 500-step archive.
            if step in existing:
                # The handoff checkpoint already exists in the next profile.
                with tempfile.TemporaryDirectory(prefix="zett-modal-compare-") as temporary:
                    staging = Path(temporary)
                    source_file = staging / "source-manifest.json"
                    get(source, f"run/step-{step:06d}/manifest.json", source_file)
                    source_manifest = sha256(source_file)
                    with tempfile.TemporaryDirectory(prefix="zett-modal-compare-target-") as other:
                        target_staging = Path(other)
                        target_file = target_staging / "target-manifest.json"
                        get(target, f"run/step-{step:06d}/manifest.json", target_file)
                        target_manifest = sha256(target_file)
                    if source_manifest != target_manifest:
                        raise ValueError(f"Conflicting checkpoint at step {step}")
                continue
            transfer_checkpoint(source, target, step)
            existing.add(step)
    grid = set(range(500, 20_001, 500))
    missing = sorted(grid - existing)
    if missing:
        raise ValueError(f"Final volume missing 500-step checkpoints: {missing}")
    print(json.dumps({"stage": "collection_complete", "profile": target,
                      "checkpoint_count": len(existing), "grid_count": len(grid)}),
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("upload", "status", "handoff", "collect"))
    parser.add_argument("profile", nargs="?", choices=PROFILES)
    parser.add_argument("target", nargs="?", choices=PROFILES)
    args = parser.parse_args()
    if args.action in ("upload", "status") and args.profile is None:
        parser.error("upload/status require a profile")
    if args.action == "handoff" and (args.profile is None or args.target is None):
        parser.error("handoff requires source and target profiles")
    if args.action == "upload":
        upload(args.profile)
    elif args.action == "status":
        print(json.dumps({"profile": args.profile, "volume": VOLUME,
                          "checkpoint_steps": checkpoint_steps(args.profile)}, indent=2))
    elif args.action == "handoff":
        handoff(args.profile, args.target)
    else:
        collect()


if __name__ == "__main__":
    main()
