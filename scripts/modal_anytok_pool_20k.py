"""Train the uniform 1000-tokenizer pool run to 20k on B300.

Fresh run from the pinned original LM and identity-warmed hypernetwork.
The default controller is python scripts/run_anytok_pool_20k.py. This launcher
runs individual bounded GPU segments; it never changes the previous ZeTT run.
Weights every 500 updates; full restarts every 2000 and at segment boundaries.
No CPU or memory resource reservation is specified.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import modal

APP_NAME = "anytok-pool-20k-b300"
VOLUME_NAME = "anytok-pool-20k"
MOUNT = Path("/mnt/zett")
CODE = Path("/opt/zett")
SOURCE = Path(__file__).resolve().parents[1] / "runtime/anytok_pool_20k"
REPO = SOURCE.parents[1]
INPUT_MANIFEST = REPO / "plans/zett-modal-20k-inputs.json"
LOCAL_MODEL = (Path.home() / ".cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-1.7B"
               / "snapshots/effd688a12921b4cc83e3312b6feb579f70f9c71")
LOCAL_WARMUP = REPO / "zett/output_smollm2_input_warmup_2048_retry/step-010000"
from anytok_pool_config import PROFILES
FINAL_STEP = 20_000
CHECKPOINT_INTERVAL = 500
RESUME_INTERVAL = 2000
WEIGHT_FILES = ("hypernet/config.json", "hypernet/model.safetensors", "lm.pt")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)
image = (
    modal.Image.from_registry("ubuntu:24.04", add_python="3.12")
    .apt_install("libpython3.12")
    .uv_pip_install("torch==2.15.0", "transformers==5.15.1",
                    "tokenizers==0.22.2", "numpy==2.3.5",
                    "safetensors==0.8.0", "accelerate==1.14.0")
    .add_local_dir(str(SOURCE / "hf_hypernet"),
                   remote_path=str(CODE / "hf_hypernet"))
    .add_local_file(str(SOURCE / "scripts/train_joint_smollm_input.py"),
                    remote_path=str(CODE / "scripts/train_joint_smollm_input.py"))
    .add_local_file(str(SOURCE / "rust_utils/target/release/librust_utils.so"),
                    remote_path=str(CODE / "rust_utils/target/release/librust_utils.so"))
)
audit_image = modal.Image.from_registry("ubuntu:24.04", add_python="3.12")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def local_input_files():
    files = {"recipe.json": REPO / "data/balanced-full-v3.json",
             "corpus/manifest.json": REPO / "zett/.joint_corpus/manifest.json"}
    corpus_manifest = json.loads(files["corpus/manifest.json"].read_text())
    for bucket in sorted(corpus_manifest):
        files[f"corpus/{bucket}/train.jsonl.gz"] = (
            REPO / "zett/.joint_corpus" / bucket / "train.jsonl.gz")
    for path in sorted((REPO / "modal_results/eval").glob("*.jsonl.gz")):
        files[f"eval/{path.name}"] = path
    for name in ("config.json", "model.safetensors", "tokenizer.json"):
        files[f"model/{name}"] = LOCAL_MODEL / name
    for name in ("config.json", "model.safetensors"):
        files[f"warmup/{name}"] = LOCAL_WARMUP / name
    if len(corpus_manifest) != 23 or len([key for key in files if key.startswith("eval/")]) < 23:
        raise ValueError("Missing training or evaluation buckets")
    return files


def make_input_manifest():
    files = local_input_files()
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    manifest = {"kind": "zett_20k_inputs", "files": {
        remote: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for remote, path in files.items()}}
    INPUT_MANIFEST.parent.mkdir(exist_ok=True)
    INPUT_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"path": str(INPUT_MANIFEST), "file_count": len(files),
            "bytes": sum(item["bytes"] for item in manifest["files"].values())}


def verify_input_files():
    manifest = json.loads((MOUNT / "input/manifest.json").read_text())
    for name, expected in manifest["files"].items():
        path = MOUNT / "input" / name
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Missing or wrong-sized input: {name}")
        if sha256(path) != expected["sha256"]:
            raise ValueError(f"Checksum mismatch: {name}")
    return {"file_count": len(manifest["files"]),
            "bytes": sum(item["bytes"] for item in manifest["files"].values()),
            "manifest_sha256": sha256(MOUNT / "input/manifest.json")}


def verify_checkpoint_files(step, expected_kind=None):
    checkpoint = MOUNT / "run" / f"step-{step:06d}"
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    kind = manifest.get("kind")
    if (kind not in ("full_restart_checkpoint", "weights_only_checkpoint")
            or manifest.get("step") != step
            or (expected_kind is not None and kind != expected_kind)):
        raise ValueError(f"Invalid checkpoint manifest: {checkpoint}")
    expected_names = (set(WEIGHT_FILES) if kind == "weights_only_checkpoint"
                      else set(WEIGHT_FILES) | {"optimizer.pt", "state.pt"})
    if set(manifest["files"]) != expected_names:
        raise ValueError(f"Unexpected checkpoint files: {checkpoint}")
    for name, expected in manifest["files"].items():
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Missing or wrong-sized checkpoint file: {path}")
        if sha256(path) != expected["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {path}")
    return {"step": step, "kind": kind, "files": len(manifest["files"]),
            "bytes": sum(item["bytes"] for item in manifest["files"].values())}


def compact_grid_checkpoints(run, keep_latest=True):
    """Keep the newest restart point until a newer full checkpoint is safe."""
    checkpoints = sorted(path for path in run.glob("step-*") if path.is_dir())
    full = [path for path in checkpoints if (path / "state.pt").is_file()]
    latest_full = full[-1] if full else None
    changed = []
    for checkpoint in checkpoints:
        step = int(checkpoint.name.removeprefix("step-"))
        if step % CHECKPOINT_INTERVAL:
            if not keep_latest:
                shutil.rmtree(checkpoint)
                changed.append(step)
            continue
        if (step % RESUME_INTERVAL == 0
                or (keep_latest and checkpoint == latest_full)):
            continue
        manifest_path = checkpoint / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["kind"] == "weights_only_checkpoint":
            continue
        if manifest["kind"] != "full_restart_checkpoint":
            raise ValueError(f"Unknown checkpoint kind at {checkpoint}")
        for name in WEIGHT_FILES:
            expected = manifest["files"][name]
            path = checkpoint / name
            if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
                raise ValueError(f"Weight checksum mismatch: {path}")
        manifest["kind"] = "weights_only_checkpoint"
        manifest["files"] = {name: manifest["files"][name] for name in WEIGHT_FILES}
        pending = checkpoint / ".manifest.weights.tmp"
        pending.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(pending, manifest_path)
        (checkpoint / "optimizer.pt").unlink(missing_ok=True)
        (checkpoint / "state.pt").unlink(missing_ok=True)
        changed.append(step)
    return changed


def rebind_warmup_volume_path(run, current_path):
    """Replace only the profile-specific backing path for the same warmup input."""
    config_path = run / "run_config.json"
    config = json.loads(config_path.read_text())
    old_path = config["warmup_checkpoint"]
    if old_path == current_path:
        return False
    suffix = "/input/warmup"
    prefix = "/__modal/volumes/"
    if not (old_path.startswith(prefix) and old_path.endswith(suffix)
            and current_path.startswith(prefix) and current_path.endswith(suffix)):
        raise ValueError("Unexpected warmup checkpoint path change")
    config["warmup_checkpoint"] = current_path
    pending = config_path.with_suffix(".tmp")
    pending.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    os.replace(pending, config_path)
    event = {"stage": "warmup_volume_path_rebound", "old_path": old_path,
             "new_path": current_path}
    with (run / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
    print(json.dumps(event), flush=True)
    return True


@app.function(image=audit_image, volumes={str(MOUNT): volume}, timeout=3600)
def audit_inputs():
    result = verify_input_files()
    print(json.dumps({"stage": "verified_inputs", **result}), flush=True)
    return result


@app.function(image=audit_image, volumes={str(MOUNT): volume}, timeout=3600)
def audit_checkpoint(step: int):
    result = verify_checkpoint_files(step, "full_restart_checkpoint")
    print(json.dumps({"stage": "verified_checkpoint", **result}), flush=True)
    return result


@app.function(image=audit_image, volumes={str(MOUNT): volume}, timeout=4 * 3600)
def compact_checkpoints(keep_latest: bool = True):
    changed = compact_grid_checkpoints(MOUNT / "run", keep_latest)
    volume.commit()
    result = {"stage": "compacted_checkpoints", "steps": changed,
              "kept_latest_full": keep_latest}
    print(json.dumps(result), flush=True)
    return result


@app.function(image=audit_image, volumes={str(MOUNT): volume}, timeout=4 * 3600)
def audit_final_grid():
    total = 0
    for step in range(CHECKPOINT_INTERVAL, FINAL_STEP + 1, CHECKPOINT_INTERVAL):
        expected = ("full_restart_checkpoint" if step % RESUME_INTERVAL == 0
                    else "weights_only_checkpoint")
        result = verify_checkpoint_files(step, expected)
        total += result["bytes"]
        if step % 2500 == 0:
            print(json.dumps({"stage": "audited_grid", "step": step}), flush=True)
    result = {"checkpoint_count": FINAL_STEP // CHECKPOINT_INTERVAL,
              "full_checkpoint_count": FINAL_STEP // RESUME_INTERVAL,
              "final_step": FINAL_STEP, "checkpoint_bytes": total,
              "input_manifest_sha256": sha256(MOUNT / "input/manifest.json")}
    (MOUNT / "run/final_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.function(image=image, gpu="B300", timeout=900)
def smoke():
    import torch
    sys.path.insert(0, str(CODE))
    from hf_hypernet.modeling_hypernet import ZettHypernet  # noqa: F401
    import scripts.train_joint_smollm_input as trainer  # noqa: F401

    if not torch.cuda.is_available():
        raise RuntimeError("B300 CUDA unavailable")
    left = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    right = left @ left
    torch.cuda.synchronize()
    result = {"gpu": torch.cuda.get_device_name(0),
              "capability": list(torch.cuda.get_device_capability(0)),
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "matmul_finite": bool(torch.isfinite(right).all().item())}
    if "B300" not in result["gpu"] or not result["matmul_finite"]:
        raise RuntimeError(f"Unexpected B300 smoke result: {result}")
    print(json.dumps(result), flush=True)
    return result


@app.function(image=image, gpu="B300", volumes={str(MOUNT): volume},
              timeout=4 * 3600)
def train_segment(max_seconds: int = 12600):
    """Resume to 20k; keep the newest restart and 2,000-step milestones."""
    if not 60 <= max_seconds <= 12600:
        raise ValueError("max_seconds must be 60..12600 for credit safety")
    import signal
    import threading
    import time
    import torch

    sys.path.insert(0, str(CODE))
    import scripts.train_joint_smollm_input as trainer

    required = [MOUNT / "input/manifest.json", MOUNT / "input/recipe.json",
                MOUNT / "input/corpus/manifest.json",
                MOUNT / "input/model/config.json", MOUNT / "input/model/model.safetensors",
                MOUNT / "input/model/tokenizer.json"]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError([str(path) for path in required if not path.is_file()])
    print(json.dumps({"stage": "verified_inputs", **verify_input_files()}), flush=True)
    run = MOUNT / "run"
    run.mkdir(exist_ok=True)
    existing = sorted(path for path in run.glob("step-*")
                      if path.is_dir() and (path / "state.pt").is_file())
    if existing:
        latest_step = int(existing[-1].name.removeprefix("step-"))
        print(json.dumps({"stage": "verified_restart",
                          **verify_checkpoint_files(latest_step)}), flush=True)
        if rebind_warmup_volume_path(
                run, str((MOUNT / "input/warmup").resolve())):
            volume.commit()
    if existing and int(existing[-1].name.removeprefix("step-")) >= FINAL_STEP:
        raise ValueError("The latest full checkpoint is already at 20k")
    if not existing and not (MOUNT / "input/warmup/config.json").is_file():
        raise FileNotFoundError("First profile needs the identity warmup checkpoint")

    start = time.monotonic()
    timer = threading.Timer(max_seconds, lambda: os.kill(os.getpid(), signal.SIGTERM))
    timer.daemon = True
    original_save = trainer.save_checkpoint

    def save_and_commit(output, step, *args, **kwargs):
        original_save(output, step, *args, **kwargs)
        checkpoint = Path(output) / f"step-{step:06d}"
        names = sorted(path.relative_to(checkpoint).as_posix()
                       for path in checkpoint.rglob("*") if path.is_file())
        manifest = {"step": step, "kind": "full_restart_checkpoint",
                    "files": {name: {"bytes": (checkpoint / name).stat().st_size,
                                     "sha256": sha256(checkpoint / name)}
                              for name in names},
                    "trainer_sha256": sha256(CODE / "scripts/train_joint_smollm_input.py")}
        (checkpoint / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        volume.commit()
        compacted = compact_grid_checkpoints(run)
        if compacted:
            volume.commit()
            print(json.dumps({"stage": "compacted_checkpoints", "steps": compacted}),
                  flush=True)
        print(json.dumps({"stage": "committed_checkpoint", "step": step,
                          "elapsed_seconds": round(time.monotonic() - start, 2)}),
              flush=True)

    trainer.save_checkpoint = save_and_commit
    old_argv = sys.argv
    sys.argv = [str(CODE / "scripts/train_joint_smollm_input.py"),
                "--recipe", str(MOUNT / "input/recipe.json"),
                "--data-dir", str(MOUNT / "input/corpus"),
                "--data-manifest", str(MOUNT / "input/corpus/manifest.json"),
                "--eval-dir", str(MOUNT / "input/eval"),
                "--output", str(run),
                "--warmup-checkpoint", str(MOUNT / "input/warmup"),
                "--model", str(MOUNT / "input/model"),
                "--tokenizer-pool", str(MOUNT / "input/pool"),
                "--microbatch-size", "32", "--save-every", str(CHECKPOINT_INTERVAL),
                "--protected-steps", *map(str, range(CHECKPOINT_INTERVAL, FINAL_STEP + 1,
                                                    CHECKPOINT_INTERVAL)),
                "--eval-every", str(CHECKPOINT_INTERVAL),
                "--stop-at-update", str(FINAL_STEP),
                "--seed", "42"]
    if existing:
        sys.argv.append("--resume")
    timer.start()
    try:
        trainer.main()
    finally:
        timer.cancel()
        sys.argv = old_argv
        volume.commit()
    completed = sorted(path for path in run.glob("step-*")
                       if path.is_dir() and (path / "state.pt").is_file())
    latest = completed[-1]
    state = torch.load(latest / "state.pt", map_location="cpu", weights_only=True)
    result = {"latest_step": state["step"],
              "latest_checkpoint": str(latest),
              "elapsed_seconds": round(time.monotonic() - start, 2),
              "full_checkpoint_count": len(completed)}
    print(json.dumps(result), flush=True)
    return result


@app.local_entrypoint()
def main(action: str = "plan", max_seconds: int = 12600, step: int = 0):
    profile = os.environ.get("MODAL_PROFILE")
    if profile not in PROFILES:
        raise ValueError(f"Set MODAL_PROFILE to one of {PROFILES}")
    if action == "plan":
        print(json.dumps({"profile": profile, "volume": VOLUME_NAME,
                          "recipe": "balanced-full-v3",
                          "model_revision": "effd688a12921b4cc83e3312b6feb579f70f9c71",
                          "input_manifest_sha256": (sha256(INPUT_MANIFEST)
                                                    if INPUT_MANIFEST.is_file() else None),
                          "remote_input": str(MOUNT / "input"),
                          "remote_output": str(MOUNT / "run"),
                          "gpu": "B300", "cpu_request": None,
                          "memory_request": None,
                          "save_every": CHECKPOINT_INTERVAL,
                          "full_checkpoint_every": RESUME_INTERVAL,
                          "weights_only_every": CHECKPOINT_INTERVAL,
                          "latest_checkpoint_kept_full": True,
                          "stop_at": FINAL_STEP, "max_seconds": max_seconds,
                          "command": (f"MODAL_PROFILE={profile} modal run --detach "
                                      "scripts/modal_anytok_pool_20k.py --action train "
                                      f"--max-seconds {max_seconds}")},
                         indent=2))
    elif action == "manifest":
        print(json.dumps(make_input_manifest(), indent=2))
    elif action == "audit":
        print(json.dumps(audit_inputs.remote(), indent=2))
    elif action == "audit-checkpoint":
        if step < 1:
            raise ValueError("audit-checkpoint requires --step")
        print(json.dumps(audit_checkpoint.remote(step), indent=2))
    elif action == "compact":
        print(json.dumps(compact_checkpoints.remote(), indent=2))
    elif action == "compact-source":
        print(json.dumps(compact_checkpoints.remote(False), indent=2))
    elif action == "audit-final":
        print(json.dumps(audit_final_grid.remote(), indent=2))
    elif action == "smoke":
        print(json.dumps(smoke.remote(), indent=2))
    elif action == "train":
        print(json.dumps(train_segment.remote(max_seconds), indent=2))
    else:
        raise ValueError("Unknown action")
