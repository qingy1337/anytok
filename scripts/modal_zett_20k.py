"""Train ZeTT to 20k on Modal B300s, keeping full checkpoints every 500 steps.

Use MODAL_PROFILE=projects, grading, then test-joining. ``--action plan`` prints
the next run without starting a worker; ``--action smoke`` checks B300 imports;
``--action train`` resumes the latest full checkpoint in this profile's volume.
Each profile has its own ``anytok-zett-20k`` volume. The inputs live under
``input/`` and the run under ``run/``. Transfer the latest full checkpoint and
the run metadata to the next profile before resuming. Collect every full
checkpoint into the test-joining volume at the end. The Function requests only
``gpu="B300"``; it does not reserve CPU or memory explicitly.
"""

import hashlib
import json
import os
from pathlib import Path
import sys

import modal

APP_NAME = "anytok-zett-20k-b300"
VOLUME_NAME = "anytok-zett-20k"
MOUNT = Path("/mnt/zett")
CODE = Path("/opt/zett")
SOURCE = Path(__file__).resolve().parents[1] / "runtime/zett_20k"
REPO = SOURCE.parents[1]
INPUT_MANIFEST = REPO / "plans/zett-modal-20k-inputs.json"
LOCAL_MODEL = (Path.home() / ".cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-1.7B"
               / "snapshots/effd688a12921b4cc83e3312b6feb579f70f9c71")
LOCAL_WARMUP = REPO / "zett/output_smollm2_input_warmup_2048_retry/step-010000"
PROFILES = ("projects", "grading", "test-joining")
FINAL_STEP = 20_000
CHECKPOINT_INTERVAL = 500

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, version=2)
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


def verify_checkpoint_files(step):
    checkpoint = MOUNT / "run" / f"step-{step:06d}"
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    if manifest.get("kind") != "full_restart_checkpoint" or manifest.get("step") != step:
        raise ValueError(f"Invalid checkpoint manifest: {checkpoint}")
    for name, expected in manifest["files"].items():
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Missing or wrong-sized checkpoint file: {path}")
        if sha256(path) != expected["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {path}")
    return {"step": step, "files": len(manifest["files"]),
            "bytes": sum(item["bytes"] for item in manifest["files"].values())}


@app.function(image=audit_image, volumes={str(MOUNT): volume}, timeout=3600)
def audit_inputs():
    result = verify_input_files()
    print(json.dumps({"stage": "verified_inputs", **result}), flush=True)
    return result


@app.function(image=audit_image, volumes={str(MOUNT): volume}, timeout=3600)
def audit_checkpoint(step: int):
    result = verify_checkpoint_files(step)
    print(json.dumps({"stage": "verified_checkpoint", **result}), flush=True)
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
def train_segment(stop_at: int, max_seconds: int = 10800):
    """Resume to stop_at or the time cap, committing every full checkpoint."""
    if not 500 <= stop_at <= FINAL_STEP or stop_at % CHECKPOINT_INTERVAL:
        raise ValueError("stop_at must be a multiple of 500 through 20000")
    if not 60 <= max_seconds <= 3 * 3600:
        raise ValueError("max_seconds must be 60..10800 for credit safety")
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
    if existing and int(existing[-1].name.removeprefix("step-")) >= stop_at:
        raise ValueError("The latest full checkpoint is already at or beyond stop_at")
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
                "--microbatch-size", "32", "--save-every", str(CHECKPOINT_INTERVAL),
                "--protected-steps", *map(str, range(CHECKPOINT_INTERVAL, FINAL_STEP + 1,
                                                    CHECKPOINT_INTERVAL)),
                "--eval-every", str(CHECKPOINT_INTERVAL), "--stop-at-update", str(stop_at),
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
def main(action: str = "plan", stop_at: int = FINAL_STEP,
         max_seconds: int = 10800, step: int = 0):
    profile = os.environ.get("MODAL_PROFILE")
    if profile not in PROFILES:
        raise ValueError(f"Set MODAL_PROFILE to one of {PROFILES}")
    if action == "plan":
        print(json.dumps({"profile": profile, "volume": VOLUME_NAME,
                          "gpu": "B300", "cpu_request": None,
                          "memory_request": None,
                          "save_every": CHECKPOINT_INTERVAL,
                          "full_checkpoints_retained": True,
                          "stop_at": stop_at, "max_seconds": max_seconds,
                          "action": "train"}, indent=2))
    elif action == "manifest":
        print(json.dumps(make_input_manifest(), indent=2))
    elif action == "audit":
        print(json.dumps(audit_inputs.remote(), indent=2))
    elif action == "audit-checkpoint":
        if step < 1:
            raise ValueError("audit-checkpoint requires --step")
        print(json.dumps(audit_checkpoint.remote(step), indent=2))
    elif action == "smoke":
        print(json.dumps(smoke.remote(), indent=2))
    elif action == "train":
        print(json.dumps(train_segment.remote(stop_at, max_seconds), indent=2))
    else:
        raise ValueError("action must be plan, manifest, audit, audit-checkpoint, smoke, or train")
