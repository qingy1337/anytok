"""Move ZeTT restart checkpoints between Modal profiles through one HF model repo."""

import json
import os
from pathlib import Path
import hashlib
import shutil

import modal

APP_NAME = "anytok-zett-20k-hf-handoff"
REPO_ID = "qingy2024/anytok-zett-smollm2-20k"
VOLUME_NAME = "anytok-zett-20k"
MOUNT = "/mnt/zett"
PROFILES = ("projects", "grading", "test-joining")
RUN_FILES = ("run_config.json", "heldout_tokenizers.json", "initial_eval.json",
             "events.jsonl")
FULL_FILES = {"hypernet/config.json", "hypernet/model.safetensors", "lm.pt",
              "optimizer.pt", "state.pt"}

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, version=2)
secret = modal.Secret.from_name("huggingface-secret")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "huggingface_hub==1.28.0"
)


def hf_token():
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN",
                 "HUGGINGFACE_HUB_TOKEN"):
        if os.environ.get(name):
            return os.environ[name]
    raise RuntimeError("Hugging Face secret lacks a recognized token variable")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint(checkpoint, step=None):
    checkpoint = Path(checkpoint)
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("kind") != "full_restart_checkpoint"
            or manifest.get("step") != int(checkpoint.name.removeprefix("step-"))
            or (step is not None and manifest["step"] != step)
            or set(manifest["files"]) != FULL_FILES):
        raise ValueError(f"Expected a full restart checkpoint: {checkpoint}")
    for name, expected in manifest["files"].items():
        path = checkpoint / name
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {path}")
    return manifest, sha256(manifest_path)


@app.function(image=image, secrets=[secret], timeout=600)
def check_identity():
    import importlib.util
    from huggingface_hub import HfApi

    account = HfApi(token=hf_token()).whoami()["name"]
    result = {"account": account,
              "hf_xet_available": importlib.util.find_spec("hf_xet") is not None,
              "token_key_names": sorted(name for name in os.environ
                                        if "HUGGING" in name or name == "HF_TOKEN")}
    print(json.dumps(result), flush=True)
    return result


@app.function(image=image, volumes={MOUNT: volume}, secrets=[secret],
              timeout=4 * 3600)
def upload_latest():
    from huggingface_hub import HfApi

    run = Path(MOUNT) / "run"
    checkpoints = sorted(path for path in run.glob("step-*")
                         if path.is_dir() and (path / "state.pt").is_file())
    if not checkpoints:
        raise ValueError("No full checkpoint to publish")
    checkpoint = checkpoints[-1]
    manifest, manifest_hash = verify_checkpoint(checkpoint)
    step = manifest["step"]
    token = hf_token()
    api = HfApi(token=token)
    repo = api.repo_info(REPO_ID, repo_type="model")
    if not repo.private:
        raise ValueError("The checkpoint repository must be private")
    print(json.dumps({"stage": "uploading_checkpoint", "step": step,
                      "repo": REPO_ID,
                      "bytes": sum(item["bytes"] for item in manifest["files"].values())}),
          flush=True)
    path_in_repo = f"checkpoints/{checkpoint.name}"
    api.upload_folder(repo_id=REPO_ID, repo_type="model",
                      folder_path=str(checkpoint), path_in_repo=path_in_repo,
                      allow_patterns=list(manifest["files"]), token=token,
                      commit_message=f"Add step {step} restart weights")
    api.upload_file(repo_id=REPO_ID, repo_type="model", token=token,
                    path_or_fileobj=str(checkpoint / "manifest.json"),
                    path_in_repo=f"{path_in_repo}/manifest.json",
                    commit_message=f"Verify step {step} checkpoint manifest")
    metadata = {}
    for name in RUN_FILES:
        path = run / name
        metadata[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        api.upload_file(repo_id=REPO_ID, repo_type="model", token=token,
                        path_or_fileobj=str(path), path_in_repo=f"run/{name}",
                        commit_message=f"Save step {step} run metadata")
    pointer = {"step": step, "checkpoint_path": path_in_repo,
               "checkpoint_manifest_sha256": manifest_hash,
               "run_metadata": metadata}
    api.upload_file(repo_id=REPO_ID, repo_type="model", token=token,
                    path_or_fileobj=(json.dumps(pointer, indent=2,
                                                sort_keys=True) + "\n").encode(),
                    path_in_repo="latest.json",
                    commit_message=f"Set latest resumable checkpoint to {step}")
    result = {"stage": "published_latest", "step": step,
              "repo": REPO_ID, "manifest_sha256": manifest_hash}
    print(json.dumps(result), flush=True)
    return result


@app.function(image=image, volumes={MOUNT: volume}, secrets=[secret],
              timeout=4 * 3600)
def download_latest():
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    token = hf_token()
    api = HfApi(token=token)
    repo = api.repo_info(REPO_ID, repo_type="model")
    revision = repo.sha
    pointer_path = hf_hub_download(repo_id=REPO_ID, repo_type="model",
                                   filename="latest.json", revision=revision,
                                   token=token)
    pointer = json.loads(Path(pointer_path).read_text())
    step = int(pointer["step"])
    name = f"step-{step:06d}"
    base = f"checkpoints/{name}"
    if pointer["checkpoint_path"] != base:
        raise ValueError("Latest pointer has a wrong checkpoint path")
    if set(pointer["run_metadata"]) != set(RUN_FILES):
        raise ValueError("Latest pointer has unexpected run metadata")
    manifest_path = hf_hub_download(repo_id=REPO_ID, repo_type="model",
                                    filename=f"{base}/manifest.json",
                                    revision=revision, token=token)
    if sha256(manifest_path) != pointer["checkpoint_manifest_sha256"]:
        raise ValueError("Hugging Face manifest hash mismatch")
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest["kind"] != "full_restart_checkpoint" or manifest["step"] != step:
        raise ValueError("Latest pointer does not name a full checkpoint")
    paths = [f"{base}/{filename}" for filename in manifest["files"]]
    paths += [f"{base}/manifest.json"]
    paths += [f"run/{filename}" for filename in pointer["run_metadata"]]
    run = Path(MOUNT) / "run"
    stage = Path(MOUNT) / "hf-stage" / name
    stage.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"stage": "downloading_checkpoint", "step": step,
                      "repo": REPO_ID, "revision": revision}), flush=True)
    snapshot_download(repo_id=REPO_ID, repo_type="model", revision=revision,
                      token=token, allow_patterns=paths, local_dir=str(stage),
                      max_workers=8)
    saved = stage / base
    verify_checkpoint(saved, step)
    for filename, expected in pointer["run_metadata"].items():
        path = stage / "run" / filename
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Run metadata hash mismatch: {filename}")
    run.mkdir(exist_ok=True)
    target = run / name
    if target.exists():
        shutil.rmtree(target)
    os.replace(saved, target)
    for filename in pointer["run_metadata"]:
        os.replace(stage / "run" / filename, run / filename)
    shutil.rmtree(stage)
    volume.commit()
    verify_checkpoint(target, step)
    result = {"stage": "downloaded_latest", "step": step,
              "repo": REPO_ID, "manifest_sha256": pointer["checkpoint_manifest_sha256"]}
    print(json.dumps(result), flush=True)
    return result


@app.local_entrypoint()
def main(action: str = "check"):
    profile = os.environ.get("MODAL_PROFILE")
    if profile not in PROFILES:
        raise ValueError(f"Set MODAL_PROFILE to one of {PROFILES}")
    if action == "check":
        print(json.dumps(check_identity.remote(), indent=2))
    elif action == "upload":
        print(json.dumps(upload_latest.remote(), indent=2))
    elif action == "download":
        print(json.dumps(download_latest.remote(), indent=2))
    else:
        raise ValueError(f"Unknown action: {action}")
