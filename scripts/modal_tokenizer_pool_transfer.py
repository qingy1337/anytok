"""Copy pool corpus and outputs directly between enabled Modal workspaces.

Prepare compact corpus prefixes once on group-cron, then import into each
training profile. Only metadata passes through the local caller. Data streams
between Modal containers and Volume APIs, authenticated by ephemeral Secrets.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import modal
from modal.config import Config
from grpclib.exceptions import StreamTerminatedError

from tokenizer_pool import digest

PROJECT = Path(__file__).resolve().parents[1]
PLAN_PATH = PROJECT / "plans/tokenizer-pool-1000-v1.json"
DISABLED = {"general", "research-everettcc"}
OUTPUT_VOLUME = "anytok-tokenizer-pool"
app = modal.App("anytok-tokenizer-pool-transfer")
corpus = modal.Volume.from_name("anytok-corpus", create_if_missing=True)
outputs = modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True, version=2)
image = modal.Image.debian_slim(python_version="3.11").add_local_python_source("tokenizer_pool")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def retry_read(operation, attempts=4):
    for attempt in range(attempts):
        try:
            return operation()
        except (StreamTerminatedError, OSError, TimeoutError, modal.exception.ConnectionError):
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)


def gather_items(source, plan, pool_prefix):
    # Close the listing stream before issuing nested file reads. Keeping it
    # open across hundreds of slow RPCs can lose the connection mid-listing.
    entries = retry_read(lambda: list(source.iterdir(pool_prefix + "/tokenizers", recursive=True)))
    manifests = [entry for entry in entries if entry.path.endswith("/manifest.json")]

    def read_manifest(entry):
        raw = retry_read(lambda: b"".join(source.read_file(entry.path)))
        manifest = json.loads(raw)
        if (manifest["status"] != "complete" or
                manifest["config"] != plan["configs"][manifest["config"]["index"]]):
            raise RuntimeError("Source tokenizer configuration mismatch")
        model = entry.path.removesuffix("manifest.json") + "tokenizer.json"
        return [
            {"source_path": entry.path, "destination_path": entry.path,
             "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
            {"source_path": model, "destination_path": model,
             "bytes": manifest["tokenizer_bytes"], "sha256": manifest["tokenizer_sha256"]},
        ]

    with ThreadPoolExecutor(max_workers=8) as executor:
        return [item for pair in executor.map(read_manifest, manifests) for item in pair]


def source_client(profile):
    if profile in DISABLED:
        raise ValueError(f"Disabled profile: {profile}")
    config = Config()
    values = {key: config.get(key, profile=profile, use_env=False)
              for key in ("token_id", "token_secret")}
    if not all(values.values()):
        raise ValueError(f"Missing credentials for {profile}")
    client = modal.Client.from_credentials(values["token_id"], values["token_secret"])
    secret = modal.Secret.from_dict({"SOURCE_MODAL_TOKEN_ID": values["token_id"],
                                     "SOURCE_MODAL_TOKEN_SECRET": values["token_secret"]})
    return client, secret


@app.function(image=image, volumes={"/corpus": corpus.with_mount_options(read_only=True),
                                   "/pool": outputs}, max_containers=100, timeout=3600)
def prepare_one(source, required_bytes, pool_prefix):
    relative = str(Path(source["source_path"]).relative_to("/corpus") / "train.jsonl.gz")
    target = Path("/pool") / pool_prefix / "corpus" / relative
    manifest_path = target.with_suffix(".manifest.json")
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved["required_bytes"] == required_bytes and sha256(target) == saved["sha256"]:
            return saved
        raise RuntimeError("Compact corpus source mismatch")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".partial")
    used = documents = 0
    # Keep original JSONL records and the entire last document, preserving all
    # per-tokenizer prefixes and document boundaries of the original corpus.
    with gzip.open(Path(source["source_path"]) / "train.jsonl.gz", "rt", encoding="utf-8") as src:
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as dst:
            for line in src:
                if used >= required_bytes:
                    break
                used += len(json.loads(line)["text"].encode("utf-8"))
                documents += 1
                dst.write(line)
    if used < required_bytes:
        raise RuntimeError(f"Insufficient source data for {source['name']}")
    temporary.replace(target)
    saved = {"bucket": source["name"], "relative_path": relative,
             "source_path": f"{pool_prefix}/corpus/{relative}",
             "required_bytes": required_bytes, "available_bytes": used,
             "documents": documents, "bytes": target.stat().st_size,
             "sha256": sha256(target)}
    manifest_path.write_text(json.dumps(saved, indent=2) + "\n")
    outputs.commit()
    print(json.dumps({"stage": "prepared", **saved}), flush=True)
    return saved


@app.function(image=image, volumes={"/corpus": corpus, "/pool": outputs},
              max_containers=100, timeout=3600)
def copy_file(item, source_volume_name, destination_kind):
    client = modal.Client.from_credentials(os.environ["SOURCE_MODAL_TOKEN_ID"],
                                          os.environ["SOURCE_MODAL_TOKEN_SECRET"])
    source_volume = modal.Volume.from_name(source_volume_name, version=2, client=client)
    relative = Path(item["destination_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Invalid destination path")
    volume = corpus if destination_kind == "corpus" else outputs
    destination = Path("/corpus" if destination_kind == "corpus" else "/pool") / relative
    if destination.exists() and destination.stat().st_size == item["bytes"] and sha256(destination) == item["sha256"]:
        return {"path": str(relative), "reused": True}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".partial")

    def download():
        h = hashlib.sha256()
        with temporary.open("wb") as stream:
            for chunk in source_volume.read_file(item["source_path"]):
                stream.write(chunk)
                h.update(chunk)
        return h

    try:
        h = retry_read(download)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if temporary.stat().st_size != item["bytes"] or h.hexdigest() != item["sha256"]:
        temporary.unlink()
        raise RuntimeError(f"Transferred checksum mismatch: {relative}")
    temporary.replace(destination)
    volume.commit()
    return {"path": str(relative), "bytes": item["bytes"], "sha256": h.hexdigest()}


@app.local_entrypoint()
def main(action: str = "prepare", source_profile: str = "group-cron"):
    target_profile = os.environ.get("MODAL_PROFILE")
    if not target_profile or target_profile in DISABLED or source_profile in DISABLED:
        raise ValueError("Explicit enabled source and destination profiles required")
    plan = json.loads(PLAN_PATH.read_text())
    pool_prefix = f"pools/{plan['plan_sha256'][:16]}"
    if action == "prepare":
        if target_profile != "group-cron":
            raise ValueError("Original corpus preparation requires group-cron")
        jobs = [(s, max(c["bucket_bytes"][s["name"]] for c in plan["configs"]), pool_prefix)
                for s in plan["recipe"]["buckets"]]
        files = list(prepare_one.starmap(jobs, order_outputs=False))
        manifest = {"plan_sha256": plan["plan_sha256"], "files": sorted(files, key=lambda f: f["bucket"])}
        payload = (json.dumps(manifest, indent=2) + "\n").encode()
        import io
        with outputs.batch_upload(force=True) as batch:
            batch.put_file(io.BytesIO(payload), pool_prefix + "/corpus-manifest.json")
        print(json.dumps({"stage": "corpus_prepared", "files": len(files),
                          "compressed_bytes": sum(f["bytes"] for f in files)}), flush=True)
        return
    client, secret = source_client(source_profile)
    src = modal.Volume.from_name(OUTPUT_VOLUME, version=2, client=client)
    if action == "import-corpus":
        manifest = json.loads(b"".join(src.read_file(pool_prefix + "/corpus-manifest.json")))
        if manifest["plan_sha256"] != plan["plan_sha256"]:
            raise RuntimeError("Corpus plan mismatch")
        items = [{**f, "destination_path": f["relative_path"]} for f in manifest["files"]]
        kind = "corpus"
    elif action == "gather":
        items = gather_items(src, plan, pool_prefix)
        kind = "outputs"
    else:
        raise ValueError("action must be prepare, import-corpus, or gather")
    worker = copy_file.with_options(secrets=[secret])
    # Copy model files before publishing their completion manifests.
    phases = ([items] if action != "gather" else
              [[item for item in items if not item["source_path"].endswith("/manifest.json")],
               [item for item in items if item["source_path"].endswith("/manifest.json")]])
    results = []
    for phase in phases:
        jobs = [(item, OUTPUT_VOLUME, kind) for item in phase]
        results.extend(worker.starmap(jobs, order_outputs=False))
    if action == "import-corpus":
        import io
        marker = {"plan_sha256": plan["plan_sha256"], "source_profile": source_profile,
                  "files": manifest["files"]}
        with corpus.batch_upload(force=True) as batch:
            batch.put_file(io.BytesIO(json.dumps(marker).encode()), "tokenizer-pool-source.json")
    print(json.dumps({"stage": "transferred", "action": action, "source_profile": source_profile,
                      "target_profile": target_profile, "files": len(results)}), flush=True)
