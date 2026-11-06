"""Generate the saved 1,000-tokenizer pool on group-cron's CPUs.

    MODAL_PROFILE=group-cron modal run --detach scripts/modal_generate_tokenizer_pool.py --stage pilot
    MODAL_PROFILE=group-cron modal run --detach scripts/modal_generate_tokenizer_pool.py --stage full

No CPU, memory or GPU reservations are specified. Each task uses a fresh
container and four Rayon threads. All outputs live on a persistent Volume.
The full stage checks pilot estimates, then submits bounded waves and rechecks
billing. Budget estimates include a margin but cannot guarantee a billing cap.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import modal

import tokenizer_pool
from tokenizer_pool import digest

PROJECT = Path(__file__).resolve().parents[1]
PLAN_PATH = PROJECT / "plans/tokenizer-pool-1000-v1.json"
BASE_WHEEL_PATH = PROJECT / "build/all-candidate/tokenizers-0.22.2-cp39-abi3-manylinux_2_34_x86_64.whl"
WHEEL_PATH = PROJECT / "build/pool-v2" / BASE_WHEEL_PATH.name
VOLUME_NAME = "anytok-tokenizer-pool"
LOG_DIR = PROJECT / "modal_results/tokenizer_pool"
DISABLED = {"general", "research-everettcc"}
MAX_CONTAINERS = 100
# The account's monthly allowance is $30. Leave $1 for billing lag/storage.
CREDIT_CEILING = 29.0
COST_MULTIPLIER = 1.5

app = modal.App("anytok-tokenizer-pool")
corpus = modal.Volume.from_name("anytok-corpus")
outputs = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)
image = (modal.Image.debian_slim(python_version="3.11")
    .add_local_file(str(WHEEL_PATH), f"/tmp/{WHEEL_PATH.name}", copy=True)
    .run_commands(f"pip install --no-deps /tmp/{WHEEL_PATH.name}")
    .env({"RAYON_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "true"})
    .add_local_python_source("tokenizer_pool"))


def load_plan():
    plan = json.loads(PLAN_PATH.read_text())
    payload = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if digest(payload) != plan["plan_sha256"]:
        raise ValueError("Saved plan hash mismatch")
    if hashlib.sha256(BASE_WHEEL_PATH.read_bytes()).hexdigest() != plan["tokenizers_wheel_sha256"]:
        raise ValueError("Patched tokenizer wheel differs from saved plan")
    return plan


def prefix(plan):
    return f"pools/{plan['plan_sha256'][:16]}"


@app.function(image=image, volumes={"/corpus": corpus, "/pool": outputs},
              max_containers=MAX_CONTAINERS, single_use_containers=True,
              timeout=6 * 3600)
def train_one(config, recipe, pool_prefix, provenance):
    destination = Path("/pool") / pool_prefix / "tokenizers" / f"{config['index']:04d}"
    saved = destination / "manifest.json"
    if saved.exists():
        manifest = json.loads(saved.read_text())
        path = destination / "tokenizer.json"
        if manifest["config"] != config or manifest["provenance"] != provenance:
            raise RuntimeError("Existing tokenizer has different configuration or provenance")
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["tokenizer_sha256"]:
            raise RuntimeError("Existing tokenizer failed checksum validation")
        return manifest
    try:
        # A Rust panic or hard crash must fail this input rather than crash
        # the Modal runner and trigger repeated infrastructure retries.
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            (temp / "config.json").write_text(json.dumps(config))
            (temp / "recipe.json").write_text(json.dumps(recipe))
            subprocess.run([sys.executable, tokenizer_pool.__file__,
                "--train-config", str(temp / "config.json"),
                "--recipe", str(temp / "recipe.json"), "--destination", str(destination)], check=True)
        manifest = json.loads(saved.read_text())
        manifest["provenance"] = provenance
        saved.write_text(json.dumps(manifest, indent=2) + "\n")
        (destination / "failure.json").unlink(missing_ok=True)
        outputs.commit()
        return manifest
    except BaseException as exc:
        destination.mkdir(parents=True, exist_ok=True)
        saved.unlink(missing_ok=True)
        (destination / "failure.json").write_text(json.dumps({
            "config": config, "error": str(exc), "at_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2) + "\n")
        outputs.commit()
        raise RuntimeError(f"{type(exc).__name__}: {exc}") from None


def cli_json(*args):
    result = subprocess.run(["modal", *args], capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    return json.loads(result.stdout)


def used_credits():
    summary = cli_json("billing", "summary", "--json")
    return -float(summary["adjustments"]["credits"])


def emit(stage, **fields):
    item = {"at_utc": datetime.now(timezone.utc).isoformat(), "stage": stage, **fields}
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "status.json").write_text(json.dumps(item, indent=2) + "\n")
    with (LOG_DIR / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(item) + "\n")
    print(json.dumps(item), flush=True)


def completed(plan):
    found = {}
    base = prefix(plan) + "/tokenizers"
    entries = list(outputs.iterdir(base, recursive=True))
    for entry in entries:
        if entry.path.endswith("/manifest.json"):
            data = json.loads(b"".join(outputs.read_file(entry.path)))
            index = data["config"]["index"]
            if data["config"] != plan["configs"][index] or data["status"] != "complete":
                raise RuntimeError(f"Saved manifest mismatch: {entry.path}")
            found[index] = data
    return found


def estimates(plan, done, indices=None, references=None, allow_unknown=False):
    groups = defaultdict(list)
    for manifest in {**(references or {}), **done}.values():
        c = manifest["config"]
        groups[c["algorithm"], c["pretokenizer"]].append(manifest["estimated_process_cost_usd"])
    costs = {}
    selected = set(indices) if indices is not None else set(range(plan["count"]))
    for c in plan["configs"]:
        if c["index"] in selected and c["index"] not in done:
            group = groups[c["algorithm"], c["pretokenizer"]]
            if not group and allow_unknown:
                # Only use for small targeted diagnostics until that family
                # has a successful measured pilot. This is not an actual cap.
                costs[c["index"]] = 5.0
                continue
            if not group:
                raise RuntimeError(f"Missing successful pilot for {c['algorithm']}/{c['pretokenizer']}")
            # Use the most expensive observed member of each category; allow
            # 50% plus $0.001/job for startup/container overhead and variability.
            costs[c["index"]] = max(group) * COST_MULTIPLIER + 0.001
    return costs


def run_stage(stage: str, wave_size: int, indices_text: str = ""):
    global LOG_DIR
    profile = os.environ.get("MODAL_PROFILE")
    if not profile or profile in DISABLED:
        raise RuntimeError("An explicit enabled Modal profile is required")
    LOG_DIR = PROJECT / "modal_results/tokenizer_pool"
    if profile != "group-cron":
        LOG_DIR = LOG_DIR / profile
    if stage not in {"pilot", "full", "selected"} or not 1 <= wave_size <= MAX_CONTAINERS:
        raise ValueError("stage must be pilot, full, or selected; wave_size must be 1..100")
    plan = load_plan()
    pool_prefix = prefix(plan)
    if profile != "group-cron":
        marker = json.loads(b"".join(corpus.read_file("tokenizer-pool-source.json")))
        if marker["plan_sha256"] != plan["plan_sha256"]:
            raise RuntimeError("Install the matching compact corpus before training")
    references = {}
    metadata_root = PROJECT / "modal_results/tokenizer_pool"
    reference_paths = list(metadata_root.glob("[0-9][0-9][0-9][0-9].json"))
    reference_paths.extend(path for folder in metadata_root.iterdir()
                           if folder.is_dir() and folder.name not in DISABLED
                           for path in folder.glob("[0-9][0-9][0-9][0-9].json"))
    for path in reference_paths:
        try:
            m = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if m["status"] == "complete" and m["config"] == plan["configs"][m["config"]["index"]]:
            references[m["config"]["index"]] = m
    provenance = {"plan_sha256": plan["plan_sha256"], "wheel_sha256": plan["tokenizers_wheel_sha256"],
                  "worker_sha256": hashlib.sha256((PROJECT / "scripts/tokenizer_pool.py").read_bytes()).hexdigest(),
                  "rayon_threads": 4,
                  "runtime_wheel_sha256": hashlib.sha256(WHEEL_PATH.read_bytes()).hexdigest(),
                  "unigram_patches": ["tokenizers-0.22.2-unigram-literal-unk.patch",
                                      "tokenizers-0.22.2-unigram-stack-drop.patch"]}
    # Upload configurations before any expensive work. Include generation code
    # so the persistent volume retains the recipe even if this checkout is lost.
    with outputs.batch_upload(force=True) as batch:
        batch.put_file(PLAN_PATH, pool_prefix + "/plan.json")
        batch.put_file(BASE_WHEEL_PATH, pool_prefix + "/artifacts/base/" + BASE_WHEEL_PATH.name)
        batch.put_file(WHEEL_PATH, pool_prefix + "/artifacts/runtime/" + provenance["runtime_wheel_sha256"] + "/" + WHEEL_PATH.name)
        batch.put_file(PROJECT / "patches/tokenizers-0.22.2-unigram-literal-unk.patch",
                       pool_prefix + "/artifacts/tokenizers-0.22.2-unigram-literal-unk.patch")
        batch.put_file(PROJECT / "patches/tokenizers-0.22.2-unigram-stack-drop.patch",
                       pool_prefix + "/artifacts/tokenizers-0.22.2-unigram-stack-drop.patch")
        for name in ("tokenizer_pool.py", "modal_generate_tokenizer_pool.py"):
            batch.put_file(PROJECT / "scripts" / name, pool_prefix + "/source/" + name)
    try:
        done = completed(plan)
    except (FileNotFoundError, modal.exception.NotFoundError):
        done = {}
    if stage == "selected":
        requested = list(dict.fromkeys(int(i) for i in indices_text.split(",") if i.strip()))
        if not requested or any(not 0 <= i < plan["count"] for i in requested):
            raise ValueError("selected stage requires --indices containing saved indices 0..999")
        indices = [i for i in requested if i not in done]
        costs = estimates(plan, done, indices, references, allow_unknown=True)
        spent = used_credits()
        emit("forecast", completed=len(done), scheduled=len(indices),
             estimated_remaining_cost_usd=sum(costs.values()), credits_used_usd=spent)
        if spent + sum(costs.values()) > CREDIT_CEILING:
            emit("budget_stop", completed=len(done), reason="Selected shard exceeds credit estimate")
            return
    elif stage == "pilot":
        indices = [i for i in plan["pilot_indices"] if i not in done]
        if used_credits() > CREDIT_CEILING - 2:
            raise RuntimeError("Insufficient credit margin for pilot")
    else:
        missing_pilots = set(plan["pilot_indices"]) - done.keys()
        if missing_pilots:
            raise RuntimeError(f"Pilot is not complete: {sorted(missing_pilots)}")
        costs = estimates(plan, done)
        spent = used_credits()
        projected = sum(costs.values())
        emit("forecast", completed=len(done), estimated_remaining_cost_usd=projected,
             credits_used_usd=spent, credit_ceiling_usd=CREDIT_CEILING)
        if spent + projected > CREDIT_CEILING:
            emit("budget_stop", completed=len(done), reason="Full pool exceeds remaining credit estimate")
            return
        indices = [c["index"] for c in plan["configs"] if c["index"] not in done]
    emit("starting", profile=profile, stage_requested=stage, scheduled=len(indices),
         completed=len(done), volume=VOLUME_NAME, path=pool_prefix)
    start = time.monotonic()
    for offset in range(0, len(indices), wave_size):
        wave = indices[offset:offset + wave_size]
        if stage in {"full", "selected"}:
            costs = estimates(plan, done, indices, references, allow_unknown=stage == "selected")
            if used_credits() + sum(costs.values()) > CREDIT_CEILING:
                emit("budget_stop", completed=len(done), reason="Updated full-pool forecast exceeds credit margin")
                return
        failures = []
        jobs = [(plan["configs"][i], plan["recipe"], pool_prefix, provenance) for i in wave]
        for result in train_one.starmap(jobs, order_outputs=False, return_exceptions=True):
            if isinstance(result, BaseException):
                failures.append(repr(result))
                emit("job_failed", error=repr(result), completed=len(done))
            else:
                index = result["config"]["index"]
                done[index] = result
                (LOG_DIR / f"{index:04d}.json").write_text(json.dumps(result, indent=2) + "\n")
                emit("job_complete", index=index, completed=len(done), target=plan["count"],
                     train_seconds=result["train_seconds"], peak_rss_gib=result["peak_rss_gib"],
                     estimated_process_cost_usd=result["estimated_process_cost_usd"])
        if failures:
            emit("failed", completed=len(done), errors=failures)
            raise RuntimeError("Generation encountered failures; saved configurations are preserved")
    summary_path = LOG_DIR / f"{stage}-results.json"
    summary_path.write_text(json.dumps({"plan_sha256": plan["plan_sha256"],
        "completed": len(done), "target": plan["count"],
        "tokenizers": [done[i] for i in sorted(done)]}, indent=2) + "\n")
    with outputs.batch_upload(force=True) as batch:
        batch.put_file(summary_path, pool_prefix + f"/{stage}-results.json")
    emit("pilot_complete" if stage == "pilot" else "complete", completed=len(done),
         elapsed_seconds=time.monotonic() - start, credits_used_usd=used_credits(),
         total_tokenizer_bytes=sum(m["tokenizer_bytes"] for m in done.values()),
         estimated_process_cost_usd=sum(m["estimated_process_cost_usd"] for m in done.values()))


@app.local_entrypoint()
def main(stage: str = "pilot", wave_size: int = 100, indices: str = ""):
    if stage == "all":
        run_stage("pilot", wave_size)
        run_stage("full", wave_size)
    else:
        run_stage(stage, wave_size, indices)
