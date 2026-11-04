"""Finish the split ZeTT 20k Modal run after the active projects segment.

Run inside tmux through ``run_modal_zett_20k_controller.sh APP_ID``. This
controller waits for APP_ID, uses the remaining credit in projects, hands off
the latest full checkpoint to grading and then test-joining, and collects the
500-step grid in test-joining. It fails closed on a missing checkpoint, failed
run, failed hash audit, or insufficient remaining compute budget. Its compact
status is ``modal_results/zett_20k_controller/status.json``.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from modal_zett_20k_artifacts import (
    INPUT_MANIFEST, PROFILES, ROOT, VOLUME, checkpoint_steps, collect, handoff,
)

STATUS_DIR = ROOT / "modal_results/zett_20k_controller"
STATUS_FILE = STATUS_DIR / "status.json"
LAUNCHER = "scripts/modal_zett_20k.py"
FINAL_STEP = 20_000
SPEND_CEILING = 29.0  # Keep a margin below each workspace's $30 credit.
ESTIMATED_HOURLY_COST = 7.3  # B300 is $7.10/h; includes a CPU/memory margin.


def emit(stage, **fields):
    item = {"at_utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage, **fields}
    print(json.dumps(item, sort_keys=True), flush=True)
    pending = STATUS_FILE.with_suffix(".tmp")
    pending.write_text(json.dumps(item, indent=2, sort_keys=True) + "\n")
    os.replace(pending, STATUS_FILE)


def cli(profile, *args, capture=False):
    env = {**os.environ, "MODAL_PROFILE": profile}
    result = subprocess.run(["modal", *args], env=env, text=True,
                            capture_output=capture, check=False)
    if result.returncode:
        raise RuntimeError(f"{profile}: modal {' '.join(args)} exited "
                           f"{result.returncode}: {result.stdout if capture else ''} "
                           f"{result.stderr if capture else ''}")
    return result.stdout if capture else ""


def latest(profile):
    steps = checkpoint_steps(profile)
    if not steps:
        raise ValueError(f"No checkpoint in {profile}")
    return steps[-1]


def billable_compute(profile):
    summary = json.loads(cli(profile, "billing", "summary", "--json", capture=True))
    return float(summary["metered_cost_breakdown"]["ephemeral_apps"])


def audit_checkpoint(profile):
    step = latest(profile)
    emit("auditing_checkpoint", profile=profile, step=step)
    cli(profile, "run", LAUNCHER, "--action", "audit-checkpoint", "--step", str(step))
    emit("audited_checkpoint", profile=profile, step=step)
    return step


def compact(profile, source_finished=False):
    action = "compact-source" if source_finished else "compact"
    emit("compacting_checkpoints", profile=profile, source_finished=source_finished)
    cli(profile, "run", LAUNCHER, "--action", action)
    emit("compacted_checkpoints", profile=profile, source_finished=source_finished)


def wait_for_app(profile, app_id):
    emit("waiting_for_app", profile=profile, app_id=app_id)
    last_report = 0
    deadline = time.monotonic() + 4 * 3600 + 1800
    while time.monotonic() < deadline:
        apps = json.loads(cli(profile, "app", "list", "--json", capture=True))
        match = next((app for app in apps if app["app_id"] == app_id), None)
        if match is None:
            raise RuntimeError(f"App {app_id} disappeared from {profile}")
        if match["state"] == "stopped" or match["tasks"] == "0":
            for _ in range(6):
                logs = cli(profile, "app", "logs", app_id, "--tail", "100", capture=True)
                if ('"stage": "paused_at"' in logs
                        or '"stage": "complete"' in logs):
                    break
                time.sleep(10)
            else:
                raise RuntimeError(f"App {app_id} stopped without a clean trainer exit:\n"
                                   + logs[-4000:])
            emit("app_complete", profile=profile, app_id=app_id, step=latest(profile))
            return
        if time.monotonic() - last_report > 300:
            emit("app_running", profile=profile, app_id=app_id,
                 latest_checkpoint=latest(profile))
            last_report = time.monotonic()
        time.sleep(60)
    raise TimeoutError(f"App {app_id} remained active beyond the 4-hour deadline")


def run_segment(profile, seconds):
    if latest(profile) >= FINAL_STEP:
        return latest(profile)
    spent = billable_compute(profile)
    projected = spent + seconds / 3600 * ESTIMATED_HOURLY_COST + 0.3
    if projected > SPEND_CEILING:
        raise RuntimeError(f"{profile} compute budget guard: ${spent:.2f} used; "
                           f"{seconds}s projects ${projected:.2f} above "
                           f"${SPEND_CEILING:.2f}")
    before = latest(profile)
    emit("segment_start", profile=profile, from_step=before,
         max_seconds=seconds, metered_compute_usd=round(spent, 2))
    cli(profile, "run", "--detach", LAUNCHER, "--action", "train",
        "--max-seconds", str(seconds))
    after = audit_checkpoint(profile)
    if after <= before:
        raise RuntimeError(f"{profile} did not advance beyond {before}")
    compact(profile)
    emit("segment_complete", profile=profile, from_step=before,
         to_step=after, metered_compute_usd=round(billable_compute(profile), 2))
    return after


def handoff_and_audit(source, target):
    emit("handoff_start", source=source, target=target, step=latest(source))
    handoff(source, target)
    if audit_checkpoint(target) != latest(source):
        raise RuntimeError("Handoff step mismatch")
    emit("handoff_complete", source=source, target=target, step=latest(target))
    compact(source, source_finished=True)


def run(wait_app=None, resume_grading=False):
    if not INPUT_MANIFEST.is_file():
        raise FileNotFoundError(INPUT_MANIFEST)
    emit("controller_started", wait_app=wait_app, resume_grading=resume_grading,
         volume=VOLUME)
    if resume_grading:
        audit_checkpoint("grading")
        if any(step % 500 for step in checkpoint_steps("projects")):
            compact("projects", source_finished=True)
    else:
        if wait_app:
            wait_for_app("projects", wait_app)
        else:
            apps = json.loads(cli("projects", "app", "list", "--json", capture=True))
            active = [app["app_id"] for app in apps
                      if app["description"] == "anytok-zett-20k-b300"
                      and app["state"] != "stopped" and app["tasks"] != "0"]
            if active:
                raise RuntimeError(f"Projects training app still active: {active}")
        audit_checkpoint("projects")
        compact("projects")
        if latest("projects") < FINAL_STEP:
            run_segment("projects", 1400 if wait_app else 9900)
        handoff_and_audit("projects", "grading")
    if latest("grading") < FINAL_STEP:
        run_segment("grading", 10800)
    if latest("grading") < FINAL_STEP:
        run_segment("grading", 3200)
    handoff_and_audit("grading", "test-joining")
    if latest("test-joining") < FINAL_STEP:
        run_segment("test-joining", 10800)
    if latest("test-joining") < FINAL_STEP:
        run_segment("test-joining", 3200)
    if latest("test-joining") < FINAL_STEP:
        raise RuntimeError("Credit-safe segments ended before step 20000")
    emit("collection_start", final_step=latest("test-joining"))
    collect()
    emit("final_audit_start")
    cli("test-joining", "run", LAUNCHER, "--action", "audit-final")
    emit("complete", final_step=FINAL_STEP, checkpoint_count=40,
         final_profile="test-joining", volume=VOLUME,
         input_manifest_sha256=hashlib.sha256(
             INPUT_MANIFEST.read_bytes()).hexdigest(),
         metered_compute_usd={profile: round(billable_compute(profile), 2)
                              for profile in PROFILES})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-app",
                        help="Existing detached projects training App ID")
    parser.add_argument("--recover-cancelled", action="store_true",
                        help="Resume the verified latest checkpoint after a cancelled app")
    parser.add_argument("--resume-grading", action="store_true",
                        help="Continue after a separately audited projects-to-grading handoff")
    parser.add_argument("--plan", action="store_true",
                        help="Print current steps, budget, and stages without launching work")
    args = parser.parse_args()
    if sum((bool(args.wait_app), args.recover_cancelled, args.resume_grading)) != 1:
        parser.error("Choose exactly one start mode")
    if args.plan:
        print(json.dumps({"wait_app": args.wait_app, "volume": VOLUME,
                          "current_steps": {profile: checkpoint_steps(profile)
                                            for profile in PROFILES},
                          "metered_compute_usd": {
                              profile: round(billable_compute(profile), 2)
                              for profile in PROFILES},
                          "next_stages": ([] if args.resume_grading else
                                          (["wait for active projects segment",
                                            "projects top-up 1400s"] if args.wait_app else
                                           ["verify checkpoint after cancelled app",
                                            "projects recovery segment 9900s"]) +
                                          ["handoff to grading"]) + [
                                          "grading 10800s + 3200s",
                                          "handoff to test-joining",
                                          "test-joining 10800s + up to 3200s",
                                          "collect and hash-audit 10 full plus 30 weights checkpoints"],
                          "compute_ceiling_usd_per_profile": SPEND_CEILING},
                         indent=2))
        return
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    with (STATUS_DIR / "controller.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args.wait_app, args.resume_grading)
        except Exception as exc:
            emit("failed", error=str(exc), traceback=traceback.format_exc()[-4000:])
            raise


if __name__ == "__main__":
    main()
