#!/usr/bin/env python3
"""Monitor-only observer for the regret-RL continuation (updates 8,000 -> 20,000) on Tillicum.

Polls the cluster over the user's existing SSH control connection by streaming
scripts/regret_continuation_status.py to the cluster Python on stdin, writes
<local-state>/status.json (atomic) and <local-state>/events.jsonl for the
dashboard (plans/run-dashboard.json, adapter progress_json).

It never submits, cancels or changes anything on the cluster and never starts
an SSH master connection (BatchMode, existing ControlPath only).

Exit codes: 0 reached --until target, 2 failed or stopped, 3 observer lost the cluster.
Python 3.9 compatible, standard library only.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PLATFORM = "Tillicum · H200"
PROFILE = "stf"
TOTAL = 20000
START_STEP = 8000
EVENT_MAX_AGE = 600.0
SSH_TIMEOUT = 120
AMBIGUOUS_POLLS = 2
STATUS_SCRIPT = Path(__file__).resolve().parent / "regret_continuation_status.py"

FAIL_PREFIXES = ("FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL",
                 "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED", "SPECIAL_EXIT")
ACTIVE_PREFIXES = ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED",
                   "REQUEUE", "RESIZING", "SIGNALING", "STAGE_OUT", "STOPPED")
FAILURE_STAGES = ("check_failed", "failed")

OPTS = {"control_path": None}


class SSHError(RuntimeError):
    """ssh itself failed (rc 255 or timeout): connectivity, not the remote command."""


class StatusError(RuntimeError):
    """The remote status command ran but failed or printed something unusable."""


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ssh_options():
    return ["-o", "ControlPath=%s" % OPTS["control_path"],
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]


def ssh_run(host, command, stdin_text=None, timeout=SSH_TIMEOUT):
    """Run one remote command. Returns (returncode, stdout, stderr); rc 255 = ssh failure."""
    argv = ["ssh"] + ssh_options() + [host, command]
    try:
        p = subprocess.run(argv, input=stdin_text, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=timeout, universal_newlines=True,
                           stdin=None if stdin_text is not None else subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return 255, "", "ssh timeout after %ss" % timeout
    except OSError as exc:
        return 255, "", "ssh could not start: %s" % exc
    return p.returncode, p.stdout, p.stderr


def dq(text):
    """Double-quote for the remote shell but let $VARS (e.g. $USER) expand there."""
    return '"%s"' % re.sub(r'(["\\`])', r"\\\1", text)


def parse_elapsed(text):
    """[D-]HH:MM:SS, HH:MM:SS or MM:SS -> seconds (None if unparsable)."""
    try:
        days = 0
        text = text.strip()
        if "-" in text:
            d, text = text.split("-", 1)
            days = int(d)
        parts = [float(x) for x in text.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0.0)
        return int(days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2])
    except (ValueError, AttributeError):
        return None


def atomic_write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp.%d" % os.getpid())
    with open(str(tmp), "w") as handle:
        json.dump(obj, handle, indent=2, sort_keys=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(tmp), str(path))


def is_fail(state):
    return bool(state) and state.upper().startswith(FAIL_PREFIXES)


def is_active(state):
    return bool(state) and state.upper().startswith(ACTIVE_PREFIXES)


def is_running(state):
    return bool(state) and state.upper().startswith(("RUNNING", "COMPLETING"))


def is_pending(state):
    return bool(state) and state.upper().startswith(("PENDING", "CONFIGURING", "REQUEUE"))


def is_completed(state):
    return bool(state) and state.upper().startswith("COMPLETED")


def num(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def latest_step(status):
    """Newest update number the cluster files show, between the 8,000 start and TOTAL."""
    seen = [START_STEP]
    run = status.get("run_status")
    if isinstance(run, dict) and num(run.get("step")) is not None:
        seen.append(run["step"])
    train = status.get("train")
    if isinstance(train, dict) and num(train.get("last_step")) is not None:
        seen.append(train["last_step"])
    return int(max(0, min(TOTAL, max(seen))))


def job_sort_key(job):
    try:
        return int(str(job["id"]).split("_")[0])
    except (ValueError, KeyError):
        return 0


def failed_checks(receipt):
    checks = receipt.get("checks") if isinstance(receipt, dict) else None
    if isinstance(checks, dict):
        return sorted(k for k, v in checks.items() if not v)
    return []


def evaluate(status, until="complete"):
    """Map one remote status object to {stage, detail, error, completed, exit, ambiguous, ...}.

    Pure function of the status (no I/O) so it can be unit-tested. 'exit' is 0 when the
    --until target is reached, 2 for a failure or hold, otherwise None. 'ambiguous' marks a
    state that should be reported as a failure only if it persists for AMBIGUOUS_POLLS polls.
    """
    jobs = status.get("jobs") or []
    checks = [j for j in jobs if j.get("role") == "check"]
    allocs = sorted([j for j in jobs if j.get("role") == "allocation"], key=job_sort_key)
    receipt = status.get("check_receipt") if isinstance(status.get("check_receipt"), dict) else None
    driver = status.get("driver_status") if isinstance(status.get("driver_status"), dict) else {}
    dstage = str(driver.get("stage") or "")
    step = latest_step(status)
    passed = bool(receipt and receipt.get("passed") is True)
    complete = isinstance(status.get("complete"), dict)
    backups = status.get("backups") or {}
    out = {"completed": step, "check_passed": passed or complete, "exit": None, "ambiguous": False}

    def result(stage, detail=None, error=None):
        out["stage"] = stage
        if detail:
            out["detail"] = detail
        if error:
            out["error"] = error
        if stage == "complete" or (until == "check" and out["check_passed"]):
            out["exit"] = 0
        elif stage in FAILURE_STAGES or stage == "stopped":
            out["exit"] = 2
        return out

    if complete:
        out["completed"] = TOTAL
        return result("complete", "update %d reached; every checkpoint backed up and verified" % TOTAL)

    # ---- resume check
    if not passed:
        bad = failed_checks(receipt)
        if receipt is not None or dstage == "check_failed":
            return result("check_failed", error="resume check did not pass%s"
                          % ((": failed checks: " + ", ".join(bad)) if bad else ""))
        failed_jobs = [j for j in checks if is_fail(j.get("state"))]
        if failed_jobs:
            j = failed_jobs[-1]
            return result("check_failed", error="check job %s %s (exit %s) without a passing receipt"
                          % (j["id"], j.get("state"), j.get("exit_code")))
        if not checks:
            return result("waiting_for_check", "no resume check has been submitted yet")
        if any(is_active(j.get("state")) for j in checks) or dstage == "checking":
            return result("checking", "resume check %s" % ", ".join(
                "%s %s" % (j["id"], j.get("state")) for j in checks))
        if all(is_completed(j.get("state")) for j in checks):
            out["ambiguous"] = True
            return result("checking", "check job ended without a receipt; confirming",
                          error="check job COMPLETED but check-receipt.json is missing")
        return result("checking", "waiting for check job state")

    # ---- driver and Slurm failures after the check passed
    if dstage == "failed":
        detail = ", ".join("%s=%s" % (k, driver[k]) for k in ("returncode", "reason", "step") if k in driver)
        return result("failed", error="trainer failed (driver-status)%s" % ((": " + detail) if detail else ""))
    if dstage == "needs_approval":
        return result("failed", error="driver needs approval: %s" % driver.get("reason", "allocations used"))
    if dstage == "paused":
        return result("stopped", error="paused by pause-request.json: %s" % driver.get("reason", ""))
    handed = dstage == "handed_off"
    for idx, job in enumerate(allocs):
        state = job.get("state")
        if not is_fail(state):
            continue
        later = [j for j in allocs[idx + 1:] if not is_fail(j.get("state"))]
        if state.upper().startswith("TIMEOUT") and (handed or later):
            continue  # walltime stop with a successor queued: a handoff, not a failure
        return result("failed", error="Slurm job %s %s (exit %s, elapsed %s)"
                      % (job["id"], state, job.get("exit_code"), job.get("elapsed")))

    # ---- after the check
    active = [j for j in allocs if is_active(j.get("state"))]
    running = [j for j in allocs if is_running(j.get("state"))]
    pending = [j for j in allocs if is_pending(j.get("state"))]
    ended = [j for j in allocs if is_completed(j.get("state")) or is_fail(j.get("state"))]
    pending_backups = backups.get("pending") or []
    if step >= TOTAL:
        return result("uploading", "update %d reached; %d checkpoint backup(s) pending"
                      % (TOTAL, len(pending_backups)))
    if running:
        return result("training", "update %d of %d" % (step, TOTAL))
    if pending and (ended or handed):
        return result("handed_off", "allocation ended at update %d; next allocation %s is queued"
                      % (step, pending[0]["id"]))
    if pending:
        return result("queued", "training allocation %s is waiting in the queue" % pending[0]["id"])
    if not allocs and not active:
        if handed:
            return result("handed_off", "driver reports a handoff; next allocation not visible yet")
        return result("check_passed", "resume check passed; training has not been submitted yet")
    if handed:
        return result("handed_off", "driver reports a handoff; waiting for the next allocation")
    out["ambiguous"] = True
    return result("training", "no running allocation at update %d" % step,
                  error="allocations ended at update %d (< %d) without a handoff or completion" % (step, TOTAL))


class Observer:
    def __init__(self, args):
        self.a = args
        self.local = Path(args.local_state)
        self.local.mkdir(parents=True, exist_ok=True)
        self.status_path = self.local / "status.json"
        self.events_path = self.local / "events.jsonl"
        self.ssh_failures = 0
        self.status_errors = 0
        self.ambiguous_polls = 0
        self.last_status = {}
        self.last_event_sig = None
        self.last_event_time = 0.0
        self.started_at = utc_now()
        try:
            prev = json.loads(self.status_path.read_text())
            if isinstance(prev.get("observer_started_at"), str):
                self.started_at = prev["observer_started_at"]
        except (OSError, ValueError):
            pass

    # ---------------------------------------------------------------- cluster
    def fetch_status(self):
        """Run the status script on the cluster (streamed on stdin). None while STATE is absent."""
        py, state = self.a.python, self.a.remote_state
        cmd = ('if [ -d %s ]; then %s -I - --state %s; else echo @@MISSING; fi'
               % (dq(state), dq(py), dq(state)))
        rc, out, err = ssh_run(self.a.host, cmd, stdin_text=Path(self.a.status_script).read_text())
        if rc == 255:
            raise SSHError("ssh failure: %s" % (err or out).strip()[:300])
        lines = [ln for ln in out.splitlines() if ln.strip()]
        if lines and lines[-1].strip() == "@@MISSING":
            return None
        if rc != 0:
            raise StatusError("status command rc=%s: %s" % (rc, (err or out).strip()[:300]))
        if not lines:
            raise StatusError("status command printed nothing")
        try:
            data = json.loads(lines[-1])
        except ValueError as exc:
            raise StatusError("status command output is not JSON: %s" % exc)
        if not isinstance(data, dict):
            raise StatusError("status command output is not an object")
        return data

    # ------------------------------------------------------------- composition
    def build_workers(self, status, step):
        train = status.get("train") or {}
        workers, eta = [], None
        for job in status.get("jobs") or []:
            slurm = job.get("state") or "UNKNOWN"
            reason = job.get("reason") or ""
            entry = {"id": str(job["id"]), "profile": PROFILE, "platform": PLATFORM}
            if job.get("role") == "check":
                entry["task"] = "resume check (update 8,001-8,002, debug queue)"
            else:
                entry["task"] = "training %d to %d" % (START_STEP, TOTAL)
            if is_running(slurm) and job.get("role") == "allocation":
                text = "Update %d/%d" % (step, TOTAL)
                if num(train.get("seconds")) is not None:
                    text += " · %.1f s/update" % train["seconds"]
                if num(train.get("peak_gib")) is not None:
                    text += " · peak %.1f GiB" % train["peak_gib"]
                entry["state"] = text
                if num(train.get("seconds")) is not None:
                    eta = int(max(0, TOTAL - step) * train["seconds"])
                    entry["eta_seconds"] = eta
            else:
                entry["state"] = "%s%s" % (slurm, " (%s)" % reason if reason and is_pending(slurm) else "")
            entry["elapsed_seconds"] = parse_elapsed(job.get("elapsed") or "")
            entry["slurm_state"] = slurm
            if job.get("exit_code"):
                entry["exit_code"] = job["exit_code"]
            workers.append(entry)
        return workers, eta

    # ------------------------------------------------------------------ output
    def emit(self, record, now):
        record["at_utc"] = utc_now()
        record.setdefault("total", TOTAL)
        record.setdefault("unit", "updates")
        record.setdefault("platform", PLATFORM)
        record.setdefault("profile", PROFILE)
        record["observer_started_at"] = self.started_at
        try:
            record["elapsed_seconds"] = max(0, int((datetime.now(timezone.utc) -
                                                    datetime.fromisoformat(self.started_at)).total_seconds()))
        except ValueError:
            pass
        self.last_status = record
        atomic_write_json(self.status_path, record)
        sig = (record.get("stage"),
               tuple((w["id"], w.get("slurm_state")) for w in record.get("workers", [])),
               record.get("state"))
        if sig != self.last_event_sig or now - self.last_event_time >= EVENT_MAX_AGE:
            ev = {"at_utc": record["at_utc"], "stage": record.get("stage"),
                  "completed": record.get("completed"), "total": record["total"]}
            for key in ("state", "error", "eta_seconds", "detail"):
                if record.get(key) is not None:
                    ev[key] = record[key]
            ev["worker_states"] = {w["id"]: w.get("slurm_state") for w in record.get("workers", [])}
            with open(str(self.events_path), "a") as handle:
                handle.write(json.dumps(ev) + "\n")
                handle.flush()
            self.last_event_sig = sig
            self.last_event_time = now
        line = {"at_utc": record["at_utc"], "stage": record.get("stage"),
                "completed": record.get("completed"), "eta_seconds": record.get("eta_seconds"),
                "ssh_failures": self.ssh_failures}
        if record.get("error"):
            line["error"] = record["error"]
        print(json.dumps(line), flush=True)

    def stop(self, why, now):
        """Record that the observer itself stopped (the Slurm jobs are not affected)."""
        rec = dict(self.last_status)
        rec.update(stage="stopped", state="stopped", error=why)
        rec.pop("eta_seconds", None)
        rec.setdefault("completed", START_STEP)
        self.emit(rec, now)

    # -------------------------------------------------------------------- poll
    def poll(self, now):
        """One poll. Returns an exit code, or None to continue."""
        try:
            status = self.fetch_status()
        except (SSHError, StatusError) as exc:
            if isinstance(exc, SSHError):
                self.ssh_failures += 1
            else:
                self.status_errors += 1
            rec = dict(self.last_status)
            rec["ssh_failures"] = self.ssh_failures
            rec["status_errors"] = self.status_errors
            rec["last_error"] = str(exc)
            if self.ssh_failures >= self.a.max_ssh_failures or self.status_errors >= self.a.max_ssh_failures:
                if isinstance(exc, SSHError):
                    why = "%d consecutive SSH failures" % self.ssh_failures
                else:
                    why = "the remote status command failed %d times in a row" % self.status_errors
                self.last_status = rec
                self.stop("observer lost the cluster (%s); the Slurm jobs are not affected: %s" % (why, exc), now)
                return 3
            rec.setdefault("stage", "waiting_for_check")
            rec.setdefault("completed", START_STEP)
            self.emit(rec, now)
            return None
        self.ssh_failures = 0
        self.status_errors = 0
        if status is None:
            self.emit({"completed": START_STEP, "stage": "waiting_for_check", "workers": [],
                       "remote_logs": [], "ssh_failures": 0,
                       "detail": "remote state directory does not exist yet"}, now)
            return None
        verdict = evaluate(status, self.a.until)
        step = verdict["completed"]
        workers, eta = self.build_workers(status, step)
        rec = {"completed": step, "stage": verdict["stage"], "workers": workers,
               "remote_logs": ["%s/run/events.jsonl" % self.a.remote_state.rstrip("/")],
               "ssh_failures": 0, "slurm": status.get("slurm"),
               "backups": status.get("backups"), "train": status.get("train"),
               "driver_stage": (status.get("driver_status") or {}).get("stage"),
               "check_passed": verdict["check_passed"],
               "worker_jobs": {w["id"]: {"state": w["slurm_state"], "exit_code": w.get("exit_code"),
                                         "elapsed_seconds": w.get("elapsed_seconds")} for w in workers}}
        for key in ("detail", "error"):
            if verdict.get(key):
                rec[key] = verdict[key]
        if eta is not None and verdict["stage"] == "training":
            rec["eta_seconds"] = eta
        if verdict["ambiguous"] and verdict["exit"] is None:  # a reached --until target wins
            self.ambiguous_polls += 1
            if self.ambiguous_polls < AMBIGUOUS_POLLS:
                rec.pop("error", None)
                self.emit(rec, now)
                return None
            rec["stage"] = "check_failed" if verdict["stage"] == "checking" else "failed"
            verdict["exit"] = 2
        else:
            self.ambiguous_polls = 0
        if rec["stage"] in FAILURE_STAGES or rec["stage"] == "stopped":
            rec["state"] = "failed" if rec["stage"] in FAILURE_STAGES else "stopped"
            rec.pop("eta_seconds", None)
        if rec["stage"] == "complete":
            rec.pop("eta_seconds", None)
        self.emit(rec, now)
        return verdict["exit"]


def parse_args(argv=None):
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--host", required=True)
    p.add_argument("--control-path", default="~/.ssh/cm-tillicum-*",
                   help="SSH ControlPath socket; the default glob must match exactly one socket")
    p.add_argument("--remote-state", default="/gpfs/scrubbed/$USER/anytok-runs/regret-rl-20k",
                   help="remote STATE directory ($USER expands on the cluster)")
    p.add_argument("--python", required=True, help="remote python executable")
    p.add_argument("--status-script", default=str(STATUS_SCRIPT))
    p.add_argument("--local-state", default=str(root / "modal_results" / "regret_rl_20k_continuation"))
    p.add_argument("--until", default="complete", choices=["check", "complete"])
    p.add_argument("--interval", type=float, default=60)
    p.add_argument("--max-ssh-failures", type=int, default=30)
    args = p.parse_args(argv)
    args.control_path = os.path.expanduser(args.control_path)
    if any(ch in args.control_path for ch in "*?["):
        import glob
        import stat

        def is_sock(path):
            try:
                return stat.S_ISSOCK(os.stat(path).st_mode)
            except OSError:
                return False
        found = [f for f in glob.glob(args.control_path) if is_sock(f)]
        if len(found) != 1:
            p.error("--control-path glob %s matched %d sockets; pass the exact path" % (args.control_path, len(found)))
        args.control_path = found[0]
    return args


def main(argv=None):
    args = parse_args(argv)
    OPTS["control_path"] = args.control_path
    obs = Observer(args)
    try:
        while True:
            code = obs.poll(time.time())
            if code is not None:
                return code
            time.sleep(args.interval)
    except KeyboardInterrupt:
        obs.stop("observer interrupted; the Slurm jobs are not affected", time.time())
        return 130


if __name__ == "__main__":
    sys.exit(main())
