"""Read-only status collector for the regret-RL 8k -> 20k continuation on Tillicum.

Runs on the cluster under the cluster Python (stdlib only, nothing is written).
Prints ONE JSON object: the small state files, the tail of run/events.jsonl, the
backup receipts and the Slurm state of every allocation, current and check job.
The user name is replaced by <user> in every string that is printed.

Meant to be streamed over ssh stdin by scripts/run_regret_continuation.py:
    python -I - --state /gpfs/scrubbed/$USER/anytok-runs/regret-rl-20k
"""
import argparse
import getpass
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

SOURCE_STEP = 8000
TAIL_BLOCK = 1 << 20  # bytes read per backwards step through events.jsonl
JOB_ID = re.compile(r"^\d+(_\d+)?$")
# Small state files: looked up in STATE first, then in these sub-folders.
SEARCH = {
    "driver-status.json": ("", "run"),
    "backup-status.json": ("", "run", "backups"),
    "check-receipt.json": ("", "check-run", "run"),
    "allocations.json": ("", "run"),
    "current-job.json": ("", "run"),
    "complete.json": ("", "run"),
}


def read_json(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def find_json(state, name):
    for sub in SEARCH[name]:
        value = read_json(os.path.join(state, sub, name) if sub else os.path.join(state, name))
        if value is not None:
            return value
    return None


def reverse_lines(path):
    """Yield the lines of a file from the end, reading it block by block."""
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        pos = handle.tell()
        carry = b""
        while pos > 0:
            size = min(TAIL_BLOCK, pos)
            pos -= size
            handle.seek(pos)
            parts = (handle.read(size) + carry).split(b"\n")
            carry = parts[0]
            for line in reversed(parts[1:]):
                if line.strip():
                    yield line
        if carry.strip():
            yield carry


def train_tail(state):
    """Last train row plus the number of train rows with step > SOURCE_STEP.

    Walks events.jsonl backwards and stops at the first train row with
    step <= SOURCE_STEP, so only the new part of the log is read.
    """
    path = os.path.join(state, "run", "events.jsonl")
    out = {"last_step": None, "seconds": None, "peak_gib": None, "rows_after_%d" % SOURCE_STEP: 0}
    if not os.path.exists(path):
        return out
    count = 0
    try:
        for raw in reverse_lines(path):
            try:
                row = json.loads(raw.decode("utf-8", "replace"))
            except ValueError:
                continue  # a torn trailing line while the trainer is writing
            if not isinstance(row, dict) or row.get("stage") != "train":
                continue
            step = row.get("step")
            if not isinstance(step, (int, float)):
                continue
            if out["last_step"] is None:
                out.update(last_step=step, seconds=row.get("seconds"), peak_gib=row.get("peak_gib"))
            if step <= SOURCE_STEP:
                break
            count += 1
    except OSError as exc:
        out["error"] = str(exc)
    out["rows_after_%d" % SOURCE_STEP] = count
    return out


def job_ids(*sources):
    """Slurm ids in allocations.json (a list), current-job.json and driver-status.json (job_id, next_job)."""
    found = []

    def add(value):
        text = str(value).strip() if isinstance(value, (int, str)) and not isinstance(value, bool) else ""
        if JOB_ID.match(text) and text not in found:
            found.append(text)

    def walk(node):
        if isinstance(node, list):
            for item in node:
                add(item)
                if isinstance(item, dict):
                    walk(item)
        elif isinstance(node, dict):
            for key in ("job_id", "next_job", "queued_by"):
                add(node.get(key))
            for key in ("allocations", "jobs"):
                walk(node.get(key))

    for source in sources:
        walk(source)
    return found


def check_job_ids(state):
    found = []
    try:
        names = os.listdir(state)
    except OSError:
        return found
    for name in sorted(names, key=lambda n: (len(n), n)):
        m = re.match(r"^slurm-check-(\d+)\.log$", name)
        if m:
            found.append(m.group(1))
    return found


def run_cmd(argv):
    try:
        p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return p.stdout if p.returncode == 0 else None


def slurm_states(ids, use_slurm):
    """{id: {state, elapsed, exit_code, reason, source}} from sacct (final) and squeue (live)."""
    rows = {}
    if not ids:
        return rows, "ok"
    if not use_slurm:
        return rows, "disabled"
    if not (shutil.which("sacct") or shutil.which("squeue")):
        return rows, "unavailable"
    joined = ",".join(ids)
    if shutil.which("sacct"):
        out = run_cmd(["sacct", "-n", "-X", "-P", "-j", joined, "-o", "JobID,State,Elapsed,ExitCode"])
        for line in (out or "").splitlines():
            parts = line.strip().split("|")
            if len(parts) >= 4 and "." not in parts[0]:
                rows[parts[0]] = {"state": parts[1].split()[0] if parts[1] else "UNKNOWN",
                                  "elapsed": parts[2], "exit_code": parts[3], "reason": "",
                                  "source": "sacct"}
    if shutil.which("squeue"):
        out = run_cmd(["squeue", "-h", "-r", "-j", joined, "-o", "%i|%T|%M|%R"])
        for line in (out or "").splitlines():
            parts = line.strip().split("|", 3)
            if len(parts) == 4:
                rows[parts[0]] = {"state": parts[1], "elapsed": parts[2], "exit_code": None,
                                  "reason": parts[3], "source": "squeue"}
    return rows, "ok"


def redact(node, user):
    if not user:
        return node
    if isinstance(node, str):
        return node.replace(user, "<user>")
    if isinstance(node, list):
        return [redact(v, user) for v in node]
    if isinstance(node, dict):
        return {redact(k, user): redact(v, user) for k, v in node.items()}
    return node


def collect(state, use_slurm=True):
    out = {"at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "state_dir": state, "state_dir_exists": os.path.isdir(state)}
    out["run_status"] = read_json(os.path.join(state, "run", "status.json"))
    out["check_run_status"] = read_json(os.path.join(state, "check-run", "status.json"))
    for name in ("driver-status.json", "backup-status.json", "allocations.json",
                 "current-job.json", "complete.json"):
        out[name[:-5].replace("-", "_")] = find_json(state, name)
    receipt = find_json(state, "check-receipt.json")
    out["check_receipt"] = ({"passed": receipt.get("passed"), "checks": receipt.get("checks")}
                            if isinstance(receipt, dict) else None)
    out["train"] = train_tail(state)

    receipts = []
    try:
        receipts = [n for n in os.listdir(os.path.join(state, "backups")) if re.match(r"^step-.*\.json$", n)]
    except OSError:
        pass
    pending = []
    try:
        for name in sorted(os.listdir(os.path.join(state, "run"))):
            if (name.startswith("step-") and name != "step-%06d" % SOURCE_STEP
                    and os.path.exists(os.path.join(state, "run", name, "manifest.json"))
                    and name + ".json" not in receipts):
                pending.append(name)
    except OSError:
        pass
    out["backups"] = {"receipts": len(receipts), "pending": pending}

    # submissions.json is written at each submission: {"check": id, "train": [ids]}. It makes a
    # queued training job visible before the driver records it, and keeps check jobs labelled as
    # checks even though they also write driver-status.json.
    submitted = read_json(os.path.join(state, "submissions.json"))
    submitted = submitted if isinstance(submitted, dict) else {}
    out["submissions"] = submitted
    check_ids = check_job_ids(state)
    for jid in job_ids([submitted["check"]] if submitted.get("check") else []):
        if jid not in check_ids:
            check_ids.append(jid)
    alloc_ids = [j for j in job_ids(out["allocations"], submitted.get("train") or []) if j not in check_ids]
    current_ids = [j for j in job_ids(out["current_job"], out["driver_status"]) if j not in check_ids]
    roles = {}
    for jid in check_ids:
        roles[jid] = "check"
    for jid in alloc_ids:
        roles[jid] = "allocation"
    for jid in current_ids:
        roles[jid] = "allocation"
    order = check_ids + [j for j in alloc_ids + current_ids if j not in check_ids]
    order = [j for i, j in enumerate(order) if j not in order[:i]]
    states, slurm = slurm_states(order, use_slurm)
    out["slurm"] = slurm
    out["current_job_ids"] = current_ids
    out["jobs"] = [dict({"id": j, "role": roles[j]}, **states.get(j, {"state": "UNKNOWN"})) for j in order]
    return out


def main(argv=None):
    user = os.environ.get("USER") or getpass.getuser()
    default = "/gpfs/scrubbed/%s/anytok-runs/regret-rl-20k" % user
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default=default)
    parser.add_argument("--no-slurm", action="store_true", help="skip sacct/squeue (tests)")
    args = parser.parse_args(argv)
    result = redact(collect(args.state.rstrip("/"), not args.no_slurm), user)
    sys.stdout.write(json.dumps(result) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
