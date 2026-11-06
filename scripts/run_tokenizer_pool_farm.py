"""Finish the saved pool across credit-limited Modal profiles and collect it.

Run under tmux. Queue every remaining configuration and dispatch measured
families without waiting for unrelated diagnostics. No tokenizer inputs
pass through this machine. Every completed shard is gathered into group-cron's
anytok-tokenizer-pool Volume before its local metadata is counted as collected.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import fcntl
import signal
import subprocess
import time

import modal

from modal_credits import billing_row
from decimal import Decimal
from modal_tokenizer_pool_transfer import source_client
from modal_generate_tokenizer_pool import estimates, load_plan

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "modal_results/tokenizer_pool"
DISABLED = {"general", "research-everettcc"}
CEILING = 29.0
PREFERRED = ["pandas", "qingy1337", "qxli2", "code-group2-evcc", "group-cron"]


def emit(stage, **fields):
    record = {"at_utc": datetime.now(timezone.utc).isoformat(), "stage": stage, **fields}
    path = STATE / "farm-status.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)
    print(json.dumps(record), flush=True)


def cli(profile, arguments, log):
    if profile in DISABLED:
        raise ValueError(f"Disabled profile: {profile}")
    with log.open("w") as stream:
        result = subprocess.run(["modal", *arguments], env={**os.environ, "MODAL_PROFILE": profile},
                                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{profile}: Modal command failed; see {log}")


def done(plan):
    result = {}
    for path in STATE.glob("[0-9][0-9][0-9][0-9].json"):
        item = json.loads(path.read_text())
        index = item["config"]["index"]
        if item["status"] != "complete" or item["config"] != plan["configs"][index]:
            raise RuntimeError(f"Invalid collected metadata: {path}")
        result[index] = item
    return result


def references(plan):
    """Cost samples need not wait for collection of a whole remote shard."""
    result = done(plan)
    for directory in STATE.iterdir():
        if not directory.is_dir() or directory.name in DISABLED:
            continue
        for path in directory.glob('[0-9][0-9][0-9][0-9].json'):
            try:
                item = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            index = item.get('config', {}).get('index')
            if (item.get('status') == 'complete' and isinstance(index, int)
                    and 0 <= index < plan['count'] and item['config'] == plan['configs'][index]):
                result[index] = item
    return result


def ready_indices(plan, completed, reserved, samples):
    measured = {(m['config']['algorithm'], m['config']['pretokenizer']) for m in samples.values()}
    return [c['index'] for c in plan['configs'] if c['index'] not in completed
            and c['index'] not in reserved and (c['algorithm'], c['pretokenizer']) in measured]


def process_alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sync_metadata(profile, plan):
    for path in (STATE / profile).glob("[0-9][0-9][0-9][0-9].json"):
        raw = path.read_text()
        item = json.loads(raw)
        index = item["config"]["index"]
        if item["status"] != "complete" or item["config"] != plan["configs"][index]:
            raise RuntimeError("Invalid shard metadata")
        destination = STATE / path.name
        if destination.exists() and json.loads(destination.read_text()) != item:
            raise RuntimeError(f"Conflicting model metadata: {index}")
        temporary = destination.with_suffix(f".{profile}.tmp")
        temporary.write_text(raw)
        temporary.replace(destination)


def gather(profile, plan, label):
    cli("group-cron", ["run", "--detach", "scripts/modal_tokenizer_pool_transfer.py",
        "--action", "gather", "--source-profile", profile], STATE / f"gather-{profile}-{label}.log")
    sync_metadata(profile, plan)


def allocate(costs, budgets):
    remaining = dict(budgets)
    assignments = {profile: [] for profile in budgets}
    for index, cost in sorted(costs.items(), key=lambda item: (-item[1], item[0])):
        profile = max(remaining, key=remaining.get)
        if cost > remaining[profile]:
            raise ValueError("Insufficient profile capacity for remaining tokenizers")
        assignments[profile].append(index)
        remaining[profile] -= cost
    return {p: indices for p, indices in assignments.items() if indices}


def capacity(costs, excluded):
    listing = subprocess.run(["modal", "profile", "list", "--json"], capture_output=True,
                             text=True, check=True)
    names = {item["name"] for item in json.loads(listing.stdout)} - DISABLED - excluded
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(lambda name: billing_row(name, "this month", Decimal(30)), sorted(names)))
    budgets = {r["profile"]: max(0.0, CEILING - float(r["credits_used_usd"]) - 0.10)
               for r in rows if r["status"] == "ok"}
    order = [p for p in PREFERRED if p in budgets]
    order.extend(sorted((p for p in budgets if p not in order), key=lambda p: -budgets[p]))
    selected = {}
    for profile in order:
        if budgets[profile] < max(costs.values()):
            continue
        try:
            # Empty app creation checks the workspace spend limit without
            # starting a container. A zero-usage month does not prove credits
            # are currently usable (as demonstrated by aurora).
            client, _ = source_client(profile)
            with modal.App("anytok-pool-capacity-check").run(client=client):
                pass
        except Exception as exc:
            emit("profile_unavailable", profile=profile, reason=str(exc))
            excluded.add(profile)
            continue
        selected[profile] = budgets[profile]
        try:
            return allocate(costs, selected)
        except ValueError:
            continue
    raise RuntimeError("Insufficient verified profile credits for remaining pool")


def run_shard(profile, indices, plan, round_number):
    directory = STATE / profile
    directory.mkdir(exist_ok=True)
    if profile != 'group-cron':
        cli(profile, ["run", "--detach", "scripts/modal_tokenizer_pool_transfer.py",
                      "--action", "import-corpus"], directory / f"corpus-copy-round-{round_number}.log")
    error = None
    try:
        cli(profile, ["run", "--detach", "scripts/modal_generate_tokenizer_pool.py",
                      "--stage", "selected", "--indices", ",".join(map(str, indices))],
            directory / f"round-{round_number}.log")
    except RuntimeError as exc:
        error = str(exc)
    # Save every completed model even when another input failed.
    if profile != 'group-cron':
        gather(profile, plan, str(round_number))
    if error:
        raise RuntimeError(error)


def finish(plan, current):
    if set(current) != set(range(plan["count"])):
        raise RuntimeError("Final index does not contain every planned tokenizer")
    for index, manifest in current.items():
        if manifest["vocab_size_actual"] != plan["configs"][index]["vocab_size"]:
            raise RuntimeError(f"Vocabulary size mismatch: {index}")
        slack = plan["total_bytes_per_tokenizer"] - manifest["total_used_bytes"]
        if not 0 <= slack <= 3 * len(plan["recipe"]["buckets"]):
            raise RuntimeError(f"Unexpected corpus byte count: {index}")
    path = STATE / "full-results.json"
    path.write_text(json.dumps({"plan_sha256": plan["plan_sha256"], "completed": len(current),
        "tokenizers": [current[i] for i in sorted(current)]}, indent=2) + "\n")
    client, _ = source_client("group-cron")
    volume = modal.Volume.from_name("anytok-tokenizer-pool", version=2, client=client)
    with volume.batch_upload(force=True) as batch:
        batch.put_file(path, f"pools/{plan['plan_sha256'][:16]}/full-results.json")
    emit("complete", completed=len(current), volume="anytok-tokenizer-pool",
         total_tokenizer_bytes=sum(m["tokenizer_bytes"] for m in current.values()))


def main():
    STATE.mkdir(exist_ok=True, parents=True)
    # A second controller must never dispatch duplicate configurations.
    lock = (STATE / 'farm.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = load_plan()
    adopted_path = STATE / 'adopted-diagnostics.json'
    adopted = json.loads(adopted_path.read_text()) if adopted_path.exists() else None
    excluded = {'aurora'}
    reserved = set(adopted['indices']) if adopted else set()
    pending = {c['index'] for c in plan['configs']} - done(plan).keys() - reserved
    queue = {'plan_sha256': plan['plan_sha256'], 'created_at_utc': datetime.now(timezone.utc).isoformat(),
             'queued_indices': sorted(pending), 'adopted': adopted,
             'policy': 'Measured families dispatch independently; unmeasured families stay queued.'}
    atomic_json(STATE / 'farm-queue.json', queue)
    emit('queued', completed=len(done(plan)), queued=len(pending), adopted=len(reserved))
    round_number = max([int(p.stem.rsplit('-', 1)[1]) for p in STATE.glob('farm-round-*.json')] or [0])
    last_capacity = 0
    failures = []
    futures = {}
    with ThreadPoolExecutor(max_workers=40) as pool:
        while True:
            current = done(plan)
            if adopted and not process_alive(adopted['launcher_pid']):
                try:
                    gather(adopted['profile'], plan, 'adopted-diagnostics')
                    missing = set(adopted['indices']) - done(plan).keys()
                    if missing:
                        failures.append({'profile': adopted['profile'], 'indices': sorted(missing),
                                         'error': 'Existing diagnostic did not complete; inspect diagnostics.log'})
                        reserved.update(missing)
                finally:
                    # Retire the obsolete controller only after its expensive
                    # launcher has exited. No remote app is stopped.
                    pid = adopted.get('legacy_controller_pid')
                    if pid and process_alive(pid):
                        os.kill(pid, signal.SIGTERM)
                        os.kill(pid, signal.SIGCONT)
                    adopted_path.unlink(missing_ok=True)
                    adopted = None
                current = done(plan)
                emit('diagnostics_collected', completed=len(current), errors=failures)
            for future in list(futures):
                if not future.done():
                    continue
                profile, indices = futures.pop(future)
                try:
                    future.result()
                except Exception as exc:
                    failures.append({'profile': profile, 'indices': sorted(set(indices) - done(plan).keys()),
                                     'error': str(exc)})
                emit('shard_finished', profile=profile, completed=len(done(plan)), errors=failures)
            current = done(plan)
            if len(current) == plan['count']:
                finish(plan, current)
                return
            samples = references(plan)
            eligible = ready_indices(plan, current, reserved, samples)
            busy = {profile for profile, _ in futures.values()}
            if adopted:
                busy.add(adopted['profile'])
            if eligible and time.monotonic() - last_capacity >= 60:
                last_capacity = time.monotonic()
                costs = estimates(plan, current, eligible, samples)
                try:
                    assignments = capacity(costs, excluded | busy)
                except RuntimeError as exc:
                    emit('waiting_for_capacity', completed=len(current), ready=len(eligible), reason=str(exc))
                    if not futures and not adopted:
                        raise
                else:
                    round_number += 1
                    atomic_json(STATE / f'farm-round-{round_number}.json', {
                        'plan_sha256': plan['plan_sha256'], 'assignments': assignments,
                        'estimated_remaining_cost_usd': sum(costs.values()),
                        'max_containers_per_profile': 100})
                    for profile, indices in assignments.items():
                        reserved.update(indices)
                        futures[pool.submit(run_shard, profile, indices, plan, round_number)] = (profile, indices)
                    emit('running', round=round_number, completed=len(current),
                         queued_unassigned=plan['count'] - len(current) - len(reserved - current.keys()),
                         estimated_remaining_cost_usd=sum(costs.values()),
                         profile_jobs={p: len(indices) for p, indices in assignments.items()})
            if not eligible and not futures and not adopted:
                raise RuntimeError(f'Pool incomplete: {failures or "No successful cost sample for remaining families"}')
            time.sleep(10)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        emit('stopped', reason=str(exc))
        raise
