"""Read Modal billing across profiles without starting compute.

    python scripts/modal_credits.py
    python scripts/modal_credits.py test-joining group-cron
    python scripts/modal_credits.py --json

Remaining credits are estimated as the monthly allowance minus credits applied.
The default $30 allowance is a user-provided assumption, not an API balance.
Disabled profiles are never queried, even when explicitly selected.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
import os
import subprocess
import sys


DISABLED_PROFILES = frozenset({"general", "research-everettcc"})


def modal_json(arguments: list[str], profile: str | None = None):
    env = os.environ.copy()
    if profile is not None:
        env["MODAL_PROFILE"] = profile
    result = subprocess.run(
        ["modal", *arguments], env=env, capture_output=True, text=True, timeout=60,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or
                           f"Modal exited with status {result.returncode}")
    return json.loads(result.stdout)


def billing_row(profile: str, cycle: str, allowance: Decimal) -> dict:
    row = {"profile": profile}
    if profile in DISABLED_PROFILES:
        return {**row, "status": "disabled"}
    try:
        summary = modal_json(["billing", "summary", "--for", cycle, "--json"], profile)
        # Missing credits must be an error, rather than looking like an unused account.
        used = -Decimal(summary["adjustments"]["credits"])
        billed = Decimal(summary["billed_cost"])
        if not used.is_finite() or not billed.is_finite():
            raise ValueError("Billing returned a non-finite amount")
        return {
            **row, "status": "ok", "credits_used_usd": str(used),
            "estimated_remaining_usd": str(allowance - used),
            "billed_cost_usd": str(billed),
        }
    except (RuntimeError, OSError, subprocess.TimeoutExpired, ValueError,
            KeyError, TypeError, InvalidOperation) as exc:
        return {**row, "status": "error", "error": str(exc)}


def positive_amount(value: str) -> Decimal:
    try:
        amount = Decimal(value)
        if amount.is_finite() and amount > 0:
            return amount
    except InvalidOperation:
        pass
    raise argparse.ArgumentTypeError("Allowance must be a positive finite USD amount")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profiles", nargs="*", help="Default: all configured profiles")
    parser.add_argument("--for", dest="cycle", default="this month",
                        help="Billing month, e.g. 2026-11 (default: this month)")
    parser.add_argument("--allowance", type=positive_amount, default=Decimal("30"),
                        help="Assumed monthly credits per profile in USD (default: 30)")
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable report")
    args = parser.parse_args()
    try:
        configured = modal_json(["profile", "list", "--json"])
        names = {profile["name"] for profile in configured}
        selected = list(dict.fromkeys(args.profiles)) if args.profiles else sorted(names)
        unknown = set(selected) - names
        if unknown:
            parser.error("Unknown profiles: " + ", ".join(sorted(unknown)))
        with ThreadPoolExecutor(max_workers=6) as executor:
            rows = list(executor.map(
                lambda profile: billing_row(profile, args.cycle, args.allowance), selected,
            ))
    except (RuntimeError, OSError, subprocess.TimeoutExpired, ValueError, KeyError,
            TypeError) as exc:
        print(f"Could not read Modal profiles: {exc}", file=sys.stderr)
        return 1
    rows.sort(key=lambda row: (
        {"ok": 0, "error": 1, "disabled": 2}[row["status"]],
        -Decimal(row.get("estimated_remaining_usd", "0")), row["profile"],
    ))
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "billing_cycle": args.cycle,
        "assumed_allowance_usd": str(args.allowance),
        "balance_method": "assumed allowance minus credits applied; billing may lag",
        "profiles": rows,
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Billing: {args.cycle} | Assumed allowance: ${args.allowance:.2f}/profile")
        print("Remaining is an estimate; billing may lag. Disabled profiles are not queried.\n")
        width = max([7] + [len(row["profile"]) for row in rows])
        print(f"{'Profile':<{width}}  {'Used':>9}  {'Est. left':>10}  {'Billed':>9}")
        for row in rows:
            name = row["profile"]
            if row["status"] == "ok":
                used, left, billed = (Decimal(row[key]) for key in (
                    "credits_used_usd", "estimated_remaining_usd", "billed_cost_usd"))
                print(f"{name:<{width}}  ${used:>8.2f}  ${left:>9.2f}  ${billed:>8.2f}")
            elif row["status"] == "disabled":
                print(f"{name:<{width}}  DISABLED — skipped")
            else:
                print(f"{name:<{width}}  ERROR: {row['error']}")
    return int(any(row["status"] == "error" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
