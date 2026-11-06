"""Verify balance arithmetic and that excluded accounts never receive requests."""

import importlib.util
from decimal import Decimal
from pathlib import Path
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "modal_credits", Path(__file__).resolve().parents[1] / "scripts/modal_credits.py")
credits = importlib.util.module_from_spec(spec)
spec.loader.exec_module(credits)


class CreditsTest(unittest.TestCase):
    def test_disabled_profiles_are_never_queried(self):
        with patch.object(credits, "modal_json") as query:
            for profile in credits.DISABLED_PROFILES:
                self.assertEqual(credits.billing_row(profile, "this month", Decimal(30)),
                                 {"profile": profile, "status": "disabled"})
            query.assert_not_called()

    def test_decimal_balance_and_profile_selection(self):
        with patch.object(credits, "modal_json", return_value={
            "adjustments": {"credits": "-28.29000000"}, "billed_cost": "0E-8",
        }) as query:
            row = credits.billing_row("test-joining", "2026-11", Decimal(30))
            self.assertEqual(Decimal(row["estimated_remaining_usd"]), Decimal("1.71"))
            query.assert_called_once_with(
                ["billing", "summary", "--for", "2026-11", "--json"], "test-joining")

    def test_exhausted_allowance_is_not_clamped(self):
        with patch.object(credits, "modal_json", return_value={
            "adjustments": {"credits": "-31.25"}, "billed_cost": "1.25",
        }):
            row = credits.billing_row("example", "this month", Decimal(30))
            self.assertEqual(Decimal(row["estimated_remaining_usd"]), Decimal("-1.25"))

    def test_missing_or_failed_billing_does_not_report_full_balance(self):
        for response in ({"adjustments": {}, "billed_cost": "0"},
                         {"adjustments": {"credits": "NaN"}, "billed_cost": "0"}):
            with patch.object(credits, "modal_json", return_value=response):
                row = credits.billing_row("example", "this month", Decimal(30))
                self.assertEqual(row["status"], "error")
                self.assertNotIn("estimated_remaining_usd", row)
        with patch.object(credits, "modal_json", side_effect=RuntimeError("access denied")):
            self.assertEqual(credits.billing_row("example", "this month", Decimal(30))
                             ["status"], "error")


if __name__ == "__main__":
    unittest.main()
