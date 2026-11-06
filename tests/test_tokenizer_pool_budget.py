"""Shard budgets must cover only assigned inputs and fail safely for unknown costs."""
import importlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
launcher = importlib.import_module("modal_generate_tokenizer_pool")


class ShardBudgetTest(unittest.TestCase):
    def setUp(self):
        self.plan = {"count": 3, "configs": [
            {"index": 0, "algorithm": "bpe", "pretokenizer": "regex"},
            {"index": 1, "algorithm": "unigram", "pretokenizer": "no-regex"},
            {"index": 2, "algorithm": "bpe", "pretokenizer": "regex"},
        ]}
        self.references = {2: {"config": self.plan["configs"][2],
                               "estimated_process_cost_usd": 0.01}}

    def test_shard_does_not_include_unassigned_or_completed_inputs(self):
        costs = launcher.estimates(self.plan, {}, [0], self.references)
        self.assertEqual(set(costs), {0})
        self.assertAlmostEqual(costs[0], 0.016)
        costs = launcher.estimates(self.plan, self.references, [0, 2], self.references)
        self.assertEqual(set(costs), {0})

    def test_missing_measurement_is_not_reported_as_free(self):
        with self.assertRaises(RuntimeError):
            launcher.estimates(self.plan, {}, [1], self.references)
        self.assertEqual(launcher.estimates(self.plan, {}, [1], self.references,
                                            allow_unknown=True), {1: 5.0})


if __name__ == "__main__":
    unittest.main()
