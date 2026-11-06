"""Assignments must preserve every input while respecting profile budgets."""
import importlib
from pathlib import Path
import sys
import unittest
import tempfile
import json
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
farm = importlib.import_module("run_tokenizer_pool_farm")


class AllocationTest(unittest.TestCase):
    def test_disjoint_complete_assignments_fit_each_budget(self):
        costs = {0: 6.0, 1: 4.0, 2: 3.0, 3: 2.0}
        budgets = {"a": 8.0, "b": 8.0}
        assignments = farm.allocate(costs, budgets)
        flattened = [i for indices in assignments.values() for i in indices]
        self.assertEqual(sorted(flattened), sorted(costs))
        self.assertEqual(len(flattened), len(set(flattened)))
        for profile, indices in assignments.items():
            self.assertLessEqual(sum(costs[i] for i in indices), budgets[profile])

    def test_insufficient_budget_does_not_drop_a_configuration(self):
        with self.assertRaises(ValueError):
            farm.allocate({0: 9.0}, {"a": 8.0})

    def test_ready_families_do_not_wait_for_unmeasured_family(self):
        plan = {'count': 5, 'configs': [
            {'index': i, 'algorithm': 'bpe' if i < 3 else 'unigram',
             'pretokenizer': 'no-regex' if i < 3 else 'line256'} for i in range(5)]}
        sample = {3: {'config': plan['configs'][3]}}
        self.assertEqual(farm.ready_indices(plan, {3}, {0, 1}, sample), [4])
        sample[0] = {'config': plan['configs'][0]}
        self.assertEqual(farm.ready_indices(plan, {3}, {0, 1}, sample), [2, 4])
        self.assertEqual(farm.ready_indices(plan, {3, 4}, {0, 1, 2}, sample), [])

    def test_uncollected_manifest_can_supply_cost_without_being_counted_collected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(farm, 'STATE', Path(tmp)):
            plan = {'count': 1, 'configs': [{'index': 0}]}
            directory = Path(tmp) / 'profile'
            directory.mkdir()
            (directory / '0000.json').write_text(json.dumps({
                'status': 'complete', 'config': plan['configs'][0], 'estimated_process_cost_usd': .2}))
            self.assertEqual(farm.done(plan), {})
            self.assertEqual(farm.references(plan)[0]['estimated_process_cost_usd'], .2)


if __name__ == "__main__":
    unittest.main()
