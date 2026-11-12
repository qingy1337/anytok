import copy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from pool_rl_sweep import rank_results, summarize_candidate
from uniform_trajectory import PROFILE_DOMAINS


class PoolSweepTest(unittest.TestCase):
    def setUp(self):
        self.panel = [{'domain': d, 'id': d, 'continuation_bytes': 10 * (i + 1)}
                      for i, d in enumerate(PROFILE_DOMAINS)]
        self.base = [r['continuation_bytes'] * math.log(2) for r in self.panel]

    def result(self, index, gaps):
        nats = [b + gap * row['continuation_bytes'] * math.log(2)
                for b, gap, row in zip(self.base, gaps, self.panel)]
        return summarize_candidate(index, {}, self.panel, nats, self.base)

    def test_equal_domain_ranking_is_distinct_from_byte_weighting(self):
        a = self.result(0, [2.] + [0.] * 8)
        b = self.result(1, [.2] * 9)
        self.assertGreater(a['mean_domain_gap_bpb'], b['mean_domain_gap_bpb'])
        self.assertLess(a['pooled_gap_bpb'], b['pooled_gap_bpb'])
        self.assertEqual(rank_results([b, a], range(2))[0]['index'], 0)

    def test_rejects_changed_contexts_missing_rows_and_baselines(self):
        a = self.result(0, [.1] * 9)
        b = self.result(1, [.2] * 9)
        for change in [lambda r: r.update(panel_sha256='changed'),
                       lambda r: r['canonical_nats'].__setitem__(0, 99.)]:
            row = copy.deepcopy(b)
            change(row)
            with self.assertRaises(ValueError):
                rank_results([a, row], range(2))
        with self.assertRaises(ValueError):
            rank_results([a], range(2))
        with self.assertRaises(ValueError):
            rank_results([a, a], range(2))

    def test_rejects_nonfinite_losses_and_preserves_counts(self):
        row = self.result(0, [.1] * 9)
        self.assertAlmostEqual(row['mean_domain_gap_bpb'], .1)
        self.assertEqual(row['domains']['web-en']['continuation_tokens'], 32)
        with self.assertRaises(ValueError):
            summarize_candidate(0, {}, self.panel, [float('nan')] * 9, self.base)


if __name__ == '__main__':
    unittest.main()
