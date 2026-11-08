import math
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from uniform_trajectory import paired_slope, select_pool_candidates, feature_aliases
from analyze_uniform_trajectory import curve_bootstrap, slope_summary


class TrajectoryTest(unittest.TestCase):
    def test_curve_analysis_preserves_pairing_and_document_clusters(self):
        canonical = np.array([[1., 2., 3.], [.9, 2.03, 3.2]])
        target = canonical + np.array([[.2, .4, .6], [.1, .2, .3]])
        counts = np.array([10, 20, 30])
        a = curve_bootstrap(canonical, target, counts, ['a', 'b', 'c'], replicates=500)
        b = curve_bootstrap(np.repeat(canonical, 3, axis=1), np.repeat(target, 3, axis=1),
                            np.repeat(counts, 3), ['a']*3 + ['b']*3 + ['c']*3, replicates=500)
        for metric in a:
            np.testing.assert_allclose(a[metric][0], b[metric][0], atol=1e-12)
            np.testing.assert_allclose(a[metric][1], b[metric][1], atol=1e-12)
        np.testing.assert_allclose(a['gap'][1], a['transferred'][1] - a['canonical'][1])
        slope, _ = slope_summary([15000, 20000], *a['gap'], [15000, 20000])
        self.assertAlmostEqual(slope['slope_bpb_per_1k'], -.01 / math.log(2) / 5)
        self.assertFalse(slope['equivalent_to_flat_within_margin']['0.002'])

    def test_known_slope_and_nonsignificance_are_distinct_from_flatness(self):
        steps = [15000, 16000, 17000, 18000, 19000, 20000]
        counts = np.array([10, 20, 30.])
        def scores(slopes):
            return [(1 + (s / 1000 - 15) * np.asarray(slopes)) * counts * math.log(2) for s in steps]
        result = paired_slope(steps, scores([-.01] * 3), counts)
        self.assertAlmostEqual(result['slope_bpb_per_1k'], -.01)
        self.assertTrue(result['clear_improvement'])
        self.assertFalse(result['equivalent_to_flat_within_margin']['0.002'])
        flat = paired_slope(steps, scores([0.] * 3), counts)
        self.assertTrue(flat['equivalent_to_flat_within_margin']['0.002'])
        noisy = paired_slope(steps, scores([-.03, .01, .02]), counts)
        self.assertLess(noisy['ci95'][0], 0)
        self.assertGreater(noisy['ci95'][1], 0)
        self.assertFalse(noisy['equivalent_to_flat_within_margin']['0.002'])

    def test_repeated_passages_are_bootstrapped_as_document_clusters(self):
        steps = [15000, 17000, 20000]
        original = np.array([[1., 2., 3.], [.95, 2.01, 3.1], [.9, 2.03, 3.2]])
        counts = [10, 20, 30]
        a = paired_slope(steps, original, counts, clusters=['a', 'b', 'c'])
        b = paired_slope(steps, np.repeat(original, 3, axis=1), np.repeat(counts, 3),
                         clusters=['a'] * 3 + ['b'] * 3 + ['c'] * 3)
        np.testing.assert_allclose(a['ci95'], b['ci95'], atol=1e-12)
        self.assertEqual(b['clusters'], 3)

    def test_pool_selection_uses_compression_without_loss_or_noise_preferences(self):
        def row(index, compression, noise):
            return {'index': index, 'config': {'algorithm': 'bpe', 'pretokenizer': 'regex', 'noise_per_million': noise},
                    'loss': 1000 - index,
                    'utility': {d: {'valid': True, 'token_reduction_percent': compression}
                                for d in ['web-zh', 'web-ja', 'medical-en', 'code']}}
        chosen = select_pool_candidates([row(0, 11, 0), row(1, 30, 700000)])
        self.assertEqual([r['index'] for r in chosen], [1])

    def test_aliases_respect_priors_and_ignore_control_rows(self):
        forms = np.array([[1, 2], [1, 2], [1, 2], [3, 4]], dtype=np.int32)
        result = feature_aliases(forms, [0, 0, -1, 0], {3}, [0, 1, 2, 2])
        self.assertEqual(result['aliased_vocabulary_tokens'], 2)
        self.assertEqual(result['aliased_prefix_token_fraction'], .5)
        result = feature_aliases(forms, [0, 0, -1, 0], {0, 3}, [1, 2])
        self.assertEqual(result['aliased_vocabulary_tokens'], 0)


if __name__ == '__main__': unittest.main()
