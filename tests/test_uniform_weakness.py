"""Independent document selection and the requested loss/usefulness criterion."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from uniform_weakness import compare_scores, classify, make_panels


class Canonical:
    def __call__(self, text, **kwargs): return type('Encoding', (), {'input_ids': [ord(c) for c in text]})()
    def decode(self, ids, **kwargs): return ''.join(chr(i) for i in ids)


class Target:
    def encode(self, text, **kwargs): return type('Encoding', (), {'ids': [ord(c) for c in text for _ in range(2)]})()
    def decode(self, ids, **kwargs): return ''.join(chr(i) for i in ids[::2])


class WeaknessTest(unittest.TestCase):
    def test_document_partitions_are_independent_and_contexts_fit(self):
        rows = [{'id': str(i), 'text': 'words and unicode 中文 repeated ' * 40} for i in range(150)]
        result = make_panels(rows, 'test', Canonical(), [Target()], [set()])
        self.assertEqual(len(result['screen']), 48)
        self.assertEqual(len(result['confirm']), 96)
        self.assertFalse({r['id'] for r in result['screen']} & {r['id'] for r in result['confirm']})
        for r in result['screen'] + result['confirm']:
            self.assertLessEqual(r['maximum_positions'], 128)
            self.assertEqual(len(r['canonical_continuation_ids']), 32)
        repeat = make_panels(rows[::-1], 'test', Canonical(), [Target()], [set()])
        self.assertEqual(result, repeat)

    def test_loss_gap_is_byte_weighted_and_needs_practical_utility(self):
        import math
        base = [10 * math.log(2), 20 * math.log(2)]
        loss = [v * 1.15 for v in base]
        score = compare_scores(loss, base, [10, 20])
        self.assertAlmostEqual(score['gap_bpb'], .15)
        self.assertGreater(score['gap_ci95'][0], .1)
        good = classify(score, {'token_reduction_vs_canonical_percent': 20})
        bad = classify(score, {'token_reduction_vs_canonical_percent': -30})
        self.assertTrue(good['confirmed_weak_and_useful'])
        self.assertFalse(bad['confirmed_weak_and_useful'])
        with self.assertRaises(ValueError): compare_scores([float('nan')], [1], [1])


if __name__ == '__main__': unittest.main()
