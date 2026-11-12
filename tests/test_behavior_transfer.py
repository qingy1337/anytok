import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from behavior_transfer import answer_present, summarize


class BehaviorMetricTest(unittest.TestCase):
    def setUp(self):
        self.row = {'id': 'copy', 'split': 'screen', 'family': 'numbers',
            'size': 'short', 'distance': 'near', 'answer_ids': [7, 8],
            'answer_bytes': 2, 'continuation_bytes': 100}

    def test_answer_gap_cannot_be_diluted_by_easy_trailing_text(self):
        score = {'nats': [math.log(2)] * 2 + [0.] * 30, 'correct': [False, True] + [True] * 30}
        result = summarize([self.row], [score])[0]
        self.assertAlmostEqual(result['answer_bpb'], 1.)
        self.assertAlmostEqual(result['bpb_32'], .02)
        self.assertEqual(result['answer_exact_match'], 0.)

    def test_ignores_tail_errors_for_answer_accuracy_but_includes_them_in_full_loss(self):
        score = {'nats': [0.] * 2 + [math.log(2)] * 30, 'correct': [True] * 2 + [False] * 30}
        result = summarize([self.row], [score])[0]
        self.assertEqual(result['answer_exact_match'], 1.)
        self.assertEqual(result['answer_bpb'], 0.)
        self.assertAlmostEqual(result['bpb_32'], .3)

    def test_rejects_missing_or_invalid_token_scores(self):
        with self.assertRaises(ValueError):
            summarize([self.row], [])
        with self.assertRaises(ValueError):
            summarize([self.row], [{'nats': [float('nan')] * 32, 'correct': [True] * 32}])

    def test_value_check_allows_formatting_but_not_inexact_numbers(self):
        row = {'answer': ' 12345678', 'family': 'numbers', 'variant': 0}
        self.assertTrue(answer_present('The value is 12,345,678.0', row))
        self.assertFalse(answer_present('The value is 912345678', row))
        self.assertFalse(answer_present('The value is12345679', row))
        self.assertTrue(answer_present('Recorded: CLARA\n BENNETT.', {'answer': ' Clara Bennett', 'family': 'names'}))


if __name__ == '__main__':
    unittest.main()
