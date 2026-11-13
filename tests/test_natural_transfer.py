import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from natural_transfer import DOMAINS, build_panel, metrics, text_tags, validate_scores


class CharacterTokenizer:
    def __init__(self):
        self._tokenizer = self

    def encode(self, text, **kwargs):
        return SimpleNamespace(ids=[ord(c) + 1 for c in text])

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=self.encode(text).ids)

    def decode(self, ids, **kwargs):
        return ''.join(chr(i - 1) for i in ids)


class NaturalTransferTest(unittest.TestCase):
    def test_gap_weighting_tails_and_negative_values(self):
        rows = [dict(continuation_bytes=b, cluster_id=str(i)) for i, b in enumerate([1, 100])]
        baseline = [[2 * b * math.log(2) / 32] * 32 for b in [1, 100]]
        target = [[3 * math.log(2) / 32] * 32, [100 * math.log(2) / 32] * 32]
        import numpy as np
        delta = np.array(target) - np.array(baseline)
        result = metrics(rows, delta, baseline, target)
        self.assertAlmostEqual(result['gap_bpb'], -99 / 101)
        self.assertAlmostEqual(result['mean_passage_gap_bpb'], 0)
        self.assertAlmostEqual(result['worst_decile_mean_gap_bpb'], 1)

    def test_descriptive_tags_do_not_depend_on_loss(self):
        tags = text_tags('Recorded value:123456\nOther value:456789\n', '123456', 'web-en')
        self.assertTrue(tags['numeric_dense'])
        self.assertTrue(tags['structured'])
        self.assertTrue(tags['repeated_number'])
        self.assertFalse(tags['image_markup'])
        self.assertTrue(text_tags('![](abc', '.png){#sp1}', 'medical-en')['image_markup'])

    def test_two_windows_and_disjoint_source_cohorts(self):
        canonical = CharacterTokenizer()
        sources = {d: [dict(id=str(i), text=(f'Document {d} {i} contains useful ordinary text. ' * 50))
                       for i in range(4)] for d in DOMAINS}
        rows, report = build_panel(sources, canonical, {7: canonical}, {7: {0}},
                                  {d: {'0'} for d in DOMAINS}, {d: 1 for d in DOMAINS})
        screen = {r['cluster_id'] for r in rows if r['split'] == 'screen'}
        confirm = {r['cluster_id'] for r in rows if r['split'] == 'confirm'}
        self.assertFalse(screen & confirm)
        self.assertEqual(len(rows), 20)
        for cluster in screen | confirm:
            pair = [r for r in rows if r['cluster_id'] == cluster]
            self.assertLessEqual(pair[0]['source_window_end'], pair[1]['source_window_start'])
            self.assertNotEqual(pair[0]['source_document_id'], '0')
            self.assertTrue(all(len(r['canonical_continuation_ids']) == 32 for r in pair))

    def test_malformed_scores_rejected(self):
        with self.assertRaises(ValueError):
            validate_scores([{}], [[0] * 31])
        with self.assertRaises(ValueError):
            validate_scores([{}], [[float('nan')] * 32])


if __name__ == '__main__':
    unittest.main()
