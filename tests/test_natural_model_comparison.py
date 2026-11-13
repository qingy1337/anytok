import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_natural_model_comparison import analyze
from pool_generalization import digest


class ModelComparisonTest(unittest.TestCase):
    def data(self):
        rows, selected = [], []
        for domain in ['web-en', 'code', 'math-en', 'medical-en', 'legal-en']:
            for split in ['screen', 'confirm']:
                for document in range(2):
                    for window in range(2):
                        cluster = f'{domain}/{split}/{document}'
                        row = dict(id=cluster + '/' + str(window), cluster_id=cluster,
                            document_sha256=cluster, split=split, domain=domain, continuation_bytes=32)
                        rows.append(row)
                        if document == 0:
                            selected.append(row['id'])
        settings = [dict(index=i, config=dict(algorithm='bpe', pretokenizer=p, vocab_size=v))
                    for i, p, v in [(3, 'line256', 32000), (11, 'regex', 64000),
                                    (771, 'no-regex', 127000), (989, 'line256', 128000)]]
        plan = dict(indices=[r['index'] for r in settings], representative_indices=[r['index'] for r in settings],
            context_selected_ids=selected, context_selected_ids_sha256=digest(selected),
            bootstrap_replicates=100, primary_tokenizer=771, large_vocab_minimum=96000)
        models = {}
        for name, canonical, gap in [('zett20k', 1.2, .3), ('uniform20k', 1., .2), ('rl20k', .8, .3)]:
            models[name] = dict(canonical_nats=[canonical * 32 * math.log(2)] * len(rows),
                transferred_nats=[[(canonical + gap) * 32 * math.log(2)] * len(rows) for _ in settings])
        return dict(plan=plan, metadata=rows, models=models, tokenizers=settings)

    def test_better_absolute_loss_can_coexist_with_worse_transfer_gap(self):
        result = analyze(self.data())
        row = next(r for r in result['paired_comparisons'] if r['comparison'] == 'rl20k-minus-uniform20k'
            and r['regime'] == 'context' and r['split'] == 'confirm' and r['cohort'] == 'weak_pretokenization')
        self.assertAlmostEqual(row['mean_absolute_bpb_change'], -.1)
        self.assertAlmostEqual(row['mean_gap_change_bpb'], .1)
        self.assertAlmostEqual(row['canonical_bpb_change'], -.2)
        self.assertTrue(all(abs(v + .1) < 1e-12 for v in row['mean_absolute_bpb_change_ci95']))

    def test_fixed_membership_and_cluster_counts_are_preserved(self):
        result = analyze(self.data())
        sizes = next(r for r in result['sizes'] if r['regime'] == 'context' and r['split'] == 'confirm')
        self.assertEqual(sizes['passages'], 10)
        self.assertEqual(sizes['documents'], 5)
        self.assertEqual(sizes['continuation_tokens'], 320)
        data = self.data();data['plan']['context_selected_ids'].append('not-a-document')
        with self.assertRaisesRegex(ValueError, 'membership changed'):
            analyze(data)

    def test_invalid_loss_totals_are_rejected(self):
        data = self.data();data['models']['rl20k']['canonical_nats'][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Invalid loss totals'):
            analyze(data)


if __name__ == '__main__':
    unittest.main()
