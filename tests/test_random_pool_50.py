import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_random_pool_50 import analyze
from pool_generalization import digest


class RandomPoolReferenceTest(unittest.TestCase):
    def data(self):
        seed = 20261117
        sample = random.Random(seed).sample(list(range(1000)), 50)
        hard = sample[:2] + [i for i in range(1000) if i not in sample][:22]
        indices = sorted(set(sample) | set(hard))
        rows = []
        for domain in ['web-en', 'code', 'math-en', 'medical-en', 'legal-en']:
            for doc in range(2):
                for window in range(2):
                    cluster = f'{domain}/{doc}'
                    rows.append(dict(id=cluster + '/' + str(window), cluster_id=cluster,
                                     domain=domain, continuation_bytes=32))
        plan = dict(seed=seed, population_size=1000, random_indices=sample,
            random_indices_sha256=digest(sample), hard_indices=hard, indices=indices, bootstrap_replicates=200)
        models = {}
        for name, canonical in [('zett20k', 1.2), ('uniform20k', 1.), ('rl20k', .8)]:
            models[name] = dict(canonical_nats=[canonical * 32 * math.log(2)] * len(rows),
                transferred_nats=[[(canonical + (.2 if i in hard else .1)) * 32 * math.log(2)] * len(rows) for i in indices])
        tokens = [dict(index=i, config=dict(algorithm='bpe', pretokenizer='regex')) for i in indices]
        return dict(plan=plan, preparation=dict(metadata=rows, trimmed_passages=0), models=models, tokenizers=tokens)

    def test_true_sample_mean_and_overlap_are_preserved(self):
        result = analyze(self.data())
        self.assertEqual(len(result['overlap']), 2)
        for row in result['models']:
            self.assertAlmostEqual(row['random50_mean_gap'], .104)
            self.assertAlmostEqual(row['hard24_mean_gap'], .2)
            self.assertAlmostEqual(row['hard_minus_random'], .096)
            self.assertLess(row['random50_ci95'][0], row['random50_ci95'][1])

    def test_draw_cannot_be_changed_even_if_hash_is_updated(self):
        data = self.data()
        data['plan']['random_indices'].reverse()
        data['plan']['random_indices_sha256'] = digest(data['plan']['random_indices'])
        with self.assertRaisesRegex(ValueError, 'draw changed'):
            analyze(data)

    def test_nonfinite_losses_are_rejected(self):
        data = self.data();data['models']['rl20k']['canonical_nats'][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Invalid loss'):
            analyze(data)


if __name__ == '__main__':
    unittest.main()
