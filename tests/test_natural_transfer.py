import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from natural_transfer import DOMAINS, align_panel, build_panel, metrics, text_tags, validate_scores
from pool_generalization import digest
from analyze_natural_transfer import analyze, bootstrap_balanced


class CharacterTokenizer:
    def __init__(self):
        self._tokenizer = self

    def encode(self, text, **kwargs):
        return SimpleNamespace(ids=[ord(c) + 1 for c in text], offsets=[(i, i + 1) for i in range(len(text))])

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
        aligned, audit = align_panel(rows, sources, canonical, {7: canonical}, {7: {0}})
        self.assertEqual([r['id'] for r in aligned], [r['id'] for r in rows])
        self.assertTrue(audit['all_canonical_joint_encodings_match'])
        for row in aligned:
            suffix = canonical.decode(row['canonical_continuation_ids'])
            self.assertEqual(canonical(row['prefix'] + suffix).input_ids,
                             row['canonical_prefix_ids'] + row['canonical_continuation_ids'])
            self.assertLessEqual(row['maximum_positions'], 128)

    def test_malformed_scores_rejected(self):
        with self.assertRaises(ValueError):
            validate_scores([{}], [[0] * 31])
        with self.assertRaises(ValueError):
            validate_scores([{}], [[float('nan')] * 32])

    def test_cut_inside_real_bpe_merge_is_moved_to_a_valid_boundary(self):
        import hashlib
        from tokenizers import Tokenizer, models, decoders
        backend = Tokenizer(models.BPE(vocab={'<unk>': 0, 'a': 1, 'b': 2, ' ': 3, 'ab': 4},
                                       merges=[('a', 'b')], unk_token='<unk>'))
        backend.decoder = decoders.Fuse()
        class Canonical:
            _tokenizer = backend
            def __call__(self, text, **kwargs):
                return SimpleNamespace(input_ids=backend.encode(text, add_special_tokens=False).ids)
            def decode(self, ids, **kwargs):
                return backend.decode(ids, skip_special_tokens=False)
        canonical = Canonical()
        text = 'ab ' * 512
        cut = 157  # Between a and b, inside the normal merged token ab.
        prefix = text[cut - 96:cut]
        continuation = canonical(text[cut:]).input_ids[:32]
        self.assertNotEqual(canonical(prefix + canonical.decode(continuation)).input_ids,
                            canonical(prefix).input_ids + continuation)
        row = dict(id='web-en/one/window-0', domain='web-en', source_document_id='one',
            document_sha256=hashlib.sha256(text.encode()).hexdigest(), source_window_start=0,
            source_window_end=512, absolute_cut_chars=cut, split='screen', cluster_id='web-en/one')
        sources = {d: [dict(id='one', text=text)] if d == 'web-en' else [] for d in DOMAINS}
        aligned, audit = align_panel([row], sources, canonical, {7: backend}, {7: {0}})
        self.assertEqual(aligned[0]['cut_shift_chars'], 1)
        self.assertEqual(aligned[0]['id'], row['id'])
        self.assertEqual(audit['changed_cuts'], 1)
        result = aligned[0]
        self.assertEqual(canonical(result['prefix'] + canonical.decode(result['canonical_continuation_ids'])).input_ids,
                         result['canonical_prefix_ids'] + result['canonical_continuation_ids'])

    def test_cluster_bootstrap_uses_equal_domain_weights(self):
        import numpy as np
        rows = [dict(domain='web-en', cluster_id='a', continuation_bytes=1),
                dict(domain='web-en', cluster_id='a', continuation_bytes=1),
                dict(domain='code', cluster_id='b', continuation_bytes=100)]
        means, draws = bootstrap_balanced(rows, np.array([[2, 2, 0]]) * math.log(2), 100)
        self.assertAlmostEqual(means[0], 1)
        self.assertTrue(np.allclose(draws, 1))

    def test_selector_is_chosen_on_discovery_not_confirmation(self):
        rows, base, no_prefix = [], [], []
        for domain in DOMAINS:
            for split in ['screen', 'confirm']:
                for document in range(2):
                    for window in range(2):
                        cluster = f'{domain}/{split}/{document}'
                        rows.append(dict(id=cluster + f'/{window}', cluster_id=cluster,
                            document_sha256=cluster, domain=domain, split=split,
                            continuation_bytes=32, tags={tag: True for tag in
                                ['structured', 'numeric_dense', 'repeated_number', 'plain_prose']} | {'image_markup': False}))
                        base.append([math.log(2)] * 32)
                        no_prefix.append([math.log(2) * (1.1 + document + window)] * 32)
        plan = dict(indices=[4, 5], representative_indices=[4, 5], bootstrap_replicates=50, limits=[])
        prep = dict(metadata=rows, plan_sha256=digest(plan), panel_sha256='frozen')
        results = []
        for index in plan['indices']:
            targets = []
            for row in rows:
                gap = .3 if row['split'] == 'screen' and index == 4 else .01 if index == 4 else .1
                targets.append([math.log(2) * (1 + gap)] * 32)
            results.append(dict(index=index, plan_sha256=digest(plan), panel_sha256='frozen',
                config=dict(algorithm='bpe', pretokenizer='regex', vocab_size=64000), transferred=targets))
        report = analyze(dict(plan=plan, preparation=prep, results=results, canonical=base, no_prefix=no_prefix))
        pick = next(r for r in report['selections'] if r['group'] == 'all')
        self.assertEqual(pick['screen_selected_index'], 4)
        self.assertAlmostEqual(pick['confirmation_selection_advantage_bpb'], -.045)
        self.assertTrue(all(abs(v + .045) < 1e-12 for v in pick['confirmation_advantage_ci95']))


if __name__ == '__main__':
    unittest.main()
