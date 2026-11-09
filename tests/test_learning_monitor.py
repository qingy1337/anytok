"""Verify live history integrity and selection probability summaries."""
import json
from pathlib import Path
import tempfile
import unittest

from monitoring.learning import LearningReader, sampled
from monitoring.policy_summary import summarize


class LearningMonitorTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = {'logs': 'rl-*.log', 'baseline_logs': 'uniform-*.log',
                       'plan': 'plan.json', 'snapshots': 'policy.json'}
        self.initial = [{'bytes': 100, 'examples': 2, 'truncated_forms': 0,
                         'generated_bpb': 1.5, 'canonical_bpb': 1.0}] * 2
        self.write('plan.json', {'configs': [{'index': 0, 'algorithm': 'bpe', 'pretokenizer': 'regex'}]})
        self.append('rl-a.log', {'stage': 'initial_eval', 'results': self.initial})
        self.append('uniform-a.log', {'stage': 'initial_eval', 'results': self.initial})
        self.reader = LearningReader(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def append(self, name, value):
        with (self.root / name).open('a') as stream:
            stream.write('2026-11-09 ' + json.dumps(value) + '\n')

    def test_scores_use_own_canonical_and_limit_uniform_to_current_training(self):
        self.append('rl-a.log', {'stage': 'eval', 'step': 500, 'results': [
            {**r, 'generated_bpb': 1.2, 'canonical_bpb': 1.05} for r in self.initial]})
        self.append('uniform-a.log', {'stage': 'eval', 'step': 500, 'results': [
            {**r, 'generated_bpb': 1.1, 'canonical_bpb': 1.01} for r in self.initial]})
        self.append('uniform-a.log', {'stage': 'eval', 'step': 1000, 'results': self.initial})
        data = self.reader.read(self.config)
        self.assertAlmostEqual(data['curves']['rl'][-1]['tokenizers'][0]['gap'], .15)
        self.assertAlmostEqual(data['curves']['uniform'][-1]['tokenizers'][0]['gap'], .09)
        self.assertEqual([r['step'] for r in data['curves']['uniform']], [0, 500])

    def test_incremental_partial_writes_and_repeated_evals_are_not_double_counted(self):
        self.reader.read(self.config)
        path = self.root / 'rl-a.log'
        with path.open('a') as stream:
            stream.write('{"stage":"train","step":1')
        self.assertEqual(self.reader.read(self.config)['latest_step'], 0)
        with path.open('a') as stream:
            stream.write(',"tokenizer_config_index":0}\n')
        data = self.reader.read(self.config)
        self.assertEqual(data['latest_step'], 1)
        self.assertEqual(data['recent_exposure']['algorithm'], {'bpe': 1.0})
        self.append('rl-b.log', {'stage': 'initial_eval', 'results': self.initial})
        self.assertEqual(len(self.reader.read(self.config)['curves']['rl']), 1)
        path.write_text('')
        self.assertEqual(self.reader.read(self.config)['latest_step'], 0)

    def test_mismatched_panel_and_baseline_initialization_are_hidden(self):
        self.append('rl-a.log', {'stage': 'eval', 'step': 500, 'results': [
            {**r, 'bytes': 101} for r in self.initial]})
        self.append('uniform-a.log', {'stage': 'initial_eval', 'results': [
            {**r, 'generated_bpb': 1.6} for r in self.initial]})
        data = self.reader.read(self.config)
        self.assertEqual([p['step'] for p in data['curves']['rl']], [0])
        self.assertEqual(data['curves']['uniform'], [])
        self.assertEqual(len(data['warnings']), 2)

    def test_snapshot_partial_write_keeps_last_valid_distribution(self):
        self.write('policy.json', {'snapshots': [{'step': 500}], 'checked_at': 'now'})
        self.assertEqual(self.reader.read(self.config)['snapshots'], [{'step': 500}])
        (self.root / 'policy.json').write_text('{')
        self.assertEqual(self.reader.read(self.config)['snapshots'], [{'step': 500}])

    def test_policy_groups_distinguish_probability_from_realized_exposure(self):
        configs = [
            {'index': 0, 'algorithm': 'bpe', 'pretokenizer': 'regex', 'vocab_size': 32000,
             'noise_per_million': 0, 'domain_bytes': {'web': 90, 'code': 10}},
            {'index': 1, 'algorithm': 'unigram', 'pretokenizer': 'line256', 'vocab_size': 128000,
             'noise_per_million': None, 'domain_bytes': {'web': 10, 'code': 90}}]
        result = summarize(configs, [.8, .2], [1, 3], 4)
        self.assertEqual(result['groups']['algorithm']['bpe'], {'policy': .8, 'uniform': .5, 'observed': .25})
        self.assertAlmostEqual(result['groups']['domain']['web']['policy'], .74)
        self.assertAlmostEqual(result['groups']['domain']['web']['observed'], .3)
        self.assertEqual(result['groups']['merge_noise']['0']['policy'], 1.)
        self.assertEqual(result['top_tokenizers'][0]['index'], 0)
        with self.assertRaises(ValueError):
            summarize(configs, [.8, .3], [1, 3], 4)
        with self.assertRaises(ValueError):
            summarize(configs, [.8, .2], [1, 3], 5)

    def test_downsampling_retains_endpoints(self):
        points = list(range(20000))
        result = sampled(points)
        self.assertEqual(len(result), 400)
        self.assertEqual((result[0], result[-1]), (0, 19999))


if __name__ == '__main__':
    unittest.main()
