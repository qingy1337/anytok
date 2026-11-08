import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from analyze_tokenizer_speed import analyze, verify_reference_texts


class SpeedAnalysisTest(unittest.TestCase):
    def fixture(self):
        cases = []
        for algorithm in ['bpe','wordpiece','unigram']:
            for pretokenizer in ['regex','line256','no-regex']:
                config = {'index':len(cases),'algorithm':algorithm,'pretokenizer':pretokenizer,
                          'vocab_size':32000,'bucket_bytes':{'web':200000000}}
                cases.append({'key':algorithm+'-'+pretokenizer,'config':config})
        plan = {'cases':cases}
        source = {'web':{'quota_bytes':200000000,'used_bytes':200000000,'documents_sha256':'same'}}
        rows = []
        for case in cases:
          for variant, seconds in [('baseline-200mb',100.),('fast-200mb',10.)]:
            rows.append({'case':case['key'],'variant':variant,'provenance':{'id':'same'},'cpu_model':'same','container_id':'same-worker',
                         'ready_seconds':seconds,'subprocess_seconds':seconds+1,'extended_audit_seconds':1.,
                         'manifest':{'config':copy.deepcopy(case['config']),'vocab_size_actual':32000,'source_stats':copy.deepcopy(source),'total_used_bytes':200000000,
                                     'cpu_seconds':seconds*2,'peak_rss_gib':1.,'tokenizer_sha256':'same'},
                         'lossless_audit':{'roundtrip_failures':0,'all_256_byte_symbols_present':True,'normalizer':None,
                                           'post_processor':None,'decode_skip_special_tokens':False,'fixtures':300,
                                           'fixture_sha256':'same','fixture_tokens':1000,'finite_model_scores':True}})
        return plan, rows

    def test_incomplete_grid_cannot_qualify(self):
        plan, rows = self.fixture()
        self.assertTrue(analyze(rows,plan)['qualified_at_least_10x'])
        with self.assertRaisesRegex(ValueError,'Full'): analyze(rows[:1],plan)
        self.assertFalse(analyze(rows[:1],plan,require_complete=False)['qualified_at_least_5x'])

    def test_changed_data_vocabulary_and_lossy_results_are_rejected(self):
        plan, rows = self.fixture()
        for mutate in [lambda r:r['manifest'].__setitem__('vocab_size_actual',16000),
                       lambda r:r['manifest'].__setitem__('total_used_bytes',20000000),
                       lambda r:r['lossless_audit'].__setitem__('roundtrip_failures',1),
                       lambda r:r['manifest']['source_stats']['web'].__setitem__('documents_sha256','other')]:
            bad = copy.deepcopy(rows); mutate(bad[1])
            with self.assertRaises(ValueError): analyze(bad,plan)

    def test_missing_fidelity_or_matched_worker_prevents_qualification(self):
        plan, rows = self.fixture()
        for change in [lambda r:r.__setitem__('container_id','different-worker'),
                       lambda r:r.pop('container_id'),
                       lambda r:r['lossless_audit'].__setitem__('finite_model_scores',False),
                       lambda r:r['lossless_audit'].pop('finite_model_scores')]:
            bad = copy.deepcopy(rows); change(bad[1])
            result = analyze(bad,plan)
            self.assertTrue(result['complete'])
            self.assertFalse(result['qualified_at_least_5x'])

    def test_both_variants_cannot_silently_use_different_training_text(self):
        plan, rows = self.fixture()
        references = {r['case']:copy.deepcopy(r['manifest']['source_stats']) for r in rows}
        verify_reference_texts(rows,references)
        rows[0]['manifest']['source_stats']['web']['documents_sha256'] = 'new-data'
        rows[1]['manifest']['source_stats']['web']['documents_sha256'] = 'new-data'
        with self.assertRaisesRegex(ValueError,'original saved pool'):
            verify_reference_texts(rows,references)


if __name__ == '__main__': unittest.main()
