import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from analyze_tokenizer_speed import analyze, verify_reference_texts, adopt_capacity_rescue


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

    def test_roundtrip_equivalence_cannot_qualify_a_different_learned_tokenizer(self):
        plan, rows = self.fixture()
        rows[-1]['manifest']['tokenizer_sha256'] = 'different-token-ids-or-scores'
        result = analyze(rows, plan)
        self.assertTrue(result['matched_complete'])
        self.assertEqual(result['total']['ready_speedup'], 10.)
        self.assertFalse(result['identical_tokenizers_complete'])
        self.assertFalse(result['qualified_at_least_5x'])
        self.assertFalse(result['qualified_at_least_10x'])

    def test_unmatched_preemption_timings_are_excluded_from_aggregate(self):
        plan,rows=self.fixture()
        rows[1]['container_id']='replacement-worker'
        result=analyze(rows,plan)
        self.assertTrue(result['complete'])
        self.assertFalse(result['matched_complete'])
        self.assertFalse(result['qualified_at_least_5x'])
        self.assertEqual(len(result['rows']),8)
        self.assertEqual(result['total']['baseline_seconds'],800.)
        self.assertEqual(result['unmatched_pairs'][0]['case'],rows[1]['case'])

    def test_capacity_rescue_preserves_method_and_cannot_replace_a_finished_pair(self):
        plan,rows=self.fixture()
        parent={'id':'parent','candidate':'v8','base_wheel_sha256':'base','fast_wheel_sha256':'fast',
                'worker_sha256':'worker','trainer_sha256':'trainer','launcher_sha256':'launcher','plan_sha256':'plan'}
        for row in rows:row['provenance']=copy.deepcopy(parent)
        rescue=copy.deepcopy(rows[:2])
        execution={**parent,'id':'rescue','parent_id':'parent','nonpreemptible':True,'rescue_launcher_sha256':'harness'}
        for row in rescue:
            row['provenance']=copy.deepcopy(execution);row['container_id']='replacement'
        selected=adopt_capacity_rescue(rows[1:],rescue,parent)
        self.assertEqual(len(selected),18)
        self.assertEqual(selected[-1]['execution_provenance'],execution)
        self.assertTrue(analyze(selected,plan)['qualified_at_least_10x'])
        with self.assertRaisesRegex(ValueError,'already matched'):adopt_capacity_rescue(rows,rescue,parent)
        for mutation in [lambda r:r['provenance'].__setitem__('fast_wheel_sha256','different'),
                         lambda r:r['provenance'].__setitem__('parent_id','different'),
                         lambda r:r['manifest']['config'].__setitem__('vocab_size',16000)]:
            bad=copy.deepcopy(rescue)
            for row in bad:mutation(row)
            with self.assertRaises(ValueError):adopt_capacity_rescue(rows[1:],bad,parent)
        bad=copy.deepcopy(selected);bad[-1]['execution_provenance']['fast_wheel_sha256']='different'
        with self.assertRaisesRegex(ValueError,'native benchmark protocol'):analyze(bad,plan)


if __name__ == '__main__': unittest.main()
