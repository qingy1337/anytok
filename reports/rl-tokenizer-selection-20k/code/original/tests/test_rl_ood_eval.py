import copy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pool_generalization import digest
from rl_ood_eval import compare, validate_result


class RLComparisonTest(unittest.TestCase):
    def setUp(self):
        case={'key':'test','document_ids':['a1','a2','b'], 'cluster_ids':['a','a','b'],
              'byte_counts':[10,20,30], 'context_sha256':'context','domain':'web-zh',
              'candidate':{'kind':'saved-native','sha256':'tokenizer'},
              'utility':{'token_reduction_vs_canonical_percent':50}}
        self.spec={'cases':[case]};self.spec['spec_sha256']=digest(self.spec)
    def result(self, canonical, gap):
        counts=self.spec['cases'][0]['byte_counts']
        row={'canonical_nats':[canonical*c*math.log(2) for c in counts],
             'transferred_nats':[(canonical+gap)*c*math.log(2) for c in counts],
             'context_sha256':'context','score':{'canonical_bpb':canonical,'bpb':canonical+gap,
                'gap_bpb':gap,'continuation_tokens':96,'continuation_bytes':60}}
        return {'step':20000,'spec_sha256':self.spec['spec_sha256'],'cases':{'test':row}}
    def test_separates_lm_change_from_transfer_change_with_paired_intervals(self):
        report=compare(self.spec,self.result(1.,.2),self.result(1.1,.1))
        scores=report['cases'][0]['scores']
        self.assertAlmostEqual(scores['transferred']['delta_rl_minus_uniform'],0)
        self.assertAlmostEqual(scores['canonical']['delta_rl_minus_uniform'],.1)
        self.assertAlmostEqual(scores['gap']['delta_rl_minus_uniform'],-.1)
        for value in scores['gap']['delta_ci95']:self.assertAlmostEqual(value,-.1)
        self.assertEqual(report['cases'][0]['group'],'ood_weak')
    def test_rejects_changed_context_counts_and_scores(self):
        for mutate in [lambda r:r.update(step=19500),
                       lambda r:r['cases']['test'].update(context_sha256='different'),
                       lambda r:r['cases']['test']['score'].update(continuation_tokens=97),
                       lambda r:r['cases']['test']['canonical_nats'].__setitem__(0,float('nan')),
                       lambda r:r['cases']['test']['score'].update(bpb=4.)]:
            r=self.result(1.,.2);mutate(r)
            with self.assertRaises(ValueError):validate_result(self.spec,r)
    def test_rejects_same_byte_total_with_reordered_document_ids(self):
        r=self.result(1.,.2)
        r['cases']['test']['input_audit']={'document_ids':['b','a2','a1'],'byte_counts':[10,20,30],
            'tokenizer_sha256':'tokenizer','suffix_tokens':32,'position_cap':128}
        with self.assertRaises(ValueError):validate_result(self.spec,r)
    def test_classification_keeps_in_pool_cases_and_controls_distinct(self):
        self.spec['cases'][0]['candidate']['kind']='pool'
        self.spec['spec_sha256']=digest({k:v for k,v in self.spec.items() if k!='spec_sha256'})
        self.assertEqual(compare(self.spec,self.result(1.,.2),self.result(1.,.1))['cases'][0]['group'],'in_pool_weak')
        self.assertEqual(compare(self.spec,self.result(1.,.05),self.result(1.,.1))['cases'][0]['group'],'control')
    def test_rejects_changed_specification(self):
        self.spec['cases'][0]['document_ids'].reverse()
        with self.assertRaises(ValueError):validate_result(self.spec,self.result(1.,.2))


if __name__=='__main__':unittest.main()
