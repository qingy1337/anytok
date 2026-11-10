import copy
import math
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_coverage_evidence import generator_support,domain_effect,cv_predictability,DOMAINS


class CoverageEvidenceTest(unittest.TestCase):
    def test_finite_pool_and_structural_support_are_different_limits(self):
        recipe={'buckets':[{'name':n,'train_target_bytes':w} for n,w in
            [('web-en',8),('web-zh',1),('web-ja',1),('wiki-en',6),('wiki-zh',2),('wiki-ja',2)]
            +[(d+'-en',1) if d!='code' else ('code',1) for d in DOMAINS[2:]]]}
        names=[r['name'] for r in recipe['buckets']]
        base={n:0 for n in names};base.update({'web-en':80,'web-zh':10,'web-ja':10})
        medical={n:0 for n in names};medical['medical-en']=100
        plan={'recipe':recipe,'configs':[{'bucket_bytes':base,'vocab_size':32000},
            {'bucket_bytes':medical,'vocab_size':128000}],'total_bytes_per_tokenizer':100}
        focus={n:0 for n in names};focus.update({'web-en':20,'web-zh':5,'wiki-en':45,'wiki-zh':30})
        specialty={'baseline':{'bucket_bytes':base},'mixtures':[{'label':'Chinese35','bucket_bytes':focus}],
                   'total_train_bytes_per_mixture':100}
        report=generator_support(plan,specialty)
        self.assertAlmostEqual(report['theoretical_max_chinese_or_japanese'],.2)
        self.assertAlmostEqual(report['observed_max_chinese'],.1)
        self.assertAlmostEqual(report['specialty'][0]['scheme_minimum_mixture_tv'],0)
        self.assertAlmostEqual(report['specialty'][1]['scheme_minimum_mixture_tv'],.175)
        self.assertGreater(report['specialty'][1]['nearest_pool_mixture_tv'],.175)

    def test_domain_intervals_preserve_shared_articles_across_tokenizers(self):
        counts=[10,20];scale=math.log(2)
        row={'document_ids':['a','b'],'byte_counts':counts,'scores':{
            'uniform':{'transferred_nats':[12*scale,24*scale],'canonical_nats':[10*scale,20*scale]},
            'rl':{'transferred_nats':[10*scale,26*scale],'canonical_nats':[10*scale,20*scale]}}}
        a=domain_effect([row],replicates=500)
        b=domain_effect([row,copy.deepcopy(row)],replicates=500)
        np.testing.assert_allclose(a['delta_gap_ci95'],b['delta_gap_ci95'])
        self.assertEqual(a['source_documents'],2)
        self.assertEqual(b['cases'],2)

    def test_predictability_uses_held_out_observations(self):
        rng=np.random.default_rng(3);features=rng.normal(size=(500,2));y=features[:,0]*2+rng.normal(size=500)*.1
        folds=np.arange(500)%5
        good=cv_predictability(y,features,folds)
        shuffled=cv_predictability(rng.permutation(y),features,folds)
        self.assertGreater(good['r2'],.98)
        self.assertLess(shuffled['r2'],.1)


if __name__=='__main__':unittest.main()
