"""Paired controls, disjoint documents, context caps and bootstrap statistics."""
import io
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from pool_generalization import copy_disjoint_documents,shared_prefix,summarize,load_plan


class GeneralizationTest(unittest.TestCase):
    def test_fresh_documents_start_after_the_last_touched_pool_record(self):
        records=[{'id':i,'text':text} for i,text in enumerate(['aaa','bbbbb','cccc','dddddd'])]
        lines=[json.dumps(r)+'\n' for r in records];out=io.StringIO()
        meta=copy_disjoint_documents(lines,out,skip_bytes=4,required_bytes=5)
        self.assertEqual(meta['skipped_bytes'],8)
        self.assertEqual(meta['skipped_documents'],2)
        self.assertEqual([json.loads(s)['id'] for s in out.getvalue().splitlines()],[2,3])
        self.assertGreaterEqual(meta['available_bytes'],5)
        with self.assertRaises(ValueError):copy_disjoint_documents(lines,io.StringIO(),100,5)

    def test_context_is_shared_and_fits_every_tokenizer(self):
        class Canonical:
            def __call__(self,text,**kwargs):return type('Result',(),{'input_ids':[ord(c) for c in text]})()
            def decode(self,ids,**kwargs):return ''.join(chr(i) for i in ids)
        class Target:
            def encode(self,text,**kwargs):return type('Result',(),{'ids':[ord(c) for c in text for _ in range(2)]})()
            def decode(self,ids,**kwargs):return ''.join(chr(i) for i in ids[::2])
        prefix,canonical,maximum=shared_prefix('word '*40,[1]*32,Canonical(),[Target()],cap=128)
        self.assertTrue(('word '*40).endswith(prefix))
        self.assertEqual(len(canonical),len(prefix))
        self.assertEqual(maximum,2*len(prefix))
        self.assertLessEqual(1+maximum+32,128)

    def test_plan_matches_knobs_but_changes_instance_and_has_no_selection_by_loss(self):
        plan=load_plan(ROOT/'plans/pool-generalization-v1.json')
        self.assertEqual(len(plan['pairs']),18)
        for pair in plan['pairs']:
            for key in ['algorithm','pretokenizer','vocab_size','noise_per_million','domain_bytes','bucket_bytes']:
                self.assertEqual(pair['seen'][key],pair['fresh'][key])
            self.assertNotEqual(pair['seen']['seed'],pair['fresh']['seed'])
            self.assertGreater(pair['seen_exposures']['6000'],0)

    def test_paired_bootstrap_detects_a_known_growing_penalty(self):
        plan={'steps':[2000,6000,12000],'documents':4,
              'pairs':[{'stratum_weight':1/9} for _ in range(18)]}
        byte_counts=[1,2,3,4];seen=np.tile(np.array(byte_counts)*math.log(2),(18,1))
        results={}
        for step,delta in [(2000,0),(6000,.01),(12000,.03)]:
            results[str(step)]={'bytes':byte_counts,'canonical_nats':np.array(byte_counts)*math.log(2),
                                'seen_nats':seen,'fresh_nats':seen+np.array(byte_counts)*math.log(2)*delta}
        stats=summarize(plan,results,replicates=100)
        self.assertAlmostEqual(stats['rows'][-1]['fresh_minus_seen_bpb'],.03)
        self.assertAlmostEqual(stats['gap_growth_6k_to_12k'],.02)
        self.assertGreater(stats['gap_growth_ci95'][0],0)
        results['12000']['bytes']=[1,2,3,5]
        with self.assertRaises(ValueError):summarize(plan,results,replicates=10)


if __name__=='__main__':unittest.main()
