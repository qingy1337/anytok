"""Generator features, cross-tokenizer sharing, RNG isolation and exact recovery."""
import copy
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('settings_policy',ROOT/'runtime/anytok_pool_settings_rl_20k/hf_hypernet/settings_policy.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
Policy=module.SettingsTokenizerBandit


def configs():
    rows=[]
    for i,algorithm in enumerate(module.ALGORITHMS):
        rows.append({'index':i,'seed':i,'algorithm':algorithm,'pretokenizer':'regex',
                     'vocab_size':32000+32000*i,'noise_per_million':300000 if algorithm=='bpe' else None,
                     'domain_bytes':{d:200000000 if d==module.DOMAINS[i] else 0 for d in module.DOMAINS}})
    return rows


class SettingsPolicyTest(unittest.TestCase):
    def test_features_cover_all_knobs_and_ignore_ids_seeds_and_metadata(self):
        original=configs()[0]
        x=module.settings_features([original])
        self.assertEqual(x.shape,(1,18))
        for key,value in [('index',999),('seed',11111),('config_sha256','different')]:
            changed={**original,key:value}
            np.testing.assert_array_equal(module.settings_features([changed]),x)
        variants=[{**original,'algorithm':'unigram','noise_per_million':None},
                  {**original,'pretokenizer':'no-regex'}, {**original,'vocab_size':128000},
                  {**original,'noise_per_million':750000},
                  {**original,'domain_bytes':{d:200000000 if d=='medical' else 0 for d in module.DOMAINS}}]
        for variant in variants:self.assertFalse(np.array_equal(module.settings_features([variant]),x))
        self.assertAlmostEqual(x[0,-9:].sum(),1.)

    def test_uniform_start_and_no_change_to_model_rng_or_cuda_seed(self):
        state=torch.get_rng_state().clone()
        with patch.object(torch.cuda,'manual_seed_all') as seed_cuda:
            policy=Policy(configs())
        seed_cuda.assert_not_called()
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        np.testing.assert_array_equal(policy.probabilities(),np.full(3,1/3))
        self.assertEqual(sum(p.numel() for p in policy.network.parameters()),5441)
        larger=Policy(configs()*10)
        self.assertEqual(sum(p.numel() for p in larger.network.parameters()),5441)

    def test_reward_shares_across_identical_settings_without_id_parameters(self):
        rows=configs();rows.insert(1,{**rows[0],'index':888,'seed':999})
        policy=Policy(rows,entropy_weight=0.)
        policy.update(3,2.)
        before=policy.probabilities()
        policy.update(0,3.)
        after=policy.probabilities()
        self.assertGreater(after[0],before[0])
        self.assertGreater(after[1],before[1])  # unsampled matching settings
        self.assertEqual(after[0],after[1])
        self.assertEqual(int(policy.counts[1]),0)
        query=[{**rows[0],'index':12345,'seed':555}]
        # BLAS reduction order can differ between one-row and batched scoring.
        self.assertAlmostEqual(float(policy.logits_for(query).detach()[0]),float(policy.logits.detach()[0]),places=14)

    def test_negative_advantage_reduces_probability(self):
        policy=Policy(configs(),entropy_weight=0.)
        policy.update(1,2.);before=policy.probabilities()[0]
        policy.update(0,1.)
        self.assertLess(policy.probabilities()[0],before)

    def test_serialized_network_optimizer_and_rng_resume_exactly(self):
        policy=Policy(configs())
        for i in range(10):policy.update(policy.sample(),1.+(i%3)*.2)
        stream=io.BytesIO();torch.save(policy.state_dict(),stream);stream.seek(0)
        restored=Policy(configs());restored.load_state_dict(torch.load(stream,weights_only=True))
        for i in range(12):
            action=policy.sample();self.assertEqual(restored.sample(),action)
            self.assertEqual(policy.update(action,1.1+i*.01),restored.update(action,1.1+i*.01))
            np.testing.assert_array_equal(policy.probabilities(),restored.probabilities())

    def test_feature_changes_or_corrupt_network_fail_resume(self):
        policy=Policy(configs());state=copy.deepcopy(policy.state_dict())
        changed=configs();changed[0]['vocab_size']=128000
        with self.assertRaisesRegex(ValueError,'features'):Policy(changed).load_state_dict(state)
        key=next(iter(state['network']));state['network'][key].flatten()[0]=float('nan')
        with self.assertRaisesRegex(ValueError,'Invalid'):policy.load_state_dict(state)


if __name__=='__main__':unittest.main()
