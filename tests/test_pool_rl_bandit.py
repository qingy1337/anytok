"""Reward direction, exploration, exact policy recovery and checkpoint integrity."""
import copy
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('bandit_sampling', ROOT/'runtime/anytok_pool_rl_20k/hf_hypernet/bandit_sampling.py')
bandit_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(bandit_module)
TokenizerBandit = bandit_module.TokenizerBandit


class PolicyTest(unittest.TestCase):
    def test_uniform_initial_policy_and_independent_rng(self):
        state = torch.get_rng_state().clone()
        policy = TokenizerBandit()
        np.testing.assert_array_equal(policy.probabilities(), np.full(1000, .001))
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        actions = [policy.sample() for _ in range(100)]
        self.assertGreater(len(set(actions)), 80)

    def test_higher_bpb_increases_sampling_probability_and_lower_bpb_decreases_it(self):
        for reward, sign in [(3.,1),(1.,-1)]:
            policy = TokenizerBandit(size=4, entropy_weight=0.)
            policy.update(0,2.)
            before = policy.probabilities()[1]
            result = policy.update(1,reward)
            self.assertGreater(sign*(policy.probabilities()[1]-before),0)
            self.assertEqual(result['policy_advantage_bpb'],reward-2.)

    def test_exact_next_actions_and_updates_after_serialized_restart(self):
        policy = TokenizerBandit(size=8)
        for i in range(20): policy.update(policy.sample(), 1.+(i%3)*.1)
        stream = io.BytesIO(); torch.save(policy.state_dict(),stream); stream.seek(0)
        resumed = TokenizerBandit(size=8)
        resumed.load_state_dict(torch.load(stream,weights_only=True))
        for i in range(30):
            action = policy.sample()
            self.assertEqual(resumed.sample(),action)
            self.assertEqual(policy.update(action,1.4+i*.01),resumed.update(action,1.4+i*.01))
            np.testing.assert_array_equal(policy.probabilities(),resumed.probabilities())

    def test_entropy_pushes_a_concentrated_policy_towards_exploration(self):
        policy=TokenizerBandit(size=4,entropy_weight=1.)
        with torch.no_grad(): policy.logits.copy_(torch.tensor([3.,0.,0.,0.]))
        before=policy.probabilities().max()
        policy.update(0,2.)  # zero reward advantage; entropy gradient alone
        self.assertLess(policy.probabilities().max(),before)

    def test_invalid_rewards_and_mismatched_checkpoint_are_rejected(self):
        policy=TokenizerBandit(size=4)
        for bad in [float('nan'),float('inf')]:
            with self.assertRaises(ValueError): policy.update(0,bad)
        other=TokenizerBandit(size=4,learning_rate=.1)
        with self.assertRaises(ValueError): other.load_state_dict(policy.state_dict())
        bad=copy.deepcopy(policy.state_dict());bad['updates']=1
        with self.assertRaises(ValueError): policy.load_state_dict(bad)

    def test_checkpoint_audit_requires_policy_and_full_restart_state(self):
        sys.path.insert(0,str(ROOT/'scripts'))
        import modal_anytok_pool_rl_20k as launcher
        import hashlib,json
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp);folder=run/'step-002000';folder.mkdir()
            files={}
            for name in launcher.WEIGHTS|{'optimizer.pt','state.pt'}:
                path=folder/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(name.encode())
                files[name]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            (folder/'manifest.json').write_text(json.dumps({'step':2000,'kind':'full_restart_checkpoint','files':files}))
            launcher.verify_checkpoint(run,2000,True)
            (folder/'policy.pt').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):launcher.verify_checkpoint(run,2000,True)


if __name__ == '__main__': unittest.main()
