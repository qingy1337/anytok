"""Reward direction, exploration, exact policy recovery and checkpoint integrity."""
import copy
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

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

    def test_observer_transport_failures_retry_the_same_call(self):
        sys.path.insert(0, str(ROOT/'scripts'))
        import modal
        import modal_anytok_pool_rl_20k as launcher
        call = Mock(object_id='fc-existing')
        call.get.side_effect = [modal.exception.ConnectionError('Deadline exceeded'),
                                ConnectionError('temporary disconnect'), TimeoutError(), {'step': 9326}]
        with patch.object(launcher.time, 'sleep') as sleep, patch('builtins.print'):
            self.assertEqual(launcher.await_call_result(call), {'step': 9326})
        self.assertEqual(call.get.call_count, 4)
        self.assertEqual(sleep.call_args_list, [unittest.mock.call(2), unittest.mock.call(4)])
        call.spawn.assert_not_called()
        call.get.side_effect = ValueError('checkpoint mismatch')
        with self.assertRaises(ValueError):
            launcher.await_call_result(call)

    def test_copy_retries_truncated_streams_without_publishing_bad_bytes(self):
        sys.path.insert(0,str(ROOT/'scripts'))
        import hashlib
        import modal_anytok_pool_rl_20k as launcher
        expected={'bytes':4,'sha256':hashlib.sha256(b'full').hexdigest()}
        source=Mock();source.read_file.side_effect=[[b'ful'],[b'fu',b'll']]
        with tempfile.TemporaryDirectory() as tmp,patch.object(launcher.time,'sleep'),patch('builtins.print'):
            target=Path(tmp)/'weights'
            self.assertEqual(launcher.verified_copy(source,'cloud/weights',target,expected),expected)
            self.assertEqual(target.read_bytes(),b'full')
            self.assertFalse(list(Path(tmp).glob('*.partial')))
            source.read_file.side_effect=None;source.read_file.return_value=[b'bad']
            other=Path(tmp)/'bad'
            with self.assertRaises(ValueError):launcher.verified_copy(source,'cloud/bad',other,expected)
            self.assertFalse(other.exists())

    def test_latest_restart_excludes_archived_and_incomplete_states(self):
        sys.path.insert(0,str(ROOT/'scripts'))
        import run_anytok_pool_rl_20k as controller
        prefix=controller.RUN_PREFIX
        paths=[prefix+'/step-002000/state.pt',prefix+'/step-002000/manifest.json',
               prefix+'/step-003000/state.pt',prefix+'/interrupted/old/step-004000/state.pt',
               prefix+'/interrupted/old/step-004000/manifest.json',
               'run/step-020000/state.pt','run/step-020000/manifest.json']
        entries=[type('Entry',(),{'path':p})() for p in paths]
        with patch.object(controller,'volume_for') as get:
            get.return_value.iterdir.return_value=entries
            self.assertEqual(controller.latest('trenton'),2000)

    def test_recovery_reattaches_to_pending_preparation_call(self):
        sys.path.insert(0,str(ROOT/'scripts'))
        import run_anytok_pool_rl_20k as controller
        import json
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'status.json').write_text(json.dumps({'stage':'preparing_inputs','profile':'trenton','step':0}))
            (root/'trenton-prepare-call.json').write_text(json.dumps({'call_id':'fc-existing'}))
            with patch.object(controller,'STATE',root),patch.object(controller,'active_apps',return_value=['ap-existing']),\
                 patch.object(controller,'call',return_value={'files':1061}) as call,\
                 patch.object(controller,'billing_row',return_value={'status':'ok','estimated_remaining_usd':'0'}):
                controller.run_locked()
                call.assert_called_once_with('trenton','prepare',resume_call='fc-existing')

    def test_final_stretch_reuses_source_credit_with_original_reserve(self):
        sys.path.insert(0,str(ROOT/'scripts'))
        import run_anytok_pool_rl_20k as controller
        import json
        class Started(Exception):pass
        def run_call(profile, action, extra=(), **kwargs):
            self.assertEqual(profile,'neural-net-1411')
            if action=='train':
                self.assertEqual(extra,['--max-seconds',2609])
                raise Started()
            self.assertEqual(action,'prepare')
            return {'files':1061}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'status.json').write_text(json.dumps({
                'stage':'failed','profile':'queues','step':19259,'resume_step':19259,
                'resume_source':'neural-net-1411','smoke_verified':True}))
            with patch.object(controller,'STATE',root),patch.object(controller,'active_apps',return_value=[]),\
                 patch.object(controller,'latest',side_effect=lambda p:19259 if p=='neural-net-1411' else 0),\
                 patch.object(controller,'billing_row',return_value={'status':'ok','estimated_remaining_usd':'7.98'}),\
                 patch.object(controller,'call',side_effect=run_call) as call:
                with self.assertRaises(Started):controller.run_locked()
                self.assertEqual([c.args[:2] for c in call.call_args_list],
                                 [('neural-net-1411','prepare'),('neural-net-1411','train')])


if __name__ == '__main__': unittest.main()
