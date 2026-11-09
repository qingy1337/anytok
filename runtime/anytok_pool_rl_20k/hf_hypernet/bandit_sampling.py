"""Adversarial REINFORCE over immutable pool entries, with resumable exploration."""
import math

import numpy as np
import torch


class TokenizerBandit:
    def __init__(self, size=1000, seed=43, learning_rate=0.01,
                 entropy_weight=0.01, baseline_decay=0.99):
        if size < 2 or learning_rate <= 0 or entropy_weight < 0 or not 0 <= baseline_decay < 1:
            raise ValueError('Invalid bandit settings')
        self.settings = dict(size=size, seed=seed, learning_rate=learning_rate,
                             entropy_weight=entropy_weight, baseline_decay=baseline_decay)
        # A separate CPU RNG/optimizer cannot consume LM dropout or corpus RNG.
        self.logits = torch.nn.Parameter(torch.zeros(size, dtype=torch.float64))
        self.optimizer = torch.optim.Adam([self.logits], lr=learning_rate)
        self.rng = np.random.default_rng(seed)
        self.baseline = None
        self.counts = torch.zeros(size, dtype=torch.int64)
        self.updates = 0

    def probabilities(self):
        return self.logits.detach().softmax(0).numpy().copy()

    def sample(self):
        return int(self.rng.choice(self.settings['size'], p=self.probabilities()))

    def update(self, index, reward_bpb):
        reward_bpb = float(reward_bpb)
        if not math.isfinite(reward_bpb) or not 0 <= index < self.settings['size']:
            raise ValueError('Invalid action/reward')
        advantage = 0. if self.baseline is None else reward_bpb - self.baseline
        logp = self.logits.log_softmax(0)
        entropy = -(logp.exp() * logp).sum()
        # Maximize BPB plus entropy. The LM separately minimizes its usual loss.
        objective = advantage * logp[index] + self.settings['entropy_weight'] * entropy
        self.optimizer.zero_grad(set_to_none=True)
        (-objective).backward()
        if not torch.isfinite(self.logits.grad).all():
            raise FloatingPointError('Non-finite policy gradient')
        self.optimizer.step()
        with torch.no_grad():
            self.logits.sub_(self.logits.mean())
        decay = self.settings['baseline_decay']
        self.baseline = reward_bpb if self.baseline is None else decay*self.baseline + (1-decay)*reward_bpb
        self.counts[index] += 1
        self.updates += 1
        p = self.probabilities()
        return {'policy_reward_bpb': reward_bpb, 'policy_advantage_bpb': advantage,
                'policy_baseline_bpb': self.baseline,
                'policy_entropy': float(-(p*np.log(np.maximum(p, np.finfo(float).tiny))).sum()),
                'policy_effective_tokenizers': float(math.exp(-(p*np.log(np.maximum(p, np.finfo(float).tiny))).sum())),
                'policy_max_probability': float(p.max()), 'policy_selected_probability': float(p[index]),
                'policy_unique_tokenizers': int((self.counts > 0).sum()), 'policy_updates': self.updates}

    def state_dict(self):
        return {'settings': self.settings, 'logits': self.logits.detach().clone(),
                'optimizer': self.optimizer.state_dict(), 'rng': self.rng.bit_generator.state,
                'baseline': self.baseline, 'counts': self.counts.clone(), 'updates': self.updates}

    def load_state_dict(self, state):
        if state['settings'] != self.settings:
            raise ValueError('Policy settings changed on resume')
        if (tuple(state['logits'].shape) != tuple(self.logits.shape)
                or not torch.isfinite(state['logits']).all()
                or tuple(state['counts'].shape) != tuple(self.counts.shape)
                or int(state['counts'].sum()) != state['updates']):
            raise ValueError('Invalid saved policy')
        with torch.no_grad():
            self.logits.copy_(state['logits'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.rng.bit_generator.state = state['rng']
        self.baseline = state['baseline']
        self.counts.copy_(state['counts'])
        self.updates = state['updates']
