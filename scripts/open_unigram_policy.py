"""Proposal sampling and selectors shared by frozen and joint experiments."""

import math

import numpy as np
import torch


POLICIES = ('random', 'reinforce', 'predictor')
SPLITS = ('regex', 'line256', 'no-regex')
MENU_SIZE = 64


def mixture(rng, names):
    support = (rng.choice(names, size=int(rng.integers(1, len(names) + 1)), replace=False)
               if rng.random() < .25 else names)
    return {str(name): float(weight) for name, weight in zip(support, rng.dirichlet(np.ones(len(support))))}


def propose(rng, generator):
    from anytok_online import DOMAINS, allocate

    actions, features = [], []
    rejected_quotas = 0
    while len(actions) < MENU_SIZE:
        domains = mixture(rng, tuple(DOMAINS))
        by_language = {d: mixture(rng, tuple(b.split('-')[1] for b in DOMAINS[d]))
                       for d in ('web', 'wikipedia')}
        domain_bytes = allocate(generator.corpus_bytes, [domains.get(d, 0.) for d in DOMAINS])
        quotas = {}
        for domain, amount in zip(DOMAINS, domain_bytes, strict=True):
            buckets = DOMAINS[domain]
            amounts = (allocate(amount, [by_language[domain].get(b.split('-')[1], 0.) for b in buckets])
                       if domain in by_language else [amount])
            quotas.update(zip(buckets, amounts, strict=True))
        if any(amount > generator.capacity[b] for b, amount in quotas.items()):
            rejected_quotas += 1
            continue
        size = int(rng.integers(32000, 128001))
        split = str(rng.choice(SPLITS))
        actions.append(dict(mixture=domains, language_mix=by_language, vocab_size=size, pretokenizer=split))
        features.append([float(split == mode) for mode in SPLITS] + [math.log2(size / 32000) / 2]
                        + [total / generator.corpus_bytes for total in domain_bytes]
                        + [quotas[b] / generator.corpus_bytes for b in generator.names
                           if b.startswith(('web-', 'wiki-'))])
    return actions, torch.tensor(features, dtype=torch.float64), rejected_quotas


class Selector:
    def __init__(self, name, width, seed):
        self.name = name
        self.rng = np.random.default_rng(seed)
        self.baseline = None
        self.network = self.optimizer = None
        if name != 'random':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(43)
                self.network = torch.nn.Sequential(
                    torch.nn.Linear(width, 64, dtype=torch.float64), torch.nn.Tanh(),
                    torch.nn.Linear(64, 64, dtype=torch.float64), torch.nn.Tanh(),
                    torch.nn.Linear(64, 1, dtype=torch.float64),
                )
                torch.nn.init.zeros_(self.network[-1].weight)
                torch.nn.init.zeros_(self.network[-1].bias)
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=.001)

    def choose(self, features, valid, step):
        if self.name == 'random':
            return int(self.rng.choice(valid)), dict(mode='random')
        with torch.no_grad():
            values = self.network(features[valid]).squeeze(-1).numpy()
        if self.name == 'predictor':
            explore = (step - 1) % 4 == 0
            indices = np.arange(len(valid)) if explore else np.flatnonzero(values == values.max())
            index = int(self.rng.choice(indices))
            return valid[index], dict(mode='explore' if explore else 'exploit', predicted_bpb=float(values[index]))
        probs = np.exp(values - values.max())
        probs /= probs.sum()
        index = int(self.rng.choice(len(valid), p=probs))
        entropy = -float(np.sum(probs * np.log(np.maximum(probs, np.finfo(float).tiny))))
        return valid[index], dict(mode='sample', selected_probability=float(probs[index]),
                                  menu_max_probability=float(probs.max()), menu_entropy=entropy)

    def observe(self, features, valid, index, raw_bpb):
        if self.name == 'random':
            return
        values = self.network(features[valid]).squeeze(-1)
        chosen = valid.index(index)
        if self.name == 'predictor':
            loss = (values[chosen] - raw_bpb).square()
        else:
            advantage = 0. if self.baseline is None else raw_bpb - self.baseline
            loss = -advantage * values.log_softmax(0)[chosen]
            self.baseline = raw_bpb if self.baseline is None else .99 * self.baseline + .01 * raw_bpb
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not torch.isfinite(loss) or any(not p.grad.isfinite().all() for p in self.network.parameters()):
            raise ValueError(f'Nonfinite selector update for {self.name}')
        self.optimizer.step()

    def state_dict(self):
        return dict(rng=self.rng.bit_generator.state, baseline=self.baseline,
                    network=None if self.network is None else self.network.state_dict(),
                    optimizer=None if self.optimizer is None else self.optimizer.state_dict())

    def restore(self, state):
        self.rng.bit_generator.state = state['rng']
        self.baseline = state['baseline']
        if self.network is not None:
            self.network.load_state_dict(state['network'])
            self.optimizer.load_state_dict(state['optimizer'])
