"""Sample tokenizer settings directly from a trainable distribution."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from anytok_online import DOMAINS


SPLITS = ('regex', 'line256', 'no-regex')
MIN_VOCAB, MAX_VOCAB = 32000, 128000
DTYPE = torch.float64
# Keep concentrations positive, with a floor for stable draws at simplex edges.
MIN_CONCENTRATION = .05
INITIAL_SHAPE = math.log(math.expm1(1. - MIN_CONCENTRATION))


def categorical(rng, logits):
    log_probs = logits.log_softmax(0)
    index = int(rng.choice(len(log_probs), p=log_probs.detach().exp().numpy()))
    return index, log_probs[index]


class MixturePolicy(nn.Module):
    """Choose which sources receive bytes, then draw their relative shares."""

    def __init__(self, names):
        super().__init__()
        self.names = tuple(names)
        count = len(self.names)
        masks = torch.arange(1, 2**count)[:, None].bitwise_and(2 ** torch.arange(count)) != 0
        sizes = masks.sum(1)
        base = [.25 / (count * math.comb(count, size)) for size in range(1, count + 1)]
        base[-1] += .75
        self.register_buffer('masks', masks, persistent=False)
        self.register_buffer('memberships', masks.to(DTYPE), persistent=False)
        self.register_buffer('size_index', sizes - 1, persistent=False)
        self.register_buffer('initial_logits', torch.tensor(base, dtype=DTYPE).log()[sizes - 1], persistent=False)
        self.source_preferences = nn.Parameter(torch.zeros(count, dtype=DTYPE))
        self.size_preferences = nn.Parameter(torch.zeros(count, dtype=DTYPE))
        self.relative_share_shapes = nn.Parameter(torch.zeros(count, dtype=DTYPE))

    def draw(self, rng):
        logits = (self.initial_logits + self.size_preferences[self.size_index]
                  + self.memberships @ self.source_preferences)
        index, log_prob = categorical(rng, logits)
        indices = self.masks[index].nonzero().flatten()
        if len(indices) == 1:
            return {self.names[int(indices[0])]: 1.}, log_prob
        concentration = MIN_CONCENTRATION + F.softplus(self.relative_share_shapes[indices] + INITIAL_SHAPE)
        weights = rng.dirichlet(concentration.detach().numpy())
        draw = torch.from_numpy(weights)
        log_prob = log_prob + torch.distributions.Dirichlet(concentration).log_prob(draw)
        return {self.names[i]: float(weight) for i, weight in zip(indices.tolist(), weights, strict=True)}, log_prob


class TokenizerPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.domains = MixturePolicy(DOMAINS)
        self.web = MixturePolicy(bucket.split('-')[1] for bucket in DOMAINS['web'])
        self.wikipedia = MixturePolicy(bucket.split('-')[1] for bucket in DOMAINS['wikipedia'])
        sizes = torch.linspace(-1., 1., MAX_VOCAB - MIN_VOCAB + 1, dtype=DTYPE)
        self.register_buffer('vocab_features', torch.stack((sizes, sizes.square()), dim=1), persistent=False)
        self.vocab_preferences = nn.Parameter(torch.zeros(2, dtype=DTYPE))
        self.split_logits = nn.Parameter(torch.zeros(len(SPLITS), dtype=DTYPE))

    def draw(self, rng):
        mixture, log_prob = self.domains.draw(rng)
        language_mix = {}
        for domain, policy in (('web', self.web), ('wikipedia', self.wikipedia)):
            if domain in mixture:
                language_mix[domain], language_log_prob = policy.draw(rng)
                log_prob = log_prob + language_log_prob
        vocab, vocab_log_prob = categorical(rng, self.vocab_features @ self.vocab_preferences)
        split, split_log_prob = categorical(rng, self.split_logits)
        return dict(mixture=mixture, language_mix=language_mix, vocab_size=MIN_VOCAB + vocab,
                    pretokenizer=SPLITS[split]), log_prob + vocab_log_prob + split_log_prob
