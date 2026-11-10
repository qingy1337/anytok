"""A shared settings-to-logit network; no per-tokenizer trainable parameters."""
import hashlib
import json
import math

import numpy as np
import torch

ALGORITHMS = ('bpe', 'wordpiece', 'unigram')
PRETOKENIZERS = ('regex', 'line256', 'no-regex')
DOMAINS = ('web', 'wikipedia', 'code', 'papers', 'math', 'qa', 'books', 'medical', 'legal')
FEATURE_NAMES = ([f'algorithm:{a}' for a in ALGORITHMS]
                 + [f'pretokenizer:{p}' for p in PRETOKENIZERS]
                 + ['vocab_log2_32k_to_128k', 'merge_noise_applicable', 'merge_noise_fraction_of_0.75']
                 + [f'domain_fraction:{d}' for d in DOMAINS])


def settings_features(configs):
    """Fixed scales from the generator; IDs, seeds and hashes are not inputs."""
    rows = []
    for config in configs:
        algorithm, pretokenizer = config['algorithm'], config['pretokenizer']
        if algorithm not in ALGORITHMS or pretokenizer not in PRETOKENIZERS:
            raise ValueError('Unknown algorithm or pretokenizer')
        size = float(config['vocab_size'])
        amounts = config['domain_bytes']
        if set(amounts) != set(DOMAINS): raise ValueError('Expected all mixture domains')
        weights = np.array([amounts[d] for d in DOMAINS], dtype=np.float64)
        if (not math.isfinite(size) or size <= 0 or not np.isfinite(weights).all()
                or (weights < 0).any() or weights.sum() <= 0):
            raise ValueError('Invalid vocabulary or domain mixture')
        noise = config['noise_per_million']
        if algorithm == 'bpe':
            if noise is None or not math.isfinite(float(noise)) or not 0 <= noise <= 750000:
                raise ValueError('Invalid BPE merge noise')
            noise_feature = float(noise)/750000.
        else:
            if noise is not None: raise ValueError('Merge noise applies only to BPE')
            noise_feature = 0.
        rows.append([float(algorithm == a) for a in ALGORITHMS]
                    + [float(pretokenizer == p) for p in PRETOKENIZERS]
                    + [math.log2(size/32000.)/2., float(algorithm == 'bpe'), noise_feature]
                    + (weights/weights.sum()).tolist())
    result = np.asarray(rows, dtype=np.float64)
    if result.shape != (len(configs), len(FEATURE_NAMES)) or not len(configs):
        raise ValueError('Nonempty settings matrix required')
    return result


class SettingsTokenizerBandit:
    def __init__(self, configs, seed=43, learning_rate=0.001,
                 entropy_weight=0.01, baseline_decay=0.99):
        if len(configs) < 2 or learning_rate <= 0 or entropy_weight < 0 or not 0 <= baseline_decay < 1:
            raise ValueError('Invalid policy settings')
        features = settings_features(configs)
        feature_inputs = [{key:config[key] for key in
                           ['algorithm','pretokenizer','vocab_size','noise_per_million','domain_bytes']}
                          for config in configs]
        feature_sha = hashlib.sha256(json.dumps({'schema':FEATURE_NAMES,'settings':feature_inputs},
                                     sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.features = torch.from_numpy(features)
        self.settings = {'kind':'shared_settings_mlp', 'size':len(configs), 'seed':seed,
                         'hidden_sizes':[64,64], 'learning_rate':learning_rate,
                         'entropy_weight':entropy_weight, 'baseline_decay':baseline_decay,
                         'feature_names':FEATURE_NAMES, 'feature_sha256':feature_sha}
        # Constructors temporarily use the CPU global RNG. Restore it, then
        # initialize using a private generator. Never reseed CUDA/model dropout.
        with torch.random.fork_rng(devices=[]):
            self.network = torch.nn.Sequential(
                torch.nn.Linear(len(FEATURE_NAMES),64,device='cpu',dtype=torch.float64), torch.nn.Tanh(),
                torch.nn.Linear(64,64,device='cpu',dtype=torch.float64), torch.nn.Tanh(),
                torch.nn.Linear(64,1,device='cpu',dtype=torch.float64))
        generator = torch.Generator(device='cpu').manual_seed(seed)
        for layer in self.network:
            if isinstance(layer,torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(layer.weight,a=math.sqrt(5),generator=generator)
                bound = 1/math.sqrt(layer.in_features)
                torch.nn.init.uniform_(layer.bias,-bound,bound,generator=generator)
        # Exactly uniform action probabilities initially, despite differing inputs.
        torch.nn.init.zeros_(self.network[-1].weight)
        torch.nn.init.zeros_(self.network[-1].bias)
        self.optimizer = torch.optim.Adam(self.network.parameters(),lr=learning_rate)
        self.rng = np.random.default_rng(seed)
        self.baseline = None
        self.counts = torch.zeros(len(configs),dtype=torch.int64)
        self.updates = 0

    @property
    def logits(self):
        return self.network(self.features).squeeze(-1)

    def logits_for(self, configs):
        """Score other settings with the same network; does not generate tokenizers."""
        return self.network(torch.from_numpy(settings_features(configs))).squeeze(-1)

    def probabilities(self):
        with torch.no_grad(): return self.logits.softmax(0).numpy().copy()

    def sample(self):
        return int(self.rng.choice(self.settings['size'],p=self.probabilities()))

    def update(self,index,reward_bpb):
        reward_bpb = float(reward_bpb)
        if not math.isfinite(reward_bpb) or not 0 <= index < self.settings['size']:
            raise ValueError('Invalid action/reward')
        advantage = 0. if self.baseline is None else reward_bpb-self.baseline
        logp = self.logits.log_softmax(0)
        entropy = -(logp.exp()*logp).sum()
        loss = -(advantage*logp[index]+self.settings['entropy_weight']*entropy)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.network.parameters()):
            raise FloatingPointError('Non-finite settings-policy gradient')
        self.optimizer.step()
        if any(not torch.isfinite(p).all() for p in self.network.parameters()):
            raise FloatingPointError('Non-finite settings-policy weights')
        decay = self.settings['baseline_decay']
        self.baseline = reward_bpb if self.baseline is None else decay*self.baseline+(1-decay)*reward_bpb
        self.counts[index] += 1
        self.updates += 1
        p = self.probabilities()
        h = float(-(p*np.log(np.maximum(p,np.finfo(float).tiny))).sum())
        return {'policy_reward_bpb':reward_bpb,'policy_advantage_bpb':advantage,
                'policy_baseline_bpb':self.baseline,'policy_entropy':h,
                'policy_effective_tokenizers':math.exp(h),'policy_max_probability':float(p.max()),
                'policy_selected_probability':float(p[index]),
                'policy_unique_tokenizers':int((self.counts>0).sum()),'policy_updates':self.updates}

    def weights_dict(self):
        return {'network':self.network.state_dict(),'logits':self.logits.detach().clone(),
                'features':self.features.clone(),
                'counts':self.counts.clone(),'updates':self.updates,'settings':self.settings}

    def state_dict(self):
        return {**self.weights_dict(),'optimizer':self.optimizer.state_dict(),
                'rng':self.rng.bit_generator.state,'baseline':self.baseline}

    def load_state_dict(self,state):
        if state['settings'] != self.settings:
            raise ValueError('Policy architecture or settings features changed on resume')
        if (tuple(state['counts'].shape) != tuple(self.counts.shape)
                or (state['counts']<0).any() or int(state['counts'].sum()) != state['updates']
                or any(not torch.isfinite(v).all() for v in state['network'].values())
                or tuple(state['features'].shape) != tuple(self.features.shape)
                or not torch.allclose(state['features'],self.features,rtol=1e-12,atol=1e-12)
                or (state['baseline'] is not None and not math.isfinite(state['baseline']))):
            raise ValueError('Invalid saved settings policy')
        self.network.load_state_dict(state['network'],strict=True)
        self.features.copy_(state['features'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.rng.bit_generator.state = state['rng']
        self.baseline = state['baseline']
        self.counts.copy_(state['counts'])
        self.updates = state['updates']
        # Different CPU BLAS kernels may differ by a few floating-point ULPs.
        if not torch.allclose(self.logits.detach(),state['logits'],rtol=1e-12,atol=1e-12):
            raise ValueError('Saved network logits do not match current features')
