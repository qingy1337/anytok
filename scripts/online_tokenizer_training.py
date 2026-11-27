"""Connect a fresh tokenizer and one policy observation to each joint update."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import tempfile
import time

import numpy as np

import anytok_online
from anytok_online import DOMAINS, Generator, anytok_online_native
from open_unigram_policy import MENU_SIZE, SPLITS, Selector, propose


class OnlineTokenizerTraining:
    def __init__(self, name, corpus, recipe, seed, progress):
        self.generator = Generator(corpus, recipe, progress=progress)
        self.proposals = np.random.default_rng(seed + 2)
        self.generation = random.Random(seed + 3)
        width = len(SPLITS) + 1 + len(DOMAINS) + len(DOMAINS['web']) + len(DOMAINS['wikipedia'])
        self.selector = Selector(name, width, seed + 1)
        self.updates = 0
        self.pending = None
        self.settings = dict(
            kind=name, proposal_count=MENU_SIZE, corpus_bytes=self.generator.corpus_bytes,
            domain_and_language_mixtures='same continuous sampler as the Rift comparison',
            pretokenizers=list(SPLITS), vocabulary_range=[32000, 128000],
            stride=32, max_piece_bytes=16, length_bias=1.0, score_noise=0.0,
            policy_seed=seed + 1, proposal_seed=seed + 2, generation_seed=seed + 3,
        )
        if name != 'random':
            self.settings.update(hidden_sizes=[64, 64], learning_rate=.001)
        if name == 'reinforce':
            self.settings.update(entropy_weight=0., baseline_decay=.99)
        if name == 'predictor':
            self.settings['random_every'] = 4

    def sample(self, canonical, fallback, build_surface_forms):
        if self.pending is not None:
            raise RuntimeError('The previous tokenizer has no observed reward')
        started = time.perf_counter()
        settings, features, quota_rejections = propose(self.proposals, self.generator)
        seed = self.generation.getrandbits(64)
        valid = list(range(len(settings)))
        ineligible = []
        while valid:
            index, choice = self.selector.choose(features, valid, self.updates + 1)
            selected = dict(settings[index], seed=seed)
            try:
                sample = self.generator.sample(**selected)
            except ValueError as exc:
                if 'distinct eligible pieces for this request' not in str(exc):
                    raise
                ineligible.append(dict(index=index, settings=selected, reason=str(exc)))
                valid.remove(index)
                continue
            break
        else:
            raise ValueError('No tokenizer in the current menu can realize its requested vocabulary')
        native, forms, priors, truncated = sample.training_inputs(canonical, fallback, build_surface_forms)
        record = dict(
            proposal_index=index, proposal_rejected_quotas=quota_rejections,
            proposal_ineligible=ineligible, tokenizer_settings=selected,
            tokenizer_bucket_quotas=sample.bucket_quotas, tokenizer_sample_bytes=sample.bytes_used,
            tokenizer_sha256=hashlib.sha256(native.to_str().encode()).hexdigest(),
            tokenizer_vocab_size=len(sample.vocabulary), truncated_forms=truncated,
            tokenizer_seconds=round(time.perf_counter() - started, 3),
            **{f'policy_{key}': value for key, value in choice.items()},
        )
        self.pending = (features, valid, index)
        return native, forms, priors, record

    def update(self, reward_bpb):
        if self.pending is None:
            raise RuntimeError('No tokenizer is waiting for a reward')
        self.selector.observe(*self.pending, reward_bpb)
        self.pending = None
        self.updates += 1
        return dict(policy_reward_bpb=reward_bpb, policy_updates=self.updates)

    def weights_dict(self):
        network = self.selector.network
        return dict(selector=self.selector.name, updates=self.updates,
                    network=None if network is None else network.state_dict())

    def state_dict(self):
        if self.pending is not None:
            raise RuntimeError('Save only after the selected tokenizer has a reward')
        return dict(updates=self.updates, proposals=self.proposals.bit_generator.state,
                    generation=self.generation.getstate(), selector=self.selector.state_dict())

    def load_state_dict(self, state):
        self.updates = state['updates']
        self.proposals.bit_generator.state = state['proposals']
        self.generation.setstate(state['generation'])
        self.selector.restore(state['selector'])


def native_source_sha256():
    return {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name, path in [('python', anytok_online.__file__),
                               ('native', anytok_online_native.__file__)]}


def rollback_uncommitted_steps(output, restart_step):
    """Preserve outputs after the last verified restart before replaying them."""
    checkpoints = [path for path in output.glob('step-*')
                   if path.is_dir() and int(path.name.removeprefix('step-')) > restart_step]
    events = output / 'events.jsonl'
    original = events.read_bytes() if events.exists() else b''
    retained = []
    lines = original.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except ValueError:
            if index == len(lines) - 1 and not line.endswith(b'\n'):
                break
            raise
        if row.get('step', 0) <= restart_step:
            retained.append(line if line.endswith(b'\n') else line + b'\n')
    replay = b''.join(retained)
    if not checkpoints and original == replay:
        return
    archive_root = output / 'superseded'
    archive_root.mkdir(exist_ok=True)
    archive = Path(tempfile.mkdtemp(prefix=f'after-{restart_step:06d}-', dir=archive_root))
    if original != replay:
        (archive / events.name).write_bytes(original)
        with tempfile.NamedTemporaryFile(dir=output, prefix='.events-', delete=False) as handle:
            handle.write(replay)
            staged = Path(handle.name)
        staged.replace(events)
    for path in checkpoints:
        path.rename(archive / path.name)


def timestamp():
    return datetime.now(timezone.utc).isoformat()
