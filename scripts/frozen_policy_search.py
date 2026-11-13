"""Exact-reward local policy search with the LM and hypernetwork frozen."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from pool_generalization import digest, file_sha, save_json
from uniform_trajectory import PROFILE_DOMAINS

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'modal_results/frozen_policy_search'
REPORT = ROOT / 'reports/frozen-policy-search-20k'


def policy_class():
    path = ROOT / 'runtime/anytok_pool_settings_rl_20k/hf_hypernet/settings_policy.py'
    spec = importlib.util.spec_from_file_location('frozen_settings_policy', path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.SettingsTokenizerBandit


def balanced_scores(rows, nats):
    array = np.asarray(nats, dtype=float)
    if array.shape[-1] != len(rows) or not np.isfinite(array).all() or (array < 0).any():
        raise ValueError('Invalid complete loss table')
    if Counter(r['domain'] for r in rows) != Counter({d: 32 for d in PROFILE_DOMAINS}):
        raise ValueError('All 32 original passages in all nine domains are required')
    result = np.zeros(array.shape[:-1])
    for domain in PROFILE_DOMAINS:
        ids = [i for i, r in enumerate(rows) if r['domain'] == domain]
        result += array[..., ids].sum(-1) / (sum(rows[i]['continuation_bytes'] for i in ids) * math.log(2)) / 9
    return result


def pack_cache(folder):
    prepared = json.loads((folder / 'preparation.json').read_text());plan = prepared['plan']
    if plan['indices'] != list(range(1000)):
        raise ValueError('Full 1000-tokenizer coverage required')
    raw = [json.loads(gzip.decompress((folder / f'scores-{k}.json.gz').read_bytes())) for k in range(2)]
    for part in raw:
        if part['plan_sha256'] != digest(plan) or part['panel_sha256'] != plan['panel_sha256']:
            raise ValueError('Shard plan or text changed')
        if part['canonical'] != raw[0]['canonical']:
            raise ValueError('Canonical baselines differ between workers')
    results = sorted([r for p in raw for r in p['results']], key=lambda r: r['index'])
    if [r['index'] for r in results] != list(range(1000)):
        raise ValueError('Missing or duplicated tokenizer results')
    rows = prepared['metadata'];base = np.asarray(raw[0]['canonical'], dtype=float)
    losses = np.asarray([r['transferred'] for r in results], dtype=float)
    if base.shape != (576, 32) or losses.shape != (1000, 576, 32):
        raise ValueError('All576 passages and32 scored targets are required for every tokenizer')
    if not np.isfinite(base).all() or not np.isfinite(losses).all() or (base < 0).any() or (losses < 0).any():
        raise ValueError('Invalid raw loss values')
    source = json.loads((ROOT / 'reports/rl-pool-sweep-20k/spec.json').read_text())
    if digest(source) != plan['source_spec_sha256']:
        raise ValueError('Original evaluation specification changed')
    for split in ['screen', 'confirm']:
        selected = [r for r in rows if r['split'] == split];original = source['panels'][split]
        if [r['original_id'] for r in selected] != original['document_ids'] or [r['domain'] for r in selected] != original['domains']:
            raise ValueError('Original source IDs, domain membership or split order changed')
    if any(not r['canonical_joint_encoding_matches'] or r['maximum_positions'] > 128
           or len(r['canonical_continuation_ids']) != 32 or len(r['native_prefix_lengths']) != 1000 for r in rows):
        raise ValueError('Incomplete shared encoding validation')
    return dict(plan=plan, rows=rows, canonical_nats=base.sum(1).tolist(),
        transferred_nats=losses.sum(2).tolist(), configs=[r['config'] for r in results],
        tokenizer_hashes=[r['sha256'] for r in source['tokenizers']])


def reference(cache):
    result = {}
    for split in ['screen', 'confirm']:
        ids = [i for i, r in enumerate(cache['rows']) if r['split'] == split]
        rows = [cache['rows'][i] for i in ids]
        native = balanced_scores(rows, np.asarray(cache['transferred_nats'])[:, ids])
        base = float(balanced_scores(rows, np.asarray(cache['canonical_nats'])[ids]))
        result[split] = dict(bpb=native, canonical_bpb=base, rank=np.argsort(np.argsort(-native, kind='stable'), kind='stable') + 1)
    return result


def entropy_optimum(rewards, weight):
    rewards = np.asarray(rewards, dtype=float)
    if weight == 0:
        p = np.zeros_like(rewards);p[np.argmax(rewards)] = 1.;return p
    p = np.exp((rewards - rewards.max()) / weight)
    return p / p.sum()


def learner_step(policy, reward_values):
    """The learner sees only the reward of its sampled action, never a ranking."""
    action = policy.sample()
    return action, policy.update(action, float(reward_values[action]))


def persist(path, destination, directory=False):
    import modal
    from modal_tokenizer_pool_transfer import source_client
    client, _ = source_client('aurora')
    volume = modal.Volume.from_name('anytok-frozen-policy-search', version=2, client=client)
    with volume.batch_upload(force=True) as upload:
        if directory:
            upload.put_directory(path, destination)
        else:
            upload.put_file(path, destination)
    if directory:
        if b''.join(volume.read_file(destination + '/manifest.json')) != (path / 'manifest.json').read_bytes():
            raise ValueError('Cloud report manifest differs')


def write_status(stage, completed, **fields):
    row = dict(at_utc=datetime.now(timezone.utc).isoformat(), stage=stage, completed=completed,
        total=20000, platform='Local · CPU',
        workers=[] if stage in ['complete', 'failed'] else [dict(id='policy-cpu', profile='local',
            platform='Local · CPU', state='Training shared-settings policy')], **fields)
    save_json(STATE / 'status.json', row)
    with (STATE / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def summarize(policy, refs, configs, uniform_counts):
    p = policy.probabilities();record = dict(step=policy.updates, max_probability=float(p.max()),
        most_probable_index=int(p.argmax()), unique_tokenizers=int((policy.counts > 0).sum()),
        uniform_unique_tokenizers=int((uniform_counts > 0).sum()),
        entropy=float(-(p * np.log(np.maximum(p, np.finfo(float).tiny))).sum()),
        pretokenizer_mass={key: float(sum(p[i] for i, c in enumerate(configs) if c['pretokenizer'] == key))
                          for key in ['regex', 'line256', 'no-regex']})
    for split, ref in refs.items():
        record[split] = dict(expected_bpb=float(p @ ref['bpb']),
            expected_gap=float(p @ ref['bpb'] - ref['canonical_bpb']),
            top1_mass=float(p[ref['rank'] <= 1].sum()), top10_mass=float(p[ref['rank'] <= 10].sum()),
            top50_mass=float(p[ref['rank'] <= 50].sum()),
            best_seen_rank=int(ref['rank'][policy.counts.numpy() > 0].min()) if policy.updates else None,
            most_probable_rank=int(ref['rank'][p.argmax()]))
    return record


def plot(curve, refs, entropy_weight, path):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = ROOT / 'monitoring/web/Manrope.ttf'
    if font.exists():
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': 'Manrope' if font.exists() else 'DejaVu Sans', 'font.size': 9,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True);x = [r['step'] for r in curve]
    for split, color, label in [('screen', '#8657BC', 'Policy reward text'), ('confirm', '#2F8FD0', 'Held-out check text')]:
        baseline = refs[split]['bpb'].mean()
        axes[0, 0].plot(x, [r[split]['expected_bpb'] - baseline for r in curve], color=color, label=label)
    oracle = entropy_optimum(refs['screen']['bpb'], entropy_weight)
    axes[0, 0].axhline(float(oracle @ refs['screen']['bpb'] - refs['screen']['bpb'].mean()), color='#8657BC', ls=':', label='Reward-text optimum with entropy bonus')
    axes[0, 0].set(title='Does the policy select harder tokenizers?', ylabel='Expected BPB above uniform selection')
    axes[0, 0].legend(frameon=False, fontsize=8)
    for key, color, label in [('top1_mass', '#8657BC', 'Hardest 1'), ('top10_mass', '#2F8FD0', 'Hardest 10'), ('top50_mass', '#36967B', 'Hardest 50')]:
        axes[0, 1].plot(x, [r['screen'][key] for r in curve], color=color, label=label)
    axes[0, 1].set(title='Probability on the true hardest reward-text tokenizers', ylabel='Selection probability', ylim=(0, .35))
    axes[0, 1].legend(frameon=False)
    axes[1, 0].plot(x, [r['unique_tokenizers'] for r in curve], color='#8657BC', label='Adversarial policy')
    axes[1, 0].plot(x, [r['uniform_unique_tokenizers'] for r in curve], color='#87919D', label='Uniform sampling')
    axes[1, 0].set(title='Exploration of the full pool', ylabel='Distinct tokenizers sampled', ylim=(0, 1050))
    axes[1, 0].legend(frameon=False)
    for key, color in [('regex', '#2F8FD0'), ('line256', '#36967B'), ('no-regex', '#8657BC')]:
        axes[1, 1].plot(x, [r['pretokenizer_mass'][key] for r in curve], color=color, label=key)
    axes[1, 1].set(title='Learned pretokenizer preferences', ylabel='Selection probability', ylim=(0, 1))
    axes[1, 1].legend(frameon=False)
    for ax in axes.flat:
        ax.set_xlabel('Policy updates');ax.grid(axis='y', color='#E9EDF1', lw=.6);ax.set_axisbelow(True)
    fig.suptitle('Frozen RL20k LM + hypernetwork · all 1,000 tokenizers · only the policy learns', fontsize=13)
    for suffix in ['png', 'pdf']:
        target = path.with_suffix('.' + suffix);temporary = target.with_name(target.stem + '.tmp.' + suffix)
        fig.savefig(temporary, dpi=170, facecolor='white');temporary.replace(target)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT / 'modal_results/frozen_policy_cache')
    args = parser.parse_args();torch.set_num_threads(1);STATE.mkdir(parents=True, exist_ok=True);REPORT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'plots').mkdir(parents=True, exist_ok=True)
    cache = pack_cache(args.cache) if args.cache.is_dir() else json.loads(gzip.decompress(args.cache.read_bytes()))
    if args.cache.is_dir() and (args.cache / 'calls.json').exists():
        calls = json.loads((args.cache / 'calls.json').read_text())
        env = {**os.environ, 'MODAL_PROFILE': calls['profile']}
        for app_id in calls.get('app_ids', [calls['app_id']]):
            subprocess.run([sys.executable, '-m', 'modal', 'app', 'stop', app_id, '--yes'], env=env, check=False)
    settings = cache['plan']['policy'];policy_path = ROOT / 'runtime/anytok_pool_settings_rl_20k/hf_hypernet/settings_policy.py'
    if file_sha(policy_path) != cache['plan']['source_policy_sha256']:
        raise ValueError('The agreed shared-settings policy source changed')
    (REPORT / 'reward_table.json.gz').write_bytes(gzip.compress(json.dumps(cache, sort_keys=True, separators=(',', ':')).encode(), mtime=0))
    refs = reference(cache);configs = cache['configs'];Policy = policy_class()
    policy = Policy(configs, **{key: settings[key] for key in ['seed', 'learning_rate', 'entropy_weight', 'baseline_decay']})
    uniform_rng = np.random.default_rng(430043);uniform_counts = np.zeros(1000, dtype=int);curve = []
    actions = [];curve.append(summarize(policy, refs, configs, uniform_counts));begin = time.monotonic()
    write_status('training policy with frozen full-screen rewards', 0)
    plot(curve, refs, settings['entropy_weight'], ROOT / 'plots/frozen_policy_search')
    for step in range(1, settings['steps'] + 1):
        action, metrics = learner_step(policy, refs['screen']['bpb']);actions.append(action)
        uniform_counts[int(uniform_rng.integers(1000))] += 1
        if step % 100 == 0:
            row = summarize(policy, refs, configs, uniform_counts);curve.append(row)
            save_json(STATE / 'curve.json', curve)
            elapsed = time.monotonic() - begin
            write_status('training policy with frozen full-screen rewards', step, elapsed_seconds=elapsed,
                eta_seconds=elapsed / step * (settings['steps'] - step), statistics=row)
        if step % 1000 == 0:
            plot(curve, refs, settings['entropy_weight'], ROOT / 'plots/frozen_policy_search')
        if step % 2000 == 0:
            checkpoint_path = STATE / f'policy-{step:06d}.pt'
            torch.save(dict(policy=policy.state_dict(), plan_sha256=digest(cache['plan']), actions=actions,
                uniform_counts=uniform_counts.tolist(), uniform_rng=uniform_rng.bit_generator.state, curve=curve),
                checkpoint_path)
            if args.cache.is_dir():
                persist(checkpoint_path, 'frozen-policy-search-v1/policy/' + checkpoint_path.name)
    probability = policy.probabilities();oracle = entropy_optimum(refs['screen']['bpb'], settings['entropy_weight'])
    summary = dict(final=curve[-1], policy_settings=policy.settings, splits={}, steps=policy.updates,
        scope='All1000 native tokenizers; all288 original screen +288 confirmation passages; corrected cuts; frozen finalRL20k; deterministic full-screen reward. Only the shared policy is updated.',
        limitations='One policy seed and one frozen model checkpoint. This stationary exact-reward test excludes training-minibatch noise and co-adapting LM/hypernetwork updates. Confirmation is evaluator-only.')
    for split, ref in refs.items():
        best_index = int(ref['bpb'].argmax())
        uniform_actions = np.random.default_rng(430043).integers(1000, size=len(actions))
        policy_hits = np.flatnonzero(np.asarray(actions) == best_index)
        uniform_hits = np.flatnonzero(uniform_actions == best_index)
        summary['splits'][split] = dict(canonical_bpb=ref['canonical_bpb'], uniform_bpb=float(ref['bpb'].mean()),
            policy_bpb=float(probability @ ref['bpb']), best_index=best_index, best_bpb=float(ref['bpb'].max()),
            first_best_selection_step=int(policy_hits[0] + 1) if len(policy_hits) else None,
            uniform_first_best_selection_step=int(uniform_hits[0] + 1) if len(uniform_hits) else None,
            screen_entropy_optimum_bpb=float(oracle @ ref['bpb']),
            ranking=[dict(index=int(i), rank=int(ref['rank'][i]), bpb=float(ref['bpb'][i]),
                gap_bpb=float(ref['bpb'][i] - ref['canonical_bpb']), probability=float(probability[i]),
                selections=int(policy.counts[i])) for i in np.argsort(-ref['bpb'], kind='stable')])
    save_json(REPORT / 'analysis.json', summary);save_json(REPORT / 'curve.json', curve)
    (REPORT / 'actions.json.gz').write_bytes(gzip.compress(json.dumps(actions).encode(), mtime=0))
    plot(curve, refs, settings['entropy_weight'], REPORT / 'frozen_policy_search')
    torch.save(policy.weights_dict(), REPORT / 'policy.pt')
    save_json(STATE / 'summary.json', summary)
    from export_frozen_policy_search import main as export_report
    export_report()
    if args.cache.is_dir():
        persist(REPORT, 'frozen-policy-search-v1/report', directory=True)
    write_status('complete', policy.updates, elapsed_seconds=time.monotonic() - begin, statistics=curve[-1])


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        STATE.mkdir(parents=True, exist_ok=True);write_status('failed', 0, error=str(exc));raise
