"""Matched raw-BPB versus canonical-gap policy report; complete fixed environment."""
import gzip
import json

import numpy as np
import torch

from pool_generalization import file_sha, save_json
from frozen_policy_search import ROOT, entropy_optimum, reference, reward_values


def export_regret(report, summary, cache):
    plan = summary['run_plan'];baseline_dir = ROOT / plan['baseline_report']
    baseline = json.loads((baseline_dir / 'analysis.json').read_text())
    if summary['policy_settings'] != baseline['policy_settings']:
        raise ValueError('Reward ablation changed policy settings')
    if file_sha(report / 'reward_table.json.gz') != file_sha(baseline_dir / 'reward_table.json.gz'):
        raise ValueError('Reward ablation changed the complete evaluation data')
    actions = json.loads(gzip.decompress((report / 'actions.json.gz').read_bytes()))
    old_actions = json.loads(gzip.decompress((baseline_dir / 'actions.json.gz').read_bytes()))
    trace = json.loads(gzip.decompress((report / 'reward_trace.json.gz').read_bytes()))
    if len(actions) != 20000 or len(trace) != 20000 or len(old_actions) != 20000:
        raise ValueError('All 20,000 updates and selected rewards are required')
    refs = reference(cache);raw = reward_values(refs['screen']);gap = reward_values(refs['screen'], 'canonical_gap')
    raw_ema = None;advantage_error = 0.;baseline_shift_error = 0.
    decay = summary['policy_settings']['baseline_decay']
    for step, (index, row) in enumerate(zip(actions, trace), 1):
        if (row['step'] != step or row['index'] != index or row['raw_bpb'] != raw[index]
                or row['canonical_bpb'] != refs['screen']['canonical_bpb']
                or row['reward_bpb'] != gap[index]):
            raise ValueError('A selected reward does not match the exact paired BPB gap')
        raw_advantage = 0. if raw_ema is None else float(raw[index]) - raw_ema
        advantage_error = max(advantage_error, abs(raw_advantage - row['advantage_bpb']))
        raw_ema = float(raw[index]) if raw_ema is None else decay * raw_ema + (1 - decay) * float(raw[index])
        baseline_shift_error = max(baseline_shift_error,
            abs(raw_ema - refs['screen']['canonical_bpb'] - row['baseline_bpb']))
    curve = json.loads((report / 'curve.json').read_text())
    old_curve = json.loads((baseline_dir / 'curve.json').read_text())
    if curve[0] != old_curve[0]:
        raise ValueError('Initial uniform policies differ')
    mismatches = np.flatnonzero(np.asarray(actions) != old_actions)
    weight = summary['policy_settings']['entropy_weight']
    optimum_delta = float(np.max(np.abs(entropy_optimum(raw, weight) - entropy_optimum(gap, weight))))
    weights = torch.load(report / 'policy.pt', weights_only=False)
    old_weights = torch.load(baseline_dir / 'policy.pt', weights_only=False)
    weight_delta = max(float((weights['network'][k] - old_weights['network'][k]).abs().max())
                       for k in weights['network'])
    comparison = dict(reward_change='raw_bpb -> transferred_bpb - trained_canonical_bpb',
        settings_changes={}, reward_table_sha256=file_sha(report / 'reward_table.json.gz'),
        canonical_reward_offset=refs['screen']['canonical_bpb'],
        exact_same_initial_statistics=True, updates=20000, selected_reward_checks=len(trace),
        identical_action_count=20000 - len(mismatches),
        first_different_action_step=int(mismatches[0] + 1) if len(mismatches) else None,
        maximum_final_probability_difference=float(np.abs(weights['logits'].softmax(0).numpy()
            - old_weights['logits'].softmax(0).numpy()).max()),
        maximum_final_network_weight_difference=weight_delta,
        maximum_advantage_difference_on_same_actions=advantage_error,
        maximum_ema_canonical_shift_error=baseline_shift_error,
        maximum_entropy_optimum_probability_difference=optimum_delta,
        before=baseline['final'], after=summary['final'], expected_scores={
            split: dict(raw_policy_bpb=baseline['splits'][split]['policy_bpb'],
                regret_policy_bpb=summary['splits'][split]['policy_bpb'],
                canonical_bpb=refs[split]['canonical_bpb'], uniform_bpb=float(refs[split]['bpb'].mean()))
            for split in ['screen', 'confirm']})
    save_json(report / 'comparison.json', comparison)
    final = summary['final']
    lines = ['# Frozen RL20k: canonical-gap reward versus raw BPB', '',
        '**Only the policy reward changes.** The language model and hypernetwork are frozen at the final jointly trained RL20k checkpoint. A fresh shared-settings policy learns for 20,000 updates, starting from the same exactly uniform initialization and seed 43 as its matched raw-BPB run.', '',
        '`reward = transferred continuation BPB - trained canonical continuation BPB`', '',
        f'The entropy coefficient remains **{weight:g}**. Adam learning rate 0.001, EMA baseline decay 0.99, network architecture, action sampling and all evaluation data remain unchanged. The matched raw-reward report is `{plan["baseline_report"]}`.', '',
        '## Exactly what is scored', '',
        'Every one of the 1,000 native training-pool tokenizers is evaluated on all 288 original reward passages and all 288 original held-out check passages. There are 32 source documents per split in each of nine domains: English web, code, Chinese, Japanese, Arabic, Hindi, math, medical and legal. No text or tokenizer is dropped or replaced.', '',
        'Each input is BOS + prefix + 32 canonical continuation tokens, at most 128 positions in total. Only those 32 continuation tokens are scored. The native-tokenizer and canonical inputs share the same raw prefix, continuation targets and continuation bytes; the output head stays canonical. Corrected token boundaries and full-pool prefix lengths are exactly those of the original complete frozen-policy evaluation.', '',
        'BPB is byte-weighted within each domain, then averaged equally across the nine domains. Subtracting the aggregated canonical BPB is equivalent to aggregating the paired per-passage loss differences with those same weights. Each split scores 9,216 target tokens; reward continuation bytes total 27,039 and held-out continuation bytes total 26,668.', '',
        'Exact cached inference losses are reused without modification. Only the sampled tokenizer’s full-reward-set gap enters each policy update. Complete rankings and all held-out losses are evaluator-only. No LM or hypernetwork optimizer or new GPU inference is used.', '',
        '## Results', '',
        '| Final measurement | Raw BPB reward | Canonical-gap reward |', '|---|---:|---:|']
    for label, split, key in [('Hardest ten probability on reward text', 'screen', 'top10_mass'),
                             ('Exact worst #0635 probability', 'screen', 'top1_mass'),
                             ('Hardest ten probability on held-out text', 'confirm', 'top10_mass')]:
        lines.append(f'| {label} | {baseline["final"][split][key]:.4%} | {final[split][key]:.4%} |')
    for split, label in [('screen', 'Expected gap on reward text'), ('confirm', 'Expected gap on held-out text')]:
        lines.append(f'| {label} | {baseline["final"][split]["expected_gap"]:+.6f} BPB | {final[split]["expected_gap"]:+.6f} BPB |')
    trajectory_note = (f'The first different sampled action occurs at step {comparison["first_different_action_step"]:,}. Tiny floating-point differences can propagate through later updates and stochastic choices; the two runs retain similar aggregate performance. This single matched seed supplies no evidence that regret improves selection.'
                       if len(mismatches) else 'All 20,000 sampled choices match the raw-reward run exactly. Expected BPB is unchanged on both reward and held-out text.')
    lines += ['', f'The final most probable tokenizer is **#{final["most_probable_index"]:04d}**, at **{final["max_probability"]:.4%}** probability (reward-text rank {final["screen"]["most_probable_rank"]}; held-out rank {final["confirm"]["most_probable_rank"]}). The regret run sampled {final["unique_tokenizers"]} distinct tokenizers.', '',
        f'Of 20,000 sampled actions, **{comparison["identical_action_count"]:,} match** the raw-reward run. The maximum final selection-probability difference is {comparison["maximum_final_probability_difference"]:.3g}. These measurements are computed from actual reruns, not inferred from the reward formula.', '',
        trajectory_note, '',
        '![Matched reward comparison](frozen_policy_search.png)', '',
        '## Why a different reward can produce the same learning', '',
        f'Canonical BPB on the full fixed reward set is the same **{refs["screen"]["canonical_bpb"]:.12f}** for every action and every update. Subtracting it shifts every reward by the same constant. Rankings, reward differences and the entropy-regularized optimum are unchanged.', '',
        'The policy already subtracts an EMA reward baseline. The first update has zero advantage; afterward the EMA shifts by the same canonical constant, so reward minus EMA is also unchanged in exact arithmetic. Floating-point rounding can introduce tiny differences; this report measures any resulting trajectory differences rather than assuming bitwise identity.', '',
        f'For the actual sampled action sequence, reconstructing the raw-reward EMA gives a maximum advantage difference of **{advantage_error:.3g} BPB**, and a maximum canonical-offset-adjusted EMA difference of **{baseline_shift_error:.3g} BPB**. The exact entropy-regularized optimum is unchanged.', '',
        'This is one seed in a stationary, exact-full-reward-set experiment. It does not test whether a paired canonical baseline reduces noise when fresh text batches or model weights change during joint training. Higher BPB here means the adversary selects harder tokenizers, not that the frozen model changed.', '',
        '## Files and reproduction', '',
        '- `reward_table.json.gz`: complete unchanged loss table, source IDs, continuation IDs/bytes, settings, hashes and model provenance.',
        '- `run_plan.json`: approved reward change, frozen source hashes and separate output paths.',
        '- `reward_trace.json.gz`: every sampled action’s raw BPB, canonical BPB, actual reward, advantage and EMA baseline.',
        '- `analysis.json`, `curve.json`, `actions.json.gz`: complete rankings, progression and all sampled IDs.',
        '- `comparison.json`: actual reward checks and trajectory/weight/probability comparison against raw reward.',
        '- `policy.pt`: final policy; resumable policy checkpoints are persisted every 2,000 steps.',
        '- `frozen_policy_search.png` and `.pdf`: four matched comparison plots.', '',
        f'Reproduce offline: `python scripts/frozen_policy_search.py --cache reports/frozen-policy-search-20k/reward_table.json.gz --run-plan {plan["plan_path"]}`. Add `--persist` to upload policy checkpoints and the report to the existing cloud volume.', '',
        f'Persistent storage: Modal profile `aurora`, volume `anytok-frozen-policy-search`, prefix `{plan["cloud_prefix"]}/`. Model, tokenizer and corpus artifacts remain in their existing persistent cloud locations.', '']
    (report / 'README.md').write_text('\n'.join(lines))
    names = ['README.md', 'reward_table.json.gz', 'run_plan.json', 'reward_trace.json.gz', 'comparison.json',
             'analysis.json', 'curve.json', 'actions.json.gz', 'policy.pt', 'frozen_policy_search.png', 'frozen_policy_search.pdf']
    save_json(report / 'manifest.json', dict(files={n: dict(bytes=(report / n).stat().st_size, sha256=file_sha(report / n)) for n in names}))
