"""Report the finished 20,000-update raw-BPB RL run (entropy weight 0) trained on Tillicum.

Inputs, all committed: reports/raw-bpb-rl-20k-tillicum/{run-tables.json, status.json, run_config.json}
(run-tables.json is the stdout of extract_raw_bpb_rl_20k_tillicum.py on the cluster; run_config.json is the run's
configuration with the cluster username in two input paths replaced by <user>), plans/tokenizer-pool-1000-v1.json,
docs/raw-bpb-handoff/training-verification.json, reports/regret-pool-ranking-6500/tokenizer-ranking.csv and
reports/rl-tokenizer-selection-20k/data/training/sanity.csv. Writes analysis.json, sanity.csv, most-picked.csv,
checkpoints.csv, jobs.csv, the PNG/PDF figure, README.md and manifest.json in the report folder.
"""
import csv
import json
from pathlib import Path

from package_training_inputs import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/raw-bpb-rl-20k-tillicum'
POOL = ROOT / 'plans/tokenizer-pool-1000-v1.json'
VERIFICATION = ROOT / 'docs/raw-bpb-handoff/training-verification.json'
RANKING = ROOT / 'reports/regret-pool-ranking-6500/tokenizer-ranking.csv'
UNIFORM_SANITY = ROOT / 'reports/rl-tokenizer-selection-20k/data/training/sanity.csv'
FOUR_WAY = ROOT / 'plans/four-way-8k-v1.json'  # records the regret run's configuration at its paused 8,000 checkpoint
SHARED_WITH_REGRET = ('fixed_eval_sha256', 'pairing_protocol', 'paired_reward_source_sha256', 'pool_plan_sha256',
                      'data_manifest_sha256', 'recipe_sha256')
FIGURE = 'raw_bpb_rl_20k_tillicum'
HF_REPO, HF_REVISION, HF_FOLDER = 'Dekiofsa/RLtokenizers', '2e2cf342a0622ba539c774837b595f84af29ec82', 'raw-bpb-rl-20k'
HF_URL = f'https://huggingface.co/{HF_REPO}/tree/{HF_REVISION}/{HF_FOLDER}'
PORT_URL = 'https://github.com/Dekiofsa/anytok/tree/hyak-tillicum-port/hyak'
ALGORITHMS = ('bpe', 'unigram', 'wordpiece')
NAMES = {'bpe': 'BPE', 'unigram': 'Unigram', 'wordpiece': 'WordPiece'}
GREEN, BLUE, PURPLE, GREY = '#36967B', '#2F8FD0', '#8657BC', '#87919D'
ALGORITHM_COLORS = {'bpe': BLUE, 'unigram': PURPLE, 'wordpiece': GREEN, None: '#C9D0D7'}
SANITY_STEPS = sorted({*range(0, 20001, 500), 16409})  # every 500 updates and the graceful stop at 16,409
MIN_SHARE = .10  # a 500-update block has a most-picked tokenizer only if it got >= 10% of the block's picks


def percent(part, whole):
    """Whole percent of two counts, rounding exact halves up (605 of 1,000 is 61%)."""
    return (200 * part + whole) // (2 * whole)


def sanity_means(sanity):
    if sorted({r['step'] for r in sanity}) != SANITY_STEPS or len(sanity) != 2 * len(SANITY_STEPS):
        raise ValueError('Expected T1 and T2 at step 0, every 500 updates and at 16,409')
    if len({(r['tokenizer'], r['examples'], r['bytes']) for r in sanity}) != 2:
        raise ValueError('Sanity examples or scored bytes changed during the run')
    table = []
    for step in SANITY_STEPS:
        t1, t2 = sorted((r for r in sanity if r['step'] == step), key=lambda r: r['tokenizer'])
        transferred = (t1['generated_bpb'] + t2['generated_bpb']) / 2
        canonical = (t1['canonical_bpb'] + t2['canonical_bpb']) / 2
        table.append(dict(step=step, transferred_bpb=transferred, canonical_bpb=canonical, gap_bpb=transferred - canonical))
    return table


def concentration_windows(policy, threshold=.95, longest_gap=100, shortest=500):
    """Stretches where the highest policy probability stayed at or above the threshold, allowing single dips."""
    windows = []
    for row in policy:
        if row['policy_max_probability'] < threshold:
            continue
        if windows and row['step'] - windows[-1][1] <= longest_gap:
            windows[-1][1] = row['step']
        else:
            windows.append([row['step'], row['step']])
    result = []
    for start, end in windows:
        inside = [r['policy_max_probability'] for r in policy if start <= r['step'] <= end]
        if end - start >= shortest:
            result.append(dict(from_step=start, to_step=end, samples=len(inside),
                               samples_at_or_above=sum(value >= threshold for value in inside)))
    return result


def hardness_ranks(ranking, configs):
    rows = list(ranking)
    if sorted(int(r['index']) for r in rows) != list(range(1000)):
        raise ValueError('The ranking does not cover the 1,000 pool tokenizers')
    for row in rows:
        config = configs[int(row['index'])]
        if (row['algorithm'], row['pretokenizer'], int(row['vocab_size'])) != (
                config['algorithm'], config['pretokenizer'], config['vocab_size']):
            raise ValueError(f"Ranking and pool disagree on tokenizer #{row['index']}")
    raw_order = sorted(rows, key=lambda r: -float(r['raw_bpb']))
    raw_rank = {int(r['index']): position for position, r in enumerate(raw_order, start=1)}
    return {int(r['index']): dict(regret_rank=int(r['rank']), raw_bpb_rank=raw_rank[int(r['index'])]) for r in rows}


def most_picked(blocks, configs, ranks):
    if [(b['from_step'], b['to_step']) for b in blocks] != [(e - 499, e) for e in range(500, 20001, 500)]:
        raise ValueError('Expected forty consecutive 500-update blocks')
    if any(sum(b[f'{a}_picks'] for a in ALGORITHMS) != 500 for b in blocks):
        raise ValueError('Every 500-update block must hold 500 picks')
    joined = []
    for block in blocks:
        size = block['to_step'] - block['from_step'] + 1
        tokenizer = block['most_picked_tokenizer'] if block['most_picked_count'] / size >= MIN_SHARE else None
        picks = 0 if tokenizer is None else block['most_picked_count']
        if joined and joined[-1]['tokenizer'] == tokenizer:
            joined[-1]['to_step'] = block['to_step']
            joined[-1]['picks'] += picks
        else:
            joined.append(dict(from_step=block['from_step'], to_step=block['to_step'], tokenizer=tokenizer, picks=picks))
    for row in joined:
        config = configs[row['tokenizer']] if row['tokenizer'] is not None else None
        row.update(algorithm=config and config['algorithm'], pretokenizer=config and config['pretokenizer'],
                   vocab_size=config and config['vocab_size'],
                   share=row['picks'] / (row['to_step'] - row['from_step'] + 1) if config else None,
                   **(ranks[row['tokenizer']] if config else dict(regret_rank=None, raw_bpb_rank=None)))
    return joined


def analyze(tables, status, run_config, pool, verification, ranking, uniform_sanity, four_way):
    configs = pool['configs']
    regret = next(m['config'] for m in four_way['models'] if m['name'] == 'regret')
    if (any(regret[k] != run_config[k] for k in SHARED_WITH_REGRET) or regret['policy']['entropy_weight'] != 0
            or regret['policy_reward'] == run_config['policy_reward']):
        raise ValueError('The regret run should differ from this run in its reward only')
    inputs = tables['inputs']
    if (status['step'], status['total'], status['stage']) != (20000, 20000, 'complete'):
        raise ValueError('The run is not complete')
    if not pool['plan_sha256'] == inputs['pool_plan_sha256_field'] == run_config['pool_plan_sha256']:
        raise ValueError('The run did not use the registered 1,000-tokenizer pool')
    if run_config['fixed_eval_sha256'] != inputs['heldout_tokenizers.json_sha256']:
        raise ValueError('The sanity tokenizers differ from the run configuration')
    handoff = verification['config']
    hash_keys = sorted(k for k in handoff if k.endswith('_sha256'))
    if any(run_config[k] != handoff[k] for k in hash_keys) or run_config['policy'] != handoff['policy']:
        raise ValueError('The run configuration differs from the handoff verification')
    if (run_config['reward_objective'], run_config['policy']['entropy_weight']) != ('raw_bpb', 0.0):
        raise ValueError('Expected the raw-BPB reward with entropy weight 0')
    sanity = tables['sanity']
    step0 = sorted((r for r in sanity if r['step'] == 0), key=lambda r: r['tokenizer'])
    reproduced = all((r['generated_bpb'], r['canonical_bpb'], r['bytes'], r['examples']) ==
                     (h['generated_bpb'], h['canonical_bpb'], h['bytes'], h['examples'])
                     for r, h in zip(step0, verification['initial_eval']['results']))
    uniform0 = sorted((r for r in uniform_sanity if r['method'] == 'uniform' and r['step'] == '0'), key=lambda r: r['tokenizer'])
    means = sanity_means(sanity)
    first, last = means[0], means[-1]
    early = next(row for row in means if row['step'] == 2000)
    policy = tables['policy']
    peak = max(policy, key=lambda r: r['policy_max_probability'])
    blocks = tables['blocks']
    algorithms = [dict(algorithm=a, picks=sum(b[f'{a}_picks'] for b in blocks),
                       pool_tokenizers=sum(c['algorithm'] == a for c in configs)) for a in ALGORITHMS]
    jobs = tables['jobs']
    restart = next(j for j in jobs if j['job'] == '481232')  # updates 4,001-16,409
    checkpoints = tables['checkpoints']
    if [c['step'] for c in checkpoints] != SANITY_STEPS or sum(c['kind'] == 'full restart' for c in checkpoints) != 12:
        raise ValueError('Expected 42 checkpoints, 12 of them full restarts')
    replay = tables['replay']
    return dict(
        updates=status['step'], finished_utc=status['at_utc'],
        sanity=dict(examples=step0[0]['examples'], continuation_bytes=[r['bytes'] for r in step0],
                    tokenizer_entries=inputs['sanity_tokenizer_pieces'], step0_reproduces_handoff_verification=reproduced,
                    uniform_run_step0=[dict(tokenizer=r['tokenizer'], continuation_bytes=int(r['continuation_bytes']),
                                            canonical_bpb=float(r['canonical_bpb'])) for r in uniform0],
                    transferred_bpb=dict(start=first['transferred_bpb'], end=last['transferred_bpb'],
                                         drop_fraction=1 - last['transferred_bpb'] / first['transferred_bpb'],
                                         share_of_drop_by_step_2000=(first['transferred_bpb'] - early['transferred_bpb'])
                                         / (first['transferred_bpb'] - last['transferred_bpb'])),
                    canonical_bpb=dict(start=first['canonical_bpb'], end=last['canonical_bpb']),
                    gap_bpb=dict(start=first['gap_bpb'], end=last['gap_bpb'],
                                 drop_fraction=1 - last['gap_bpb'] / first['gap_bpb'])),
        policy=dict(highest_probability=peak['policy_max_probability'], highest_probability_step=peak['step'],
                    final_highest_probability=policy[-1]['policy_max_probability'],
                    windows_at_or_above_95pct=concentration_windows(policy),
                    distinct_tokenizers=blocks[-1]['distinct_tokenizers_so_far'],
                    distinct_tokenizers_by_step_1500=blocks[2]['distinct_tokenizers_so_far'],
                    sampled_every='50 updates, and update 1'),
        most_picked=most_picked(blocks, configs, hardness_ranks(ranking, configs)),
        algorithms=algorithms,
        configuration=dict(reward_objective=run_config['reward_objective'], policy_reward=run_config['policy_reward'],
                           entropy_weight=run_config['policy']['entropy_weight'],
                           hn_lr_fraction_of_peak_at_20000=status['hn_lr'] / run_config['hn_peak_lr'],
                           hashes_matching_handoff_verification=hash_keys,
                           shared_with_regret_run=list(SHARED_WITH_REGRET), regret_run_policy_reward=regret['policy_reward']),
        replay=dict((k, replay[k]) for k in ('updates', 'first_update_any_field_differs', 'first_update_tokenizer_differs',
                                             'updates_with_a_different_tokenizer', 'timing_fields_ignored')),
        jobs=dict(gpu_hours=sum(j['gpu_hours'] for j in jobs),
                  redo_4001_5000_gpu_hours=1000 * restart['gpu_hours'] / 12409),
        checkpoints=dict(repo_id=HF_REPO, revision=HF_REVISION, folder=HF_FOLDER, count=len(checkpoints),
                         full_restarts=[c['step'] for c in checkpoints if c['kind'] == 'full restart']),
        inputs=dict(run_tables_sha256=sha256(REPORT / 'run-tables.json'), status_sha256=sha256(REPORT / 'status.json'),
                    run_config_redacted_sha256=sha256(REPORT / 'run_config.json'), cluster_files=inputs,
                    pool_plan_sha256=pool['plan_sha256'], ranking_sha256=sha256(RANKING),
                    handoff_verification_sha256=sha256(VERIFICATION), uniform_sanity_sha256=sha256(UNIFORM_SANITY),
                    four_way_spec_sha256=sha256(FOUR_WAY)),
        sanity_means=means)


def plot(result, tables):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Patch
    font_manager.fontManager.addfont(str(ROOT / 'monitoring/web/Manrope.ttf'))
    plt.rcParams.update({'font.family': 'Manrope', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    means = result['sanity_means']
    steps = [row['step'] for row in means]
    ax = axes[0, 0]
    ax.plot(steps, [r['transferred_bpb'] for r in means], '-o', ms=3, color=GREEN, label='Transferred · mean of T1 and T2')
    ax.plot(steps, [r['canonical_bpb'] for r in means], '-o', ms=3, color=GREY, label='Canonical · same continuations')
    ax.set(title='A · Periodic sanity evaluation', ylabel='BPB · lower is better')
    ax.legend(frameon=False)
    ax = axes[0, 1]
    ax.plot(steps, [r['gap_bpb'] for r in means], '-o', ms=3, color=PURPLE)
    ax.set(title='B · Transfer gap on the sanity tokenizers', ylabel='Transferred − canonical · BPB', ylim=(0, None))
    ax = axes[1, 0]
    top = 1.3  # room above the data for the legend
    for row in result['most_picked']:
        ax.axvspan(row['from_step'] - 1, row['to_step'], ymax=1 / top, color=ALGORITHM_COLORS[row['algorithm']], alpha=.16, lw=0)
    policy = tables['policy']
    ax.plot([r['step'] for r in policy], [r['policy_max_probability'] for r in policy], color=BLUE, lw=1.2)
    ax.set(title='C · Policy concentration', ylabel='Highest policy probability', ylim=(0, top), yticks=[0, .25, .5, .75, 1])
    ax.legend(handles=[Patch(color=ALGORITHM_COLORS[a], alpha=.35,
                             label=f'{NAMES[a]} most picked' if a else 'No clear favorite') for a in (*ALGORITHMS, None)],
              frameon=False, fontsize=7.5, loc='upper center', ncol=2)
    ax = axes[1, 1]
    blocks = tables['blocks']
    centers = [b['from_step'] + 249.5 for b in blocks]
    bottom = [0] * len(blocks)
    total = sum(a['picks'] for a in result['algorithms'])
    for entry in result['algorithms']:
        heights = [b[f"{entry['algorithm']}_picks"] for b in blocks]
        ax.bar(centers, heights, width=470, bottom=bottom, color=ALGORITHM_COLORS[entry['algorithm']],
               label=f"{NAMES[entry['algorithm']]} · {percent(entry['picks'], total)}% of picks")
        bottom = [b + h for b, h in zip(bottom, heights)]
    ax.set(title='D · Algorithm of the selected tokenizer', ylabel='Selections per 500 updates', ylim=(0, 590),
           yticks=range(0, 501, 100))
    ax.legend(frameon=False, fontsize=8, loc='upper center', ncol=3)
    for ax in axes.flat:
        ax.set(xlabel='Training update', xlim=(0, 20000))
        ax.grid(axis='y', color='#E9EDF1', lw=.6)
        ax.set_axisbelow(True)
    fig.suptitle('Raw-BPB RL · entropy weight 0 · 20k steps · Tillicum H200 · complete', fontsize=13)
    fig.supxlabel('Sanity: T1 and T2, 92 examples each, every 500 updates and at the graceful stop · '
                  'Policy reward: raw transferred continuation BPB before each update\n'
                  'C: highest policy probability every 50 updates, shaded by the algorithm of the most-picked tokenizer '
                  '(blocks below 10% have no clear favorite)', fontsize=8)
    for suffix in ('png', 'pdf'):
        target = REPORT / f'{FIGURE}.{suffix}'
        temporary = target.with_name(f'{target.stem}.tmp.{suffix}')
        fig.savefig(temporary, dpi=170, facecolor='white', metadata={'CreationDate': None} if suffix == 'pdf' else None)
        temporary.replace(target)
    plt.close(fig)


def write_csv(name, fields, rows):
    with (REPORT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def write_tables(result, tables):
    write_csv('sanity.csv', ['step', 'tokenizer', 'bpb', 'canonical_bpb', 'gap_bpb', 'examples', 'continuation_bytes',
                             'truncated_vocabulary_forms'],
              [dict(step=r['step'], tokenizer=r['tokenizer'], bpb=r['generated_bpb'], canonical_bpb=r['canonical_bpb'],
                    gap_bpb=r['generated_bpb'] - r['canonical_bpb'], examples=r['examples'], continuation_bytes=r['bytes'],
                    truncated_vocabulary_forms=r['truncated_forms']) for r in tables['sanity']])
    write_csv('most-picked.csv', ['from_step', 'to_step', 'tokenizer', 'algorithm', 'pretokenizer', 'vocab_size', 'picks',
                                  'share', 'regret_rank', 'raw_bpb_rank'], result['most_picked'])
    write_csv('checkpoints.csv', ['step', 'kind', 'files', 'bytes', 'manifest_sha256'], tables['checkpoints'])
    write_csv('jobs.csv', ['job', 'part', 'what_it_did', 'state', 'started_utc', 'ended_utc', 'gpu_hours'], tables['jobs'])


def readme(result, tables):
    sanity, policy, jobs, replay = (result[k] for k in ('sanity', 'policy', 'jobs', 'replay'))
    transferred, canonical, gap = sanity['transferred_bpb'], sanity['canonical_bpb'], sanity['gap_bpb']
    windows = ' and '.join(f"{w['from_step']:,}–{w['to_step']:,}" for w in policy['windows_at_or_above_95pct'])
    named = [r for r in result['most_picked'] if r['tokenizer'] is not None]
    longest = max(named, key=lambda r: r['to_step'] - r['from_step'])
    totals = {a['algorithm']: a for a in result['algorithms']}
    picks = sum(a['picks'] for a in result['algorithms'])
    split = sorted({f"#{r['tokenizer']}" for r in named if r['pretokenizer'] != 'no-regex'})
    uniform = sanity['uniform_run_step0']
    full = result['checkpoints']['full_restarts']
    lines = [
        '# Raw-BPB RL · entropy weight 0 · 20k steps on Tillicum', '',
        f"**Complete: {result['updates']:,} of 20,000 updates.** Mean transferred sanity BPB over T1 and T2 fell from "
        f"**{transferred['start']:.3f}** to **{transferred['end']:.3f}** ({transferred['drop_fraction']:.0%} lower), and the "
        f"gap to canonical from **{gap['start']:+.3f}** to **{gap['end']:+.3f} BPB**; canonical BPB went from "
        f"{canonical['start']:.3f} to {canonical['end']:.3f}. {transferred['share_of_drop_by_step_2000']:.0%} of the "
        'transferred-BPB drop came in the first 2,000 updates.', '',
        'This is the raw-BPB counterpart of the regret run (`anytok-pool-regret-rl-training`): the same inputs, as packaged '
        'by the handoff bundle, corrected paired boundaries, entropy weight 0 and sanity evaluation, with the raw '
        "transferred BPB instead of the paired gap as the policy reward. The regret run's configuration in "
        '[plans/four-way-8k-v1.json](../../plans/four-way-8k-v1.json) has the same `fixed_eval_sha256`, '
        '`pairing_protocol`, paired-reward source, pool, corpus and recipe hashes as this run, so the two sanity curves can '
        'be compared update by update through update 8,000, where the regret run is paused. This sanity curve is not '
        "directly comparable with the uniform run's ([Limits](#limits)). This is also not the entropy-0.01 “Raw-BPB RL” "
        'run of the [four-way 8k evaluation](../../plans/four-way-8k-evaluation.md).', '',
        f'![Raw-BPB RL on Tillicum]({FIGURE}.png)', '',
        f'[PDF]({FIGURE}.pdf) · [Sanity evaluation](sanity.csv) · [Most-picked tokenizers](most-picked.csv) · '
        '[Checkpoints](checkpoints.csv) · [Jobs](jobs.csv) · [Analysis](analysis.json) · [Run tables](run-tables.json) · '
        '[Run plan](../../plans/raw-bpb-rl-20k-tillicum.md)', '',
        '## Run', '',
        '- Trainer `runtime/anytok_pool_raw_bpb_rl_20k` and launcher `scripts/run_raw_bpb_training.py` from the raw-BPB '
        'handoff ([docs/raw-bpb-handoff](../../docs/raw-bpb-handoff), commit 670993b), with the input bundle it pins: '
        'Hugging Face dataset `cminst/anytok-training-inputs`, revision `c2421b0c`. All source and input hashes in '
        '[run_config.json](run_config.json) and its policy settings equal those of the handoff verification, and `pool_plan_sha256` '
        'equals [plans/tokenizer-pool-1000-v1.json](../../plans/tokenizer-pool-1000-v1.json). The committed '
        '`run_config.json` replaces the cluster username in two input paths with `<user>`.',
        '- Policy reward: the raw transferred continuation BPB of the update\'s 128 paired training examples, measured '
        f"before the model update (`{result['configuration']['policy_reward']}`). Entropy weight "
        f"{result['configuration']['entropy_weight']:g}.",
        '- The 20,000 updates are the start of the 190,000-update learning-rate schedule, as in the uniform run: a '
        '10,000-update ramp, then cosine decay. At update 20,000 the hypernetwork learning rate was '
        f"{result['configuration']['hn_lr_fraction_of_peak_at_20000']:.1%} of its peak.",
        '- UW Hyak Tillicum, one H200 per job, in chained 24-hour Slurm jobs from 2026-11-15 to 2026-11-17 UTC; '
        f"{jobs['gpu_hours']:.1f} GPU-hours in all. The cluster wrappers are on the `hyak-tillicum-port` branch of the "
        f'fork [Dekiofsa/anytok]({PORT_URL}).',
        "- The first attempt crashed while saving update 5,000, when the cluster's shared scratch file system was over "
        'its quota. The run resumed from the update-4,000 full restart, copied to project storage. Redoing updates '
        f"4,001–5,000 took about {jobs['redo_4001_5000_gpu_hours']:.1f} GPU-hours.",
        f"- The replay was not bit-identical. Apart from per-update wall-clock time, every logged training value of the "
        f"{replay['updates']:,} replayed updates matched the first attempt through update "
        f"{replay['first_update_any_field_differs'] - 1:,}; losses first differed at {replay['first_update_any_field_differs']:,}, "
        f"and from {replay['first_update_tokenizer_differs']:,} the policy drew a different tokenizer in "
        f"{replay['updates_with_a_different_tokenizer']} of the {replay['updates']:,} replayed updates. The event log keeps "
        'both versions; these tables use the last row for each update, which is the trajectory the run kept.',
        '- The first restart job was signaled before its 24-hour limit, finished its update, ran the sanity evaluation and '
        'saved a full restart at update 16,409 (a graceful stop); the second job resumed from it.', '',
        '## Periodic sanity evaluation', '',
        'Every 500 updates and at the graceful stop, the trainer scores the two fixed ZeTT sanity tokenizers T1 and T2 '
        f"(`uniform_heldout_tokenizers.json` in the bundle, {sanity['tokenizer_entries'][0]:,} entries each, not in the pool). "
        'For each tokenizer, '
        f"`build_eval_set` takes {sanity['examples']} examples, 4 from each of the 23 held-out buckets: each window is "
        'shortened from 240 down to 80 characters, or the next document is used, until a cut near the middle passes the '
        'paired checks and fits in 128 positions. The prefix is read through the sanity tokenizer and the continuation is '
        f"scored in canonical tokens, so T1 and T2 score different bytes: {sanity['continuation_bytes'][0]:,} and "
        f"{sanity['continuation_bytes'][1]:,}. The step-0 values "
        f"{'reproduce' if sanity['step0_reproduces_handoff_verification'] else 'do not reproduce'} the handoff's H200 "
        'verification exactly (`docs/raw-bpb-handoff/training-verification.json`).', '',
        'Changes between evaluations, such as the rise at update 9,000, are real changes on this fixed text, not '
        'measurement noise, but they rest on only about 10,500 scored bytes per tokenizer.', '',
        '## Policy', '',
        f"Panel C shows the highest policy probability, read every 50 updates. It reached "
        f"**{policy['highest_probability']:.1%}** at update {policy['highest_probability_step']:,}, stayed at or above 95% "
        f"for most of updates {windows}, and ended at {policy['final_highest_probability']:.1%}. Over the run the policy "
        f"drew **{policy['distinct_tokenizers']}** different tokenizers, {policy['distinct_tokenizers_by_step_1500']} of them "
        'in the first 1,500 updates. No tokenizer stayed the most-picked for longer than '
        f"{longest['to_step'] - longest['from_step'] + 1:,} updates (#{longest['tokenizer']}, "
        f"{longest['from_step']:,}–{longest['to_step']:,}).", '',
        '| Updates | Most picked | Algorithm / pretokenizer / vocab | Share of picks | Hardness rank, regret | '
        'Hardness rank, raw BPB |', '| --- | --- | --- | ---: | ---: | ---: |']
    for row in result['most_picked']:
        if row['tokenizer'] is None:
            lines.append(f"| {row['from_step']:,}–{row['to_step']:,} | No clear favorite | | | | |")
        else:
            lines.append(f"| {row['from_step']:,}–{row['to_step']:,} | #{row['tokenizer']} | {NAMES[row['algorithm']]} / "
                         f"{row['pretokenizer']} / {row['vocab_size'] // 1000}k | "
                         f"{percent(row['picks'], row['to_step'] - row['from_step'] + 1)}% | {row['regret_rank']} | "
                         f"{row['raw_bpb_rank']} |")
    lines += [
        '',
        "A 500-update block has a most-picked tokenizer only if it got at least 10% of the block's picks; neighboring "
        'blocks with the same one are joined. The hardness ranks (1 = hardest of the 1,000) come from the '
        "[regret run's step-6,500 ranking](../regret-pool-ranking-6500/README.md), which measured a different model: the "
        'first orders tokenizers by mean regret, as that report does, and the second orders the same measurements by raw '
        "BPB, this run's reward. Neither describes this run's own model.", '',
        f"Over the run, {percent(totals['wordpiece']['picks'], picks)}% of selections were WordPiece, "
        f"{percent(totals['unigram']['picks'], picks)}% Unigram and {percent(totals['bpe']['picks'], picks)}% BPE (rounded), "
        f"while the pool holds {totals['wordpiece']['pool_tokenizers']} WordPiece, {totals['unigram']['pool_tokenizers']} "
        f"Unigram and {totals['bpe']['pool_tokenizers']} BPE tokenizers. Every most-picked tokenizer had a vocabulary of "
        f"{min(r['vocab_size'] for r in named) // 1000}k–{max(r['vocab_size'] for r in named) // 1000}k, and all but "
        f"{', '.join(split)} use the no-regex pretokenizer.", '',
        '## Jobs', '',
        '| Job | What it did | Slurm state | GPU-hours |', '| --- | --- | --- | ---: |']
    for job in tables['jobs']:
        lines.append(f"| {job['part']} ({job['job']}) | {job['what_it_did']} | {job['state']} | {job['gpu_hours']:.2f} |")
    lines += [
        f"| **Total** | | | **{jobs['gpu_hours']:.1f}** |", '',
        'Start and end times (UTC) are in [jobs.csv](jobs.csv).', '',
        '## Limits', '',
        "- **Not directly comparable with the uniform run's sanity curve.** Both trainers cut each sanity passage once and "
        'use that cut for the sanity tokenizer and the canonical control, but in different places. The uniform trainer '
        "cuts at the exact middle character, even inside a word or a canonical token (`make_split_prefix_example` in "
        '`runtime/anytok_pool_20k/hf_hypernet/joint_training.py`). This trainer moves the cut to the nearest canonical '
        'token boundary that passes its reversibility checks (`make_paired_prefix_example` in '
        '`runtime/anytok_pool_regret_rl_20k/hf_hypernet/paired_reward.py`, which `scripts/run_raw_bpb_training.py` '
        'stages into this trainer). Already at step 0 the uniform run scored '
        f"{uniform[0]['continuation_bytes']:,} and {uniform[1]['continuation_bytes']:,} bytes with canonical BPB "
        f"{uniform[0]['canonical_bpb']:.3f} and {uniform[1]['canonical_bpb']:.3f} "
        '([sanity.csv](../rl-tokenizer-selection-20k/data/training/sanity.csv)), against '
        f"{sanity['continuation_bytes'][0]:,} and {sanity['continuation_bytes'][1]:,} bytes with "
        f"{tables['sanity'][0]['canonical_bpb']:.3f} and {tables['sanity'][1]['canonical_bpb']:.3f} here.",
        '- **No final evaluation yet.** Only an evaluation that scores the models the same way can rank this run against '
        'the others. The regret run has no model past update 8,000, so the like-for-like comparison with it is this '
        "run's update-8,000 full restart scored with the four-way 8k panel; against the uniform run, both final 20k "
        'models can be scored.',
        "- **The hardness ranks describe another model.** This run's own model has not been scored on all 1,000 "
        'tokenizers, so it is unknown whether its most-picked tokenizers were the hardest ones for it.',
        '- One run and one seed; the learning rate is still near its peak at update 20,000.', '',
        '## Checkpoints and reproduction', '',
        f"The {result['checkpoints']['count']} checkpoints (every 500 updates, plus 16,409), the event log, the final status "
        f'and the run configuration are on Hugging Face at a pinned revision: [{HF_REPO} · {HF_FOLDER}]({HF_URL}). Each '
        'checkpoint holds `lm.pt`, `hypernet/`, `policy.pt` and a `manifest.json` with the size and SHA-256 of every '
        f"file; the {len(full)} full restarts (0, 2,000, …, 20,000 and 16,409) also hold `state.pt` and `optimizer.pt`. "
        '[checkpoints.csv](checkpoints.csv) lists the SHA-256 of each `manifest.json`. On 2026-11-18 UTC the 239 run '
        'files (the 42 checkpoint folders plus `events.jsonl`, `status.json`, `initial_eval.json`, '
        '`heldout_tokenizers.json` and a package list) were downloaded again from that revision and matched the cluster '
        'copy. The redacted `run_config.json` was uploaded separately and checked against its local copy.', '',
        '`run-tables.json` is the output of `scripts/extract_raw_bpb_rl_20k_tillicum.py`, run read-only on the cluster, '
        "and `status.json` is the run's final status, which the dashboard entry `raw-bpb-entropy0-training` reads. "
        'Rebuild everything else here with `python scripts/report_raw_bpb_rl_20k_tillicum.py`; '
        '`tests/test_raw_bpb_rl_20k_tillicum.py` checks `analysis.json` against these inputs and the manifest against '
        'every file.']
    (REPORT / 'README.md').write_text('\n'.join(lines) + '\n')


def main():
    tables = json.loads((REPORT / 'run-tables.json').read_text())['tables']
    status = json.loads((REPORT / 'status.json').read_text())
    run_config = json.loads((REPORT / 'run_config.json').read_text())
    pool = json.loads(POOL.read_text())
    verification = json.loads(VERIFICATION.read_text())
    four_way = json.loads(FOUR_WAY.read_text())
    with RANKING.open() as ranking, UNIFORM_SANITY.open() as uniform:
        result = analyze(tables, status, run_config, pool, verification, csv.DictReader(ranking),
                         list(csv.DictReader(uniform)), four_way)
    atomic_json(REPORT / 'analysis.json', result)
    write_tables(result, tables)
    plot(result, tables)
    readme(result, tables)
    atomic_json(REPORT / 'manifest.json', dict(files={str(path.relative_to(REPORT)): dict(bytes=path.stat().st_size, sha256=sha256(path))
        for path in sorted(REPORT.rglob('*')) if path.is_file() and path.name != 'manifest.json'}))
    print(json.dumps({key: result[key] for key in ('sanity', 'policy', 'replay', 'jobs')}, indent=1))


if __name__ == '__main__':
    main()
