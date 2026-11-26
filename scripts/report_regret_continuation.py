"""Build the regret RL continuation report from reports/regret-rl-20k-continuation/data/ alone, with no network."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import getpass
import gzip
import io
import json
import math
from pathlib import Path
import re
import shutil
import tempfile

from package_training_inputs import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[1]
START, STOP, SEGMENT, WINDOW, TOP, SPELL = 8000, 20000, 2000, 500, 6, 100
FIGURE = 'regret_rl_20k_continuation'
HF = 'https://huggingface.co'
MOVE = 'Modal B300 to Tillicum H200'  # reason of the source run's first execution_migration
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']  # categorical slots 1-6, in this order
OTHER, REGRET, TWIN, INK, MUTED, AXIS, GRID = '#c3c2b7', '#6250d6', '#898781', '#0b0b0b', '#52514e', '#c3c2b7', '#e1e0d9'
CHECKS = {'trainer_exit_zero': 'The trainer exited with status 0',
          'handoff_recorded': 'One execution_migration event at update 8,000, changing only input locations and protected steps',
          'updates_8001_8002': 'Exactly updates 8,001 and 8,002 ran',
          'policy_counter': 'The policy update counter read 8,001 and 8,002',
          'valid_examples': '128 valid paired examples in each update',
          'maximum_positions': 'At most 128 positions in each update',
          'regret_reward': 'Reward objective regret, with the policy reward equal to the paired gap',
          'same_tokenizer_as_source': "Update 8,001 picked the source run's tokenizer",
          'same_batch_as_source': "Update 8,001 drew the source run's paired batch (same SHA-256)",
          'restart_saved': 'The step-8,002 restart was saved'}


def read_csv(path):
    with (gzip.open(path, 'rt', newline='') if path.suffix == '.gz' else path.open(newline='')) as stream:
        return list(csv.DictReader(stream))


def load(data):
    """The compact inputs in data/, typed."""
    updates = [{key: int(value) if key in ('step', 'tokenizer_pool_index', 'rejected_examples') else float(value)
                for key, value in row.items()} for row in read_csv(data / 'updates.csv.gz')]
    evals = [{key: value if key == 'run' else int(value) if key == 'step' or key.endswith(('_examples', '_bytes'))
              else float(value) for key, value in row.items()} for row in read_csv(data / 'evals.csv')]
    checkpoints = [dict(row, step=int(row['step']), files=int(row['files']), bytes=int(row['bytes']))
                   for row in read_csv(data / 'checkpoints.csv')]
    read = lambda name: json.loads((data / name).read_text())
    return updates, evals, checkpoints, read('receipts.json'), read('jobs.json'), read('source.json')


def complete_updates(updates, total=STOP):
    """The updates sorted by step; refused unless they are updates 1..total, each exactly once."""
    if sorted(row['step'] for row in updates) != list(range(1, total + 1)):
        raise ValueError('Missing or duplicated updates')
    return sorted(updates, key=lambda row: row['step'])


def gaps(row):
    return [row[f't{i}_generated_bpb'] - row[f't{i}_canonical_bpb'] for i in (1, 2)]


def series(evals, run):
    """One run's check rows by step; a step may appear only once."""
    rows = [row for row in evals if row['run'] == run]
    if len({row['step'] for row in rows}) != len(rows):
        raise ValueError(f'{run}: a check step appears twice')
    return {row['step']: row for row in rows}


def sanity(evals):
    """Both runs at every check step they share; both must have scored the same check text there."""
    regret, twin = series(evals, 'regret'), series(evals, 'raw_bpb')
    rows = []
    for step in sorted(set(regret) & set(twin)):
        if any(regret[step][f't{i}_{key}'] != twin[step][f't{i}_{key}'] for i in (1, 2) for key in ('examples', 'bytes')):
            raise ValueError(f'The runs scored different check text at step {step}')
        (a1, a2), (b1, b2) = gaps(regret[step]), gaps(twin[step])
        rows.append(dict(step=step, regret_gap_1=a1, regret_gap_2=a2, regret_mean_gap=(a1 + a2) / 2,
                         raw_bpb_gap_1=b1, raw_bpb_gap_2=b2, raw_bpb_mean_gap=(b1 + b2) / 2,
                         regret_minus_raw_bpb=(a1 + a2) / 2 - (b1 + b2) / 2))
    return rows


def picks(updates, first, last, top=TOP):
    """Picks among updates first..last: the most picked (ties by index) with counts and shares, and the distinct count."""
    counts = Counter(row['tokenizer_pool_index'] for row in updates if first <= row['step'] <= last)
    total = sum(counts.values())
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top]
    return dict(first=first, last=last, updates=total, distinct=len(counts),
                top=[dict(tokenizer=tokenizer, picks=count, share=count / total) for tokenizer, count in ranked])


def segment_picks(updates, size=SEGMENT, top=5):
    last = max(row['step'] for row in updates)
    return [picks(updates, first, min(first + size - 1, last), top) for first in range(1, last + 1, size)]


def segments(updates, moved, size=SEGMENT):
    """Training reward, update time, memory and policy focus per segment of complete, sorted updates."""
    out = []
    for first in range(1, len(updates) + 1, size):
        rows = updates[first - 1:first - 1 + size]
        last = rows[-1]['step']
        mean = lambda key: math.fsum(row[key] for row in rows) / len(rows)
        out.append(dict(first=first, last=last, mean_paired_gap_bpb=mean('paired_gap_bpb'),
                        mean_paired_canonical_bpb=mean('paired_canonical_bpb'), mean_seconds=mean('seconds'),
                        mean_policy_max_probability=mean('policy_max_probability'),
                        max_peak_gib=max(row['peak_gib'] for row in rows),
                        rejected_examples=sum(row['rejected_examples'] for row in rows),
                        gpu='B300' if last <= moved else 'H200' if first > moved else f'B300, H200 from {moved + 1:,}'))
    return out


def window_shares(updates, tokenizers, size=WINDOW):
    """Share of picks in each window of size updates for each of tokenizers, and for all others together."""
    out = []
    for first in range(1, len(updates) + 1, size):
        counts = Counter(row['tokenizer_pool_index'] for row in updates[first - 1:first - 1 + size])
        total = sum(counts.values())
        out.append(dict(first=first, last=first + total - 1, shares=[counts[t] / total for t in tokenizers],
                        other=(total - sum(counts[t] for t in tokenizers)) / total))
    return out


def favorites(updates):
    """Spells of the favorite (highest probability after the update), seen where the pick had that probability."""
    spells, seen = [], 0
    for row in updates:
        if row['policy_selected_probability'] != row['policy_max_probability']:
            continue
        seen += 1
        tokenizer, p = row['tokenizer_pool_index'], row['policy_max_probability']
        if spells and spells[-1]['tokenizer'] == tokenizer:
            spell = spells[-1]
            spell.update(last=row['step'], updates=spell['updates'] + 1, peak_probability=max(spell['peak_probability'], p))
        else:
            spells.append(dict(tokenizer=tokenizer, first=row['step'], last=row['step'], updates=1, peak_probability=p))
    return seen, spells


def milestones(updates, steps):
    """The policy after each of steps: focus, distinct tokenizers picked so far and that update's pick."""
    seen, out, wanted = set(), [], set(steps)
    for row in updates:
        seen.add(row['tokenizer_pool_index'])
        if row['step'] in wanted:
            out.append(dict(step=row['step'], max_probability=row['policy_max_probability'],
                            effective_tokenizers=row['policy_effective_tokenizers'], entropy=row['policy_entropy'],
                            distinct_so_far=len(seen), picked=row['tokenizer_pool_index'],
                            picked_is_favorite=row['policy_selected_probability'] == row['policy_max_probability']))
    return out


def analyze(updates, evals, checkpoints, receipts, jobs):
    """Everything the README, CSVs and figure show, with the consistency checks between the inputs."""
    updates = complete_updates(updates)
    rows = sanity(evals)
    shared = {row['step'] for row in rows}
    if any(step not in shared for step in range(0, STOP + 1, SEGMENT)):
        raise ValueError('Both runs need a check at update 0 and every 2,000 updates')
    text = {tuple(row[f't{i}_{key}'] for i in (1, 2) for key in ('examples', 'bytes')) for row in evals}
    if len(text) != 1:
        raise ValueError('The check text changed between steps')
    later = [row for row in rows if row['step'] >= SEGMENT]
    continued = [row for row in rows if row['step'] > START]
    migration = receipts['events']['migrations'][0]
    if not migration['reason'].startswith(MOVE) or migration['step'] >= START:
        raise ValueError('The source run did not move from Modal B300 to Tillicum H200 before update 8,000')
    if [row['step'] for row in checkpoints] != list(range(START + WINDOW, STOP + 1, WINDOW)):
        raise ValueError('Expected one new checkpoint every 500 updates, 8,500-20,000')
    full = [row for row in checkpoints if row['kind'] == 'full_restart_checkpoint']
    latest = receipts['latest_restart']
    if [latest[key] for key in ('step', 'manifest_sha256', 'revision')] != [full[-1][key] for key in ('step', 'manifest_sha256', 'revision')]:
        raise ValueError('latest_restart.json does not name the newest full restart')
    seen, spells = favorites(updates)
    before = [spell for spell in spells if spell['first'] <= START]
    if not before:
        raise ValueError('No favorite was seen by update 8,000')
    holder = before[-1]
    after = [spell for spell in spells if spell['first'] > holder['last']]
    returns = [spell for spell in after if spell['tokenizer'] == holder['tokenizer']]
    tail = [row for row in updates if row['step'] > START]
    job = next(row for row in jobs['jobs'] if row['part'].startswith('Training'))
    wait = datetime.fromisoformat(job['start']) - datetime.fromisoformat(job['submit'])
    continuation = picks(updates, START + 1, STOP)
    tokenizers = [item['tokenizer'] for item in continuation['top']]
    examples1, bytes1, examples2, bytes2 = text.pop()
    return dict(
        updates=dict(total=len(updates), first=updates[0]['step'], last=updates[-1]['step'],
                     continuation=dict(first=START + 1, last=STOP, updates=len(tail),
                                       max_peak_gib=max(row['peak_gib'] for row in tail),
                                       rejected_examples=sum(row['rejected_examples'] for row in tail),
                                       mean_logged_seconds=math.fsum(row['seconds'] for row in tail) / len(tail))),
        moved=dict(step=migration['step'], reason=migration['reason']),
        sanity=dict(rows=rows, shared_steps=len(rows),
                    check_text=[dict(examples=examples1, bytes=bytes1), dict(examples=examples2, bytes=bytes2)],
                    max_change_between_checks_from_2000={
                        key: max(abs(b[f'{key}_mean_gap'] - a[f'{key}_mean_gap']) for a, b in zip(later, later[1:]))
                        for key in ('regret', 'raw_bpb')},
                    after_start=dict(steps=len(continued), regret_higher=sum(row['regret_minus_raw_bpb'] > 0 for row in continued),
                                     minimum_difference=min(row['regret_minus_raw_bpb'] for row in continued),
                                     maximum_difference=max(row['regret_minus_raw_bpb'] for row in continued))),
        segments=segments(updates, migration['step']),
        picks=dict(before=picks(updates, 1, START), continuation=continuation, segments=segment_picks(updates)),
        figure=dict(tokenizers=tokenizers, windows=window_shares(updates, tokenizers)),
        favorites=dict(seen_updates=seen, spells=len(spells), at_start=holder, spells_after=len(after),
                       returns=dict(spells=len(returns), updates=sum(spell['updates'] for spell in returns),
                                    longest=max(returns, key=lambda spell: spell['updates']) if returns else None),
                       long_spells=[spell for spell in spells if spell['updates'] >= SPELL and spell['last'] > START]),
        milestones=milestones(updates, range(SEGMENT, STOP + 1, SEGMENT)),
        checkpoints=dict(count=len(checkpoints), kinds=dict(sorted(Counter(row['kind'] for row in checkpoints).items())),
                         full_restart_steps=[row['step'] for row in full], files=sum(row['files'] for row in checkpoints),
                         bytes=sum(row['bytes'] for row in checkpoints),
                         first_verified_utc=checkpoints[0]['verified_at_utc'], last_verified_utc=checkpoints[-1]['verified_at_utc']),
        jobs=dict(gpu_hours=jobs['total']['gpu_hours'],
                  training=dict(job=job['job'], queue_hours=wait.total_seconds() / 3600,
                                seconds_per_update=job['elapsed_seconds'] / len(tail))))


def manrope_regular(folder):
    """Static Regular (wght 400) instance of monitoring/web/Manrope.ttf, as in report_rawbpb_pool_ranking.py.

    The source font's timestamp is kept, so the instance, and the PDF that embeds it, are the same on every run."""
    from fontTools.ttLib import TTFont
    from fontTools.varLib import instancer
    path = Path(folder) / 'Manrope-Regular.ttf'
    instancer.instantiateVariableFont(TTFont(ROOT / 'monitoring/web/Manrope.ttf', recalcTimestamp=False), {'wght': 400},
                                      updateFontNames=True).save(path)
    return path


def plot(evals, analysis, target, copy):
    """Two stacked panels, one y axis each: (a) held-out sanity gap, (b) share of picks per 500 updates."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter, MultipleLocator, PercentFormatter
    with tempfile.TemporaryDirectory() as fonts:
        font_manager.fontManager.addfont(str(manrope_regular(fonts)))
        plt.rcParams.update({'font.family': 'Manrope', 'font.size': 9, 'axes.titlesize': 10, 'axes.titlelocation': 'left',
                             'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': AXIS,
                             'axes.labelcolor': MUTED, 'xtick.color': AXIS, 'ytick.color': AXIS,
                             'xtick.labelcolor': MUTED, 'ytick.labelcolor': MUTED, 'pdf.fonttype': 42})
        fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 6.6), sharex=True, constrained_layout=True)
        values = {}
        for run, color, label in [('regret', REGRET, 'Regret run (continued from update 8,000)'), ('raw_bpb', TWIN, 'Raw-BPB twin')]:
            values[run] = [(step, sum(gaps(row)) / 2) for step, row in sorted(series(evals, run).items())]
            top.plot(*zip(*values[run]), color=color, lw=1.5, label=label, solid_joinstyle='round',
                     zorder=3 if run == 'regret' else 2)
        shown = [gap for run in values for step, gap in values[run] if step > 0]
        low, high = math.floor((min(shown) - .005) / .02) * .02, math.ceil((max(shown) + .01) / .02) * .02
        top.set(ylim=(low, high), ylabel='Gap, BPB')
        top.yaxis.set_major_locator(MultipleLocator(.02))
        top.set_title('a  Held-out sanity gap: generated minus canonical BPB, mean of 2 check tokenizers')
        start = {run: dict(values[run])[0] for run in values}
        top.text(650, high - .004, f'Update 0, off scale: {start["regret"]:.3f} for both runs'
                 if f'{start["regret"]:.3f}' == f'{start["raw_bpb"]:.3f}' else
                 f'Update 0, off scale: regret {start["regret"]:.3f}, raw-BPB {start["raw_bpb"]:.3f}',
                 ha='left', va='top', fontsize=8, color=MUTED)
        top.text(START + 180, low + .003, 'continuation start', ha='left', va='bottom', fontsize=8, color=MUTED)
        top.legend(loc='upper right', frameon=False, fontsize=8)
        top.grid(axis='y', color=GRID, lw=.6)
        top.set_axisbelow(True)
        tokenizers, windows = analysis['figure']['tokenizers'], analysis['figure']['windows']
        edges = [row['first'] - 1 for row in windows] + [windows[-1]['last']]
        layers = [('All others', OTHER, [row['other'] for row in windows])] + [
            (f'#{tokenizers[i]}', COLORS[i], [row['shares'][i] for row in windows]) for i in reversed(range(len(tokenizers)))]
        lower = [0.] * len(windows)
        for index, (name, color, shares) in enumerate(layers):  # bottom to top, a white gap along each boundary
            upper = [a + b for a, b in zip(lower, shares)]
            bottom.fill_between(edges, lower + lower[-1:], upper + upper[-1:], step='post', color=color, lw=0)
            if index < len(layers) - 1:
                bottom.step(edges, upper + upper[-1:], where='post', color='white', lw=.8)
            lower = upper
        handles = [Patch(facecolor=color, label=name) for name, color, _ in reversed(layers)]
        bottom.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8,
                      title='Most picked in\n8,001–20,000', title_fontsize=8, alignment='left')
        bottom.set(ylim=(0, 1), xlim=(0, STOP), ylabel='Share of picks', xlabel='Update')
        bottom.yaxis.set_major_locator(MultipleLocator(.25))
        bottom.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        bottom.xaxis.set_major_locator(MultipleLocator(4000))
        bottom.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
        bottom.set_title(f'b  Share of picks per {WINDOW} updates')
        for ax in (top, bottom):
            ax.axvline(START, color=INK, lw=.8, zorder=5)
        target.parent.mkdir(parents=True, exist_ok=True)
        copy.parent.mkdir(parents=True, exist_ok=True)
        for suffix, metadata in [('png', {'Software': None}), ('pdf', {'CreationDate': None})]:
            path = target.with_name(f'{target.name}.{suffix}')
            temporary = target.with_name(f'{target.name}.tmp.{suffix}')
            fig.savefig(temporary, dpi=200, facecolor='white', metadata=metadata)
            temporary.replace(path)
            shutil.copyfile(path, copy.with_name(f'{copy.name}.{suffix}'))
        plt.close(fig)


def short(value, size=12):
    return f'`{value[:size]}…`'


def readme(a, receipts, jobs, source, figure):
    s = {row['step']: row for row in a['sanity']['rows']}
    check, config, latest, restart = receipts['check'], receipts['run_config'], receipts['latest_restart'], receipts['restart']
    origin, cont, moved, fav = receipts['source'], a['updates']['continuation'], a['moved']['step'], a['favorites']
    train = next(job for job in jobs['jobs'] if job['job'] == a['jobs']['training']['job'])
    resume = next(job for job in jobs['jobs'] if job['job'] == check['job_id'])
    repo, revision, text = source['repo_id'], source['revision'], a['sanity']['check_text']
    gb = lambda n: f'{n / 1e9:,.2f} GB'
    local = lambda value: value.replace('T', ' ')
    utc = lambda value: value[:16].replace('T', ' ') + ' UTC'
    zone = f'UTC{jobs["utc_offset_hours"]:+d}'.replace('-', '−')
    pick = lambda item: f'#{item["tokenizer"]} {item["picks"]:,} ({item["share"]:.1%})'
    first, last, final = s[START], s[STOP], a['segments'][-1]
    full, kinds = a['checkpoints']['full_restart_steps'], a['checkpoints']['kinds']
    handoff = next(row for row in receipts['events']['migrations'] if row['step'] == START)
    repeated = next((value['eval_steps_logged_twice'] for value in source['events'].values() if 'eval_steps_logged_twice' in value), [])
    lines = [
        '# Regret RL continuation: updates 8,001–20,000', '',
        f"Qingyun's entropy-0 regret RL run ([{origin['continuation_of']}]({HF}/{origin['continuation_of']})) was continued "
        f"from its step-{START:,} full restart to update **{STOP:,}** in **one** Tillicum H200 allocation (job "
        f"{train['job']}, **{train['elapsed']}**). The trainer, inputs and training settings are the source run's, and update "
        f"8,001 reproduced the source run's logged tokenizer, paired batch, losses and reward exactly. The event log holds "
        f"updates **{a['updates']['first']:,}–"
        f"{a['updates']['last']:,}** once each, and the **{a['checkpoints']['count']}** new checkpoints (**{len(full)}** "
        f"full restarts) are on Hugging Face and match their manifests. The held-out sanity gap went from "
        f"**{first['regret_mean_gap']:.4f}** at update {START:,} to **{last['regret_mean_gap']:.4f}** at {STOP:,} (raw-BPB "
        f"twin: **{first['raw_bpb_mean_gap']:.4f}** and **{last['raw_bpb_mean_gap']:.4f}**). The training reward averaged "
        f"**{final['mean_paired_gap_bpb']:.5f} BPB** over updates {final['first']:,}–{final['last']:,}. With the resume "
        f"check, the continuation used **{a['jobs']['gpu_hours']:.3f}** H200 GPU-hours.", '',
        '## What ran', '',
        f'- **Source run.** Updates 1–{moved:,} ran on Modal (B300) and {moved + 1:,}–{START:,} on Tillicum (H200), where '
        'the run paused. Its log records pauses at '
        + ' and '.join(f'{step:,}' for step in receipts['events']['paused_at'] if step < START)
        + f', and an `execution_migration` event at {moved:,} ("{a["moved"]["reason"]}")'
        + (' that also records the new trainer file (`training_script_sha256`).'
           if 'training_script_sha256' in receipts['events']['migrations'][0]['changed'] else '.'),
        f'- **Starting point.** The step-{START:,} full restart at revision {short(restart["revision"])} of '
        f'{restart["source_repo"]}, prefix `{restart["prefix"]}`, manifest SHA-256 {short(restart["manifest_sha256"])}: '
        f'{restart["files"]} files, {gb(restart["bytes"])}, copied to Tillicum, where every file matched the size and '
        "SHA-256 in that manifest.",
        f'- **Unchanged.** The trainer (SHA-256 {short(origin["trainer_sha256"])}), the input bundle (manifest SHA-256 '
        f'{short(origin["input_bundle_manifest_sha256"])}) and every training setting: reward `{config["policy_reward"]}` '
        f'(objective {config["reward_objective"]}), a `{config["policy"]["kind"]}` policy over {config["pool_size"]:,} '
        f'tokenizers with entropy weight {config["policy"]["entropy_weight"]:g}, batch {config["batch_size"]} (microbatch '
        f'{config["microbatch_size"]}), at most {config["sequence_length"]} positions, and a learning rate that ramps up '
        f'over {config["lr_ramp_updates"]:,} of {config["main_updates"]:,} planned updates. The resume guard '
        '(`scripts/resume_regret_continuation.py`) lets only the input locations change; the `execution_migration` event '
        f'it recorded at update {START:,} lists ' + ' and '.join(f'`{name}`' for name in handoff['changed']) + '.',
        f'- **Continuation.** Updates {cont["first"]:,}–{cont["last"]:,} ({cont["updates"]:,} updates) in job '
        f'{train["job"]}, account {train["account"]}, one H200, from {local(train["start"])[:16]} to '
        f'{local(train["end"])[:16]} (cluster time, {zone}). In all {receipts["train_checks"]["updates"]:,} logged updates, '
        f'each had {" or ".join(map(str, receipts["train_checks"]["valid_examples"]))} valid paired examples within '
        f'{receipts["train_checks"]["maximum_positions"]} positions and a policy reward equal to its paired gap. Over the '
        f'continuation {cont["rejected_examples"]} examples were rejected and replaced, and peak GPU memory was at most '
        f'{cont["max_peak_gib"]:.2f} GiB.', '',
        '## Resume check', '',
        f'Job {resume["job"]} ({resume["qos"]} QOS, {resume["elapsed"]}, node {resume["node"]}) resumed a hard-linked copy '
        f'of the step-{START:,} restart to update 8,002, with no uploads. '
        + (f'All {len(check["checks"])} checks passed.' if all(check['checks'].values()) else 'Not every check passed.'), '',
        '| Check | Meaning | Passed |', '| --- | --- | --- |']
    order = list(CHECKS)
    lines += [f'| `{name}` | {CHECKS.get(name, "")} | {"yes" if passed else "no"} |' for name, passed in
              sorted(check['checks'].items(), key=lambda item: order.index(item[0]) if item[0] in order else len(order))]
    show = lambda key, value: short(value, 16) if key.endswith('sha256') else f'`{value!r}`'
    lines += ['', f"Update 8,001 against the source run's log (`{restart['source_events']['path']}`, "
              f"{restart['source_events']['rows']:,} rows). Every field matches except the time per update.", '',
              f'| Field | Source run | Resume check ({resume["job"]}) | Training job ({train["job"]}) |',
              '| --- | --- | --- | --- |']
    lines += [f'| `{key}` | {show(key, row["source"])} | {show(key, row["check"])} | '
              f'{show(key, receipts["training_update_8001"][key])} |' for key, row in check['update_8001'].items()]
    lines += ['', f'The first {receipts["handoff_prefix"]["rows"]:,} rows of the published `events.jsonl` are '
              f"byte-identical to the source run's log at the handoff (SHA-256 {short(receipts['handoff_prefix']['sha256'])}).",
              '', '## Checkpoints and verification', '',
              f'- **Where.** [{repo}, folder regret-rl-20k]({HF}/{repo}/tree/{revision}/regret-rl-20k): '
              f'`checkpoints/step-{START + WINDOW:06d}` to `step-{STOP:06d}`, one every {WINDOW} updates. '
              f'{kinds.get("weights_only_checkpoint", 0)} hold weights only (`lm.pt`, `hypernet/`, `policy.pt`). The '
              f'{len(full)} full restarts, at ' + ', '.join(f'{step:,}' for step in full[:-1]) + f' and {full[-1]:,}, also '
              f'hold `state.pt` and `optimizer.pt`, so training can resume from them. {a["checkpoints"]["files"]} payload '
              f'files, {gb(a["checkpoints"]["bytes"])}.',
              '- **Verified on upload.** While training ran, the driver uploaded each new checkpoint, compared every '
              "file's size, and each LFS file's SHA-256, on Hugging Face with the checkpoint's `manifest.json`, then wrote an upload "
              f'receipt (`records/receipts/backups/`): the first at {utc(a["checkpoints"]["first_verified_utc"])} (step '
              f'{START + WINDOW:,}), the last at {utc(a["checkpoints"]["last_verified_utc"])} (step {STOP:,}).',
              f"- **Verified again for this report**, at revision {short(revision)}: each manifest hashes to its receipt's "
              '`manifest_sha256`, each folder holds exactly the files its manifest lists, the '
              f'{source["checkpoint_payload"]["lfs_sha256_matched"]} LFS payload files match their manifest size and SHA-256, '
              f'and the {source["checkpoint_payload"]["git_size_matched"]} `hypernet/config.json` files, kept in git rather '
              'than LFS, match their manifest size.',
              f'- **latest_restart.json** names step {latest["step"]:,} (upload revision {short(latest["revision"])}, manifest '
              f"SHA-256 {short(latest['manifest_sha256'])}) and equals that checkpoint's upload receipt. The driver publishes "
              f'it only after a full restart is verified. `complete.json` marks the run complete at '
              f'{utc(receipts["complete"]["at_utc"])}.',
              '- The folder also holds `metadata/` (run config, initial check, check tokenizers, `events.jsonl`), '
              f'`SOURCE.json` and `records/` (receipts, Slurm logs, as-run code): {source["folder"]["files"]} files, '
              f'{gb(source["folder"]["bytes"])} in all. The step-{START:,} restart this run started from stays in the source '
              'repository.', '',
              '## Held-out sanity gap', '',
              f'Every {WINDOW} updates the trainer scores the same {text[0]["examples"]} passages through each of 2 check '
              f'tokenizers that are not in the training pool ({text[0]["bytes"]:,} and {text[1]["bytes"]:,} scored bytes). The '
              'gap is generated minus canonical BPB; the mean is over the 2 tokenizers, and update 0 is the check before the '
              'first update. The raw-BPB twin ([raw-bpb-rl-20k-tillicum](../raw-bpb-rl-20k-tillicum/README.md)) uses the same '
              'check tokenizers' + (' (`heldout_tokenizers.json` is the same git blob in both Hugging Face folders)'
                                    if source['heldout_tokenizers_same_git_blob'] else '')
              + f' and scored the same bytes at all {a["sanity"]["shared_steps"]} check steps the runs share.'
              + (" The twin's log holds its checks at " + ' and '.join(f'{step:,}' for step in repeated)
                 + ' twice; as in its own report, the last row of each is used.' if repeated else ''), '',
              '| Update | Regret, tokenizer 1 | Regret, tokenizer 2 | Regret mean | Raw-BPB mean | Regret − raw-BPB |',
              '| ---: | ---: | ---: | ---: | ---: | ---: |']
    lines += [f'| {step:,} | {s[step]["regret_gap_1"]:.4f} | {s[step]["regret_gap_2"]:.4f} | {s[step]["regret_mean_gap"]:.4f} | '
              f'{s[step]["raw_bpb_mean_gap"]:.4f} | {s[step]["regret_minus_raw_bpb"]:+.4f} |' for step in range(0, STOP + 1, SEGMENT)]
    change, after = a['sanity']['max_change_between_checks_from_2000'], a['sanity']['after_start']
    lines += ['', f'**2 tokenizers × {text[0]["examples"]} passages cannot rank the runs.** After update {SEGMENT:,}, the '
              f"regret run's mean gap moves by up to {change['regret']:.4f} BPB between consecutive shared checks (twin: "
              f'{change["raw_bpb"]:.4f}). At the {after["steps"]} shared checks after update {START:,}, the regret run '
              f'minus the twin ranges from {after["minimum_difference"]:+.4f} to {after["maximum_difference"]:+.4f} BPB and '
              f'is positive at {after["regret_higher"]}. Every shared check step is in [sanity.csv](sanity.csv).', '',
              '## Training reward', '',
              "`paired_gap_bpb` is the policy's reward: on each update's 128 training examples, the transferred BPB through "
              'the picked tokenizer minus the canonical BPB, measured before the model update. Means per '
              f'{SEGMENT:,} updates:', '',
              '| Updates | Mean paired gap, BPB | Mean canonical BPB | Seconds per update | GPU |',
              '| --- | ---: | ---: | ---: | --- |']
    lines += [f'| {row["first"]:,}–{row["last"]:,} | {row["mean_paired_gap_bpb"]:.5f} | {row["mean_paired_canonical_bpb"]:.5f} | '
              f'{row["mean_seconds"]:.2f} | {row["gpu"]} |' for row in a['segments']]
    holder, returns = fav['at_start'], fav['returns']
    at = {row['step']: row for row in a['milestones']}
    lines += ['', "The reward depends on the tokenizers the policy picked, so its fall mixes model progress with the "
              f"policy's choices. Updates 1–{moved:,} ran on a B300, so their times are not comparable with the H200's.", '',
              '## Policy', '',
              "**Concentration.** After each update the log records the policy's highest probability, its entropy in nats "
              'and the effective number of tokenizers, exp(entropy). The entropy weight is 0, so nothing in the objective '
              'keeps probability spread out.', '',
              '| Update | Highest probability | Effective tokenizers | Entropy | Distinct picked so far | Picked | Pick is the favorite |',
              '| ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    lines += [f'| {row["step"]:,} | {row["max_probability"]:.4f} | {row["effective_tokenizers"]:.3f} | {row["entropy"]:.4f} | '
              f'{row["distinct_so_far"]} | #{row["picked"]} | {"yes" if row["picked_is_favorite"] else "no"} |'
              for row in a['milestones']]
    lines += ['', f'**The favorite moved off #{holder["tokenizer"]}.** The favorite is the tokenizer with the highest '
              f'probability after an update; the log shows it at the {fav["seen_updates"]:,} of {a["updates"]["total"]:,} '
              f'updates whose pick had that probability. #{holder["tokenizer"]} was the favorite at update {START:,} '
              f'({at[START]["max_probability"]:.2%}, a spell seen from update {holder["first"]:,}) and was last seen as the '
              f'favorite of that spell at update {holder["last"]:,}. The favorite then changed {fav["spells_after"]} times'
              + (f'; #{holder["tokenizer"]} came back in {returns["spells"]} spells ({returns["updates"]:,} seen updates), '
                 f'the longest {returns["longest"]["first"]:,}–{returns["longest"]["last"]:,}' if returns['spells'] else '')
              + f'. Spells of at least {SPELL} seen updates that reach past update {START:,}:', '',
              '| Favorite | First seen | Last seen | Updates seen as favorite | Peak probability |',
              '| ---: | ---: | ---: | ---: | ---: |']
    lines += [f'| #{spell["tokenizer"]} | {spell["first"]:,} | {spell["last"]:,} | {spell["updates"]:,} | '
              f'{spell["peak_probability"]:.4f} |' for spell in fav['long_spells']]
    before, continuation = a['picks']['before'], a['picks']['continuation']
    lines += ['', f'**Picks.** Over updates {START + 1:,}–{STOP:,} the policy picked {continuation["distinct"]} distinct '
              f'tokenizers; the {TOP} most picked: ' + '; '.join(map(pick, continuation['top']))
              + f'. Over updates 1–{START:,} it picked {before["distinct"]} distinct tokenizers; the most picked: '
              + '; '.join(map(pick, before['top'])) + f'. Per {SEGMENT:,} updates (the top 5 of each are in '
              '[picks.csv](picks.csv)):', '',
              '| Updates | Distinct | Mean highest probability | Most picked | Second | Third |',
              '| --- | ---: | ---: | --- | --- | --- |']
    focus = {row['first']: row['mean_policy_max_probability'] for row in a['segments']}
    lines += [f'| {block["first"]:,}–{block["last"]:,} | {block["distinct"]} | {focus[block["first"]]:.4f} | '
              + ' | '.join(map(pick, block['top'][:3])) + ' |' for block in a['picks']['segments']]
    lines += ['', f'Tokenizer numbers are pool indices (pool index SHA-256 {short(config["pool_index_sha256"])} in the run '
              'config).', '', '## Jobs', '',
              '| Job | Part | QOS | State | Elapsed | Start | End | Node | GPU-hours |',
              '| --- | --- | --- | --- | --- | --- | --- | --- | ---: |']
    lines += [f'| {job["job"]} | {job["part"]} | {job["qos"]} | {job["state"]} ({job["exit"]}) | {job["elapsed"]} | '
              f'{local(job["start"])} | {local(job["end"])} | {job["node"]} | {job["gpu_hours"]:.3f} |'
              for job in jobs['jobs']]
    lines += [f'| **Total** | | | | | | | | **{jobs["total"]["gpu_hours"]:.3f}** |', '',
              f'Times are cluster time ({zone}), from `sacct`. Each job held {train["gpus"]} H200 on account '
              f'{train["account"]}, partition {train["partition"]}. '
              f'Job {train["job"]} waited {a["jobs"]["training"]["queue_hours"]:.1f} hours in the queue and ran '
              f'{cont["updates"]:,} updates in {train["elapsed_seconds"]:,} s: '
              f'{a["jobs"]["training"]["seconds_per_update"]:.2f} s per update with checks, saves and start-up included '
              f'(the logged update times average {cont["mean_logged_seconds"]:.2f} s). The driver could queue a second '
              f'24-hour allocation; it was not needed, and `allocations.json` lists only {train["job"]}.', '',
              '## Interpretation limits', '',
              f'- The sanity check is small: 2 tokenizers × {text[0]["examples"]} passages, about '
              f'{round((text[0]["bytes"] + text[1]["bytes"]) / 2, -2):,.0f} scored bytes each. It tracks how one run\'s '
              'transfer gap moves; it does not say which run transfers better.',
              f'- The twin differs in its reward (raw BPB), in the tokenizers its policy picked, and in hardware and code for '
              f'updates 1–{moved:,}: those ran here on a B300 with an earlier trainer file and memory path, which the '
              f'step-{moved:,} `execution_migration` event replaced (it records new '
              + ', '.join(f'`{key}`' for key in receipts['events']['migrations'][0]['changed']
                          if key in ('training_script_sha256', 'memory_execution', 'memory_source_sha256'))
              + f' values); the twin ran on an H200 with the later trainer and memory path from update 1. '
              f'Before any update their checks already differ by '
              f'{abs(s[0]["regret_minus_raw_bpb"]):.5f} BPB ({s[0]["regret_mean_gap"]:.5f} here, '
              f'{s[0]["raw_bpb_mean_gap"]:.5f} for the twin).',
              '- The training reward is not a fixed benchmark: each update scores different training text through the '
              'tokenizer the policy picked.',
              f'- Policy statistics are read after each update, and the favorite is seen only at the '
              f'{fav["seen_updates"]:,} updates whose pick was the favorite.',
              '- This report does not compare the run with the uniform baseline, whose built-in check cuts passages '
              'differently ([raw-bpb-rl-20k-tillicum](../raw-bpb-rl-20k-tillicum/README.md#limits)), '
              'and it does not rank the pool.', '',
              f'![Held-out sanity gap and picks]({figure}.png)', '',
              f'(a) Held-out sanity gap of this run and the raw-BPB twin; update 0 is off scale. (b) Share of picks per '
              f'{WINDOW} updates for the {len(a["figure"]["tokenizers"])} most-picked tokenizers of {START + 1:,}–{STOP:,} and '
              f'all others. The vertical rule marks update {START:,}, where the continuation starts.', '',
              f'[PDF]({figure}.pdf) · [Sanity checks](sanity.csv) · [Picks](picks.csv) · [Analysis](analysis.json) · '
              '[Inputs](data/) · [Manifest](manifest.json)', '',
              '`data/` holds the compact inputs that `scripts/extract_regret_continuation.py` read from Hugging Face revision '
              f'{short(revision)} (files and hashes in `data/source.json`); `scripts/report_regret_continuation.py` builds '
              'the rest of this folder from them alone.']
    return '\n'.join(lines) + '\n'


def write_csv(path, rows, digits=6):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows({key: f'{value:.{digits}f}' if isinstance(value, float) else value for key, value in row.items()}
                     for row in rows)
    path.write_text(stream.getvalue())


def private_files(folder):
    """Report text files holding a local or unmasked cluster path, an email address, a token or the local account name."""
    patterns = [r'(?<![\w.~-])/(?:Users|home|private|tmp|var|root|mnt|scratch)/', r'/gpfs/(?![\w./-]*<user>)',
                r'(?<![\w.])~/', r'[\w.+-]+@[\w-]+(?:\.[\w-]+)+', r'\bhf_[A-Za-z0-9]{30,}']
    try:
        user = getpass.getuser()
    except Exception:
        user = ''
    if len(user) > 2:
        patterns.append(r'(?<![\w-])' + re.escape(user) + r'(?![\w-])')
    pattern = re.compile('|'.join(patterns))
    return [path.relative_to(folder).as_posix() for path in sorted(folder.rglob('*'))
            if path.is_file() and path.suffix in {'.md', '.json', '.csv'} and pattern.search(path.read_text(errors='replace'))]


def manifest(folder):
    """Bytes and SHA-256 of every file under folder by relative path; dot files and manifest.json are left out."""
    files = {}
    for path in sorted(folder.rglob('*')):
        relative = path.relative_to(folder)
        if path.is_file() and relative.as_posix() != 'manifest.json' and not any(part.startswith('.') for part in relative.parts):
            files[relative.as_posix()] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    return dict(files=files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT / 'reports/regret-rl-20k-continuation',
                        help='Report folder; its data/ holds the inputs (default reports/regret-rl-20k-continuation)')
    parser.add_argument('--plots', type=Path, default=ROOT / 'plots' / FIGURE,
                        help=f'Figure copy without suffix; .png and .pdf are written (default plots/{FIGURE})')
    args = parser.parse_args()
    report = args.report
    updates, evals, checkpoints, receipts, jobs, source = load(report / 'data')
    analysis = analyze(updates, evals, checkpoints, receipts, jobs)
    try:
        json.dumps(analysis, allow_nan=False)
    except ValueError:
        raise ValueError('Nonfinite number in the analysis') from None
    atomic_json(report / 'analysis.json', analysis)
    write_csv(report / 'sanity.csv', analysis['sanity']['rows'])
    write_csv(report / 'picks.csv', [dict(first=block['first'], last=block['last'], distinct_tokenizers=block['distinct'],
                                          rank=rank, tokenizer=item['tokenizer'], picks=item['picks'], share=item['share'])
                                     for block in analysis['picks']['segments'] for rank, item in enumerate(block['top'], 1)],
              digits=4)
    plot(evals, analysis, report / FIGURE, args.plots)
    (report / 'README.md').write_text(readme(analysis, receipts, jobs, source, FIGURE))
    private = private_files(report)
    if private:
        raise ValueError('Absolute path, email address, token or account name in ' + ', '.join(private)
                         + '; manifest.json not written')
    atomic_json(report / 'manifest.json', manifest(report))
    continuation, s = analysis['picks']['continuation'], {row['step']: row for row in analysis['sanity']['rows']}
    print(json.dumps(dict(updates=analysis['updates']['total'], checkpoints=analysis['checkpoints']['count'],
                          full_restarts=len(analysis['checkpoints']['full_restart_steps']),
                          sanity={step: [round(s[step]['regret_mean_gap'], 4), round(s[step]['raw_bpb_mean_gap'], 4)]
                                  for step in (START, STOP)},
                          last_segment_gap=round(analysis['segments'][-1]['mean_paired_gap_bpb'], 5),
                          picks=[(item['tokenizer'], item['picks']) for item in continuation['top']],
                          distinct=continuation['distinct'], **{key: analysis['updates']['continuation'][key]
                                                                for key in ('max_peak_gib', 'rejected_examples')})))


if __name__ == '__main__':
    main()
