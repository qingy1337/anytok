"""Read-only: extract the tables behind reports/raw-bpb-rl-20k-tillicum from the finished Tillicum run.

Run on a Tillicum login node with the run's Python, as the user who owns the run:

    python -I scripts/extract_raw_bpb_rl_20k_tillicum.py --run RUN --first-attempt FIRST --pool POOL \
        > reports/raw-bpb-rl-20k-tillicum/run-tables.json

RUN is the run's output directory (events.jsonl, status.json, run_config.json, heldout_tokenizers.json and the
checkpoints from 4,000 on). FIRST is the first attempt's output directory, which still holds checkpoints 0-3,500.
POOL is the input bundle's pool/plan.json. Slurm accounting supplies the jobs, in UTC. Only stdout is written.

events.jsonl holds updates 4,001-5,000 twice: the first attempt crashed while saving update 5,000 and the run
resumed from the update-4,000 restart. The last row for each (stage, step) is the one the finished run kept; the
first copy is compared with it in the replay table. Values keep the trainer's names and full precision.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ALGORITHMS = ('bpe', 'unigram', 'wordpiece')
JOBS = {'473379': ('Smoke test', 'A 3-update test before the real run'),
        '473452': ('First attempt, job 1', 'Updates 1–5,000; crashed while saving update 5,000'),
        '473453': ('First attempt, job 2', 'Exited at once because job 1 had failed'),
        '473454': ('First attempt, job 3', 'Exited at once because job 1 had failed'),
        '481232': ('Restart, job 1', 'Updates 4,001–16,409 from the update-4,000 restart; graceful stop'),
        '481233': ('Restart, job 2', 'Updates 16,410–20,000'),
        '481234': ('Restart, job 3', 'Third queued job, canceled before it started')}
TIMING = {'seconds'}  # wall-clock per update; not a training value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--first-attempt', type=Path, required=True)
    parser.add_argument('--pool', type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in (args.run / 'events.jsonl').read_text().splitlines() if line.strip()]
    first, last = {}, {}
    for row in rows:
        first.setdefault((row.get('stage'), row.get('step')), row)
        last[(row.get('stage'), row.get('step'))] = row
    train = {step: row for (stage, step), row in last.items() if stage == 'train'}
    if sorted(train) != list(range(1, 20001)):
        raise ValueError('train steps are not exactly 1..20000')

    status = json.loads((args.run / 'status.json').read_text())
    run = [dict(steps_done=status['step'], steps_total=status['total'], stage=status['stage'], finished_utc=status['at_utc'])]

    sanity = []
    initial = [row for row in rows if row.get('stage') == 'initial_eval'][-1]
    for step, row in [(0, initial)] + sorted(((s, r) for (st, s), r in last.items() if st == 'eval'), key=lambda x: x[0]):
        for i, result in enumerate(row['results'], start=1):
            sanity.append(dict(step=step, tokenizer=f'T{i}', generated_bpb=result['generated_bpb'],
                               canonical_bpb=result['canonical_bpb'], examples=result['examples'],
                               bytes=result['bytes'], truncated_forms=result['truncated_forms']))

    policy = [dict(step=s, policy_max_probability=train[s]['policy_max_probability'],
                   policy_effective_tokenizers=train[s]['policy_effective_tokenizers'])
              for s in [1] + list(range(50, 20001, 50))]

    plan = json.loads(args.pool.read_text())
    configs = plan['configs']
    if [c['index'] for c in configs] != list(range(1000)):
        raise ValueError('expected the 1,000 pool tokenizers in index order')
    for s in (1, 20000):  # the trainer's pool index must name the same tokenizer as the plan
        if train[s]['tokenizer_vocab_size'] != configs[train[s]['tokenizer_pool_index']]['vocab_size']:
            raise ValueError(f'pool index mismatch at step {s}')

    blocks, seen = [], set()
    for end in range(500, 20001, 500):
        picks = collections.Counter(train[s]['tokenizer_pool_index'] for s in range(end - 499, end + 1))
        seen.update(picks)
        most, count = picks.most_common(1)[0]
        by_algorithm = collections.Counter()
        for index, n in picks.items():
            by_algorithm[configs[index]['algorithm']] += n
        blocks.append(dict(from_step=end - 499, to_step=end, most_picked_tokenizer=most, most_picked_count=count,
                           distinct_tokenizers=len(picks), distinct_tokenizers_so_far=len(seen),
                           **{f'{a}_picks': by_algorithm[a] for a in ALGORITHMS}))

    replayed = [s for s in range(4001, 5001) if first[('train', s)] is not last[('train', s)]]
    if replayed != list(range(4001, 5001)):
        raise ValueError('expected updates 4,001-5,000 to be logged twice')
    fields = sorted((set(first[('train', 4001)]) & set(last[('train', 4001)])) - TIMING - {'stage', 'step'})
    differs = {f: [s for s in replayed if first[('train', s)].get(f) != last[('train', s)].get(f)] for f in fields}
    replay = dict(updates=len(replayed), fields_compared=fields, timing_fields_ignored=sorted(TIMING),
                  first_update_any_field_differs=min((d[0] for d in differs.values() if d), default=None),
                  first_update_tokenizer_differs=(differs['tokenizer_pool_index'] or [None])[0],
                  updates_with_a_different_tokenizer=len(differs['tokenizer_pool_index']),
                  per_field={f: dict(first_differs=d[0] if d else None, updates_differing=len(d)) for f, d in differs.items()})

    checkpoints = []
    for step in sorted({*range(0, 20001, 500), 16409}):
        folder = (args.first_attempt if step < 4000 else args.run) / f'step-{step:06d}'
        files = json.loads((folder / 'manifest.json').read_text())['files']
        checkpoints.append(dict(step=step, kind='full restart' if 'optimizer.pt' in files else 'weights only',
                                manifest_sha256=sha256(folder / 'manifest.json'), files=len(files),
                                bytes=sum(f['bytes'] for f in files.values())))

    held = json.loads((args.run / 'heldout_tokenizers.json').read_text())
    if not isinstance(held, list) or len(held) != 2:
        raise ValueError('expected two sanity tokenizers')
    inputs = dict(pool_plan_sha256_field=plan.get('plan_sha256'), pool_plan_file_sha256=sha256(args.pool),
                  sanity_tokenizer_pieces=[len(vocab) for vocab in held],
                  **{f'{name}_sha256': sha256(args.run / name)
                     for name in ('events.jsonl', 'status.json', 'run_config.json', 'heldout_tokenizers.json')})

    sacct = subprocess.run(['sacct', '-X', '-P', '-n', '-S', '2026-11-14', '-E', '2026-11-18', '-j', ','.join(JOBS),
                            '-o', 'JobID,State,Start,End,ElapsedRaw,AllocTRES'],
                           capture_output=True, text=True, check=True, env={**os.environ, 'TZ': 'UTC'}).stdout
    jobs = []
    for line in sacct.splitlines():
        jobid, state, start, end, elapsed, tres = line.split('|')
        if jobid not in JOBS:
            continue
        gpus = sum(int(item.split('=')[1]) for item in tres.split(',') if item.startswith('gres/gpu='))
        jobs.append(dict(job=jobid, part=JOBS[jobid][0], what_it_did=JOBS[jobid][1], state=state.split()[0],
                         started_utc=None if start in ('None', 'Unknown') else start + '+00:00',
                         ended_utc=None if end in ('None', 'Unknown') else end + '+00:00',
                         gpu_hours=int(elapsed) * gpus / 3600))
    if sorted(j['job'] for j in jobs) != sorted(JOBS):
        raise ValueError(f'sacct returned jobs {sorted(j["job"] for j in jobs)}')
    jobs.sort(key=lambda j: j['job'])

    json.dump(dict(tables=dict(run=run, sanity=sanity, policy=policy, blocks=blocks, replay=replay,
                               checkpoints=checkpoints, inputs=inputs, jobs=jobs)),
              sys.stdout, sort_keys=True, separators=(',', ':'))
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
