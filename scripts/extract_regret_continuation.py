"""Extract the regret RL continuation's report inputs from its public Hugging Face records (read-only)."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request

from package_training_inputs import atomic_json

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = 'Dekiofsa/RLtokenizers'
REVISION = '40ae95d65a36e6649c51c6ec1c9a89928d57f5c4'  # main on 2026-11-22, after records/ was added
FOLDER, TWIN = 'regret-rl-20k', 'raw-bpb-rl-20k'
START, STOP, EVERY = 8000, 20000, 500
UPDATES = ['step', 'tokenizer_pool_index', 'paired_gap_bpb', 'paired_canonical_bpb', 'policy_max_probability',
           'policy_selected_probability', 'policy_effective_tokenizers', 'policy_entropy', 'seconds', 'peak_gib',
           'rejected_examples']
RECEIPTS = ['check-receipt', 'restart-receipt', 'allocations', 'submissions', 'complete']
SETTINGS = ['reward_objective', 'policy_reward', 'tokenizer_sampling', 'pool_size', 'batch_size', 'microbatch_size',
            'sequence_length', 'main_updates', 'lr_ramp_updates', 'lr_final_fraction', 'lm_peak_lr', 'hn_peak_lr',
            'lexical_loss_weight', 'pairing_protocol', 'memory_execution', 'training_script_sha256',
            'fixed_eval_sha256', 'data_manifest_sha256', 'recipe_sha256', 'pool_index_sha256', 'pool_plan_sha256', 'seed']
POLICY = ['kind', 'entropy_weight', 'learning_rate', 'baseline_decay', 'hidden_sizes', 'size', 'seed']
JOBS = {'505608': 'Resume check: updates 8,001-8,002, no uploads', '505613': 'Training: updates 8,001-20,000'}


def get(url):
    with urllib.request.urlopen(url, timeout=300) as response:
        return response.read(), response.headers.get('Link') or ''


def tree(revision, folder, recursive=False):
    """Entries under folder at revision, from the Hugging Face tree API (following its Link pagination)."""
    url = f'https://huggingface.co/api/models/{REPO_ID}/tree/{revision}/{folder}' + ('?recursive=true' if recursive else '')
    entries = {}
    while url:
        body, link = get(url)
        entries.update((row['path'], row) for row in json.loads(body))
        url = next(iter(re.findall(r'<([^>]+)>;\s*rel="next"', link)), None)
    return entries


def matches(data, entry):
    """Size and content agree with the tree entry: its LFS SHA-256, or the git blob id of a file kept in git."""
    if entry.get('lfs'):
        return len(data) == entry['lfs']['size'] and hashlib.sha256(data).hexdigest() == entry['lfs']['oid']
    return len(data) == entry['size'] and hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest() == entry['oid']


def fetch(cache, revision, path, entry):
    """The file at revision: the cached copy when it matches the tree, otherwise a fresh anonymous download."""
    target = cache / revision / path
    if target.is_file():
        data = target.read_bytes()
        if matches(data, entry):
            return data
    data, _ = get(f'https://huggingface.co/{REPO_ID}/resolve/{revision}/{path}')
    if not matches(data, entry):
        raise ValueError('Download differs from the Hugging Face tree: ' + path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.part')
    temporary.write_bytes(data)
    temporary.replace(target)
    return data


def lines(data):
    return [json.loads(line) for line in data.decode('utf-8').split('\n') if line.strip()]


def train_rows(events):
    """Updates 1-20,000 once each and in order, with the training invariants the report relies on."""
    train = [row for row in events if row.get('stage') == 'train']
    if [row['step'] for row in train] != list(range(1, STOP + 1)):
        raise ValueError('Train rows are not updates 1-20,000 once each')
    seen = set()
    for row in train:
        seen.add(row['tokenizer_pool_index'])
        if (row['paired_valid_examples'] != 128 or row['paired_maximum_positions'] > 128
                or row['policy_reward_bpb'] != row['paired_gap_bpb'] or row['policy_updates'] != row['step']
                or row['policy_unique_tokenizers'] != len(seen)
                or row['tokenizer_config_index'] != row['tokenizer_pool_index']):
            raise ValueError(f'Training invariant fails at update {row["step"]}')
    return train


def eval_rows(events, run, keep_last=False):
    """One row per check step (initial_eval is step 0); a repeated step is refused unless keep_last."""
    found = {}
    for row in events:
        if row.get('stage') in ('initial_eval', 'eval'):
            step = row['step'] if row['stage'] == 'eval' else 0
            if step in found and not keep_last:
                raise ValueError(f'{run}: check step {step} logged twice')
            found[step] = row['results']
    table = []
    for step, results in sorted(found.items()):
        if len(results) != 2:
            raise ValueError(f'{run}: expected 2 check tokenizers at step {step}')
        row = dict(run=run, step=step)
        for i, result in enumerate(results, 1):
            row.update({f't{i}_{key}': result[key] for key in ('generated_bpb', 'canonical_bpb', 'examples', 'bytes')})
        table.append(row)
    return table


def checkpoint_rows(entries, files, receipts):
    """Each new checkpoint's manifest against its upload receipt and the tree: file set, sizes and LFS SHA-256."""
    rows, checked = [], Counter()
    for step in range(START + EVERY, STOP + 1, EVERY):
        prefix = f'{FOLDER}/checkpoints/step-{step:06d}'
        data, receipt = files[prefix + '/manifest.json'], receipts[step]
        manifest, digest = json.loads(data), hashlib.sha256(data).hexdigest()
        if (manifest['step'] != step or receipt['step'] != step or receipt['kind'] != manifest['kind']
                or receipt['manifest_sha256'] != digest or receipt['prefix'] != prefix or receipt['repo_id'] != REPO_ID):
            raise ValueError(f'Checkpoint {step} differs from its upload receipt')
        stored = {path[len(prefix) + 1:] for path, entry in entries.items()
                  if entry['type'] == 'file' and path.startswith(prefix + '/')}
        if stored != set(manifest['files']) | {'manifest.json'}:
            raise ValueError(f'Checkpoint {step} files differ from its manifest')
        for name, expected in manifest['files'].items():
            entry = entries[f'{prefix}/{name}']
            lfs = entry.get('lfs')  # files kept in git (hypernet/config.json) have no SHA-256 in the tree
            have, want = ((lfs['size'], lfs['oid']), (expected['bytes'], expected['sha256'])) if lfs else (entry['size'], expected['bytes'])
            if have != want:
                raise ValueError(f'Checkpoint {step}: {name} differs from its manifest')
            checked['lfs_sha256_matched' if lfs else 'git_size_matched'] += 1
        rows.append(dict(step=step, kind=manifest['kind'], files=len(manifest['files']),
                         bytes=sum(row['bytes'] for row in manifest['files'].values()), manifest_sha256=digest,
                         revision=receipt['revision'], verified_at_utc=receipt['at_utc']))
    return rows, dict(sorted(checked.items()))


def migrations(events):
    return [dict(step=row['step'], reason=row['reason'], changed=sorted(row.get('changes') or row.get('changed') or []),
                 **({'check_run': row['check_run']} if 'check_run' in row else {}))
            for row in events if row.get('stage') == 'execution_migration']


def summary(files, events, train, prefix_sha256):
    """Provenance, the resume check and the run's receipts, each checked against the others."""
    load = lambda name: json.loads(files[f'{FOLDER}/{name}'])
    check, restart = load('records/receipts/check-receipt.json'), load('records/receipts/restart-receipt.json')
    source, config = load('SOURCE.json'), load('metadata/run_config.json')
    latest, submissions = load('latest_restart.json'), load('records/receipts/submissions.json')
    allocations, complete = load('records/receipts/allocations.json'), load('records/receipts/complete.json')
    moved = migrations(events)
    if (source['check'] != check or not check['passed'] or not all(check['checks'].values())
            or source['source_files'] != restart['files'] or source['source_revision'] != restart['revision']
            or source['source_prefix'] != restart['prefix'] or source['events_at_handoff'] != restart['events']
            or source['source_manifest_sha256'] != restart['latest_restart']['manifest_sha256']
            or restart['total_bytes'] != sum(row['bytes'] for row in restart['files'].values())
            or (source['start_step'], source['stop_step'], restart['step']) != (START, STOP, START)):
        raise ValueError('SOURCE.json, the restart receipt and the check receipt disagree')
    if (config['training_script_sha256'] != source['trainer_sha256'] or config['reward_objective'] != 'regret'
            or config['policy']['entropy_weight'] != 0 or prefix_sha256 != restart['events']['sha256']):
        raise ValueError('The run differs from the source run it continues')
    handoff = [row for row in moved if row['step'] == START]
    if len(handoff) != 1 or not set(handoff[0]['changed']) <= {'model', 'warmup_checkpoint', 'protected_steps'}:
        raise ValueError('The step-8,000 handoff changed more than input locations')
    first = {key: train[START][key] for key in check['update_8001']}  # train[START] is update 8,001
    for name, values in [('resume check', {key: row['check'] for key, row in check['update_8001'].items()}),
                         ('training job', first)]:
        if any(values[key] != row['source'] for key, row in check['update_8001'].items() if key != 'seconds'):
            raise ValueError(f'Update 8,001 of the {name} differs from the source run')
    if (submissions['check'], submissions['train'], allocations, check['job_id']) != ('505608', ['505613'], ['505613'], '505608'):
        raise ValueError('Receipts name other jobs')
    return dict(
        source={key: source[key] for key in ['continuation_of', 'source_revision', 'source_prefix',
                                             'source_manifest_sha256', 'trainer_sha256',
                                             'input_bundle_manifest_sha256', 'start_step', 'stop_step']},
        restart=dict(source_repo=restart['source_repo'], revision=restart['revision'], prefix=restart['prefix'],
                     step=restart['step'], files=len(restart['files']), bytes=restart['total_bytes'],
                     uploaded_revision=restart['latest_restart']['revision'],
                     manifest_sha256=restart['latest_restart']['manifest_sha256'], events_at_handoff=restart['events'],
                     source_events=restart['source_events'], fetched_at_utc=restart['at_utc']),
        handoff_prefix=dict(rows=restart['events']['rows'], sha256=prefix_sha256, matches_source=True),
        check=dict(job_id=check['job_id'], passed=check['passed'], at_utc=check['at_utc'], checks=check['checks'],
                   handoff={key: check['handoff'][key] for key in ['step', 'changed', 'check_run', 'reason']},
                   input_bundle_manifest_sha256=check['input_bundle_manifest_sha256'], update_8001=check['update_8001']),
        training_update_8001=first,
        events=dict(rows=len(events), stages=dict(sorted(Counter(row.get('stage') for row in events).items())),
                    paused_at=[row['step'] for row in events if row.get('stage') == 'paused_at'],
                    checkpoint_steps=[row['step'] for row in events if row.get('stage') == 'checkpoint'],
                    migrations=moved),
        train_checks=dict(updates=len(train), first=train[0]['step'], last=train[-1]['step'],
                          valid_examples=sorted({row['paired_valid_examples'] for row in train}),
                          maximum_positions=max(row['paired_maximum_positions'] for row in train),
                          reward_equals_gap=all(row['policy_reward_bpb'] == row['paired_gap_bpb'] for row in train),
                          policy_counter_equals_update=all(row['policy_updates'] == row['step'] for row in train)),
        run_config=dict({key: config[key] for key in SETTINGS}, policy={key: config['policy'][key] for key in POLICY},
                        protected_steps=config['protected_steps']),
        latest_restart=latest, allocations=allocations, submissions=submissions, complete=complete)


def jobs(path, check, complete):
    """The resume check and training jobs from sacct -X -P rows, with their GPU-hours."""
    header, *body = [line.split('|') for line in Path(path).read_text().splitlines() if line.strip()]
    table = {row[0]: dict(zip(header, row)) for row in body}
    out = []
    for job, part in JOBS.items():
        row = table[job]
        gpus = int(re.search(r'gres/gpu=(\d+)', row['AllocTRES']).group(1))
        hours = int(row['ElapsedRaw']) * gpus / 3600
        out.append(dict(job=job, name=row['JobName'], part=part, account=row['Account'], partition=row['Partition'],
                        qos=row['QOS'], state=row['State'], exit=row['ExitCode'], elapsed=row['Elapsed'],
                        elapsed_seconds=int(row['ElapsedRaw']), timelimit=row['Timelimit'], submit=row['Submit'],
                        start=row['Start'], end=row['End'], node=row['NodeList'], gpus=gpus, gpu_hours=hours))
    offsets = set()  # sacct prints cluster time; each job's last receipt gives its offset from UTC
    for job, at in [('505608', check['at_utc']), ('505613', complete['at_utc'])]:
        seconds = (datetime.fromisoformat(table[job]['End']) - datetime.fromisoformat(at).replace(tzinfo=None)).total_seconds()
        if abs(seconds - 3600 * round(seconds / 3600)) > 120:
            raise ValueError(f'Job {job} did not end when its receipt was written')
        offsets.add(round(seconds / 3600))
    if len(offsets) != 1:
        raise ValueError('The jobs disagree on the cluster time zone')
    return dict(source='sacct -X -P', utc_offset_hours=offsets.pop(), jobs=out,
                total=dict(gpu_hours=sum(row['gpu_hours'] for row in out)))


def write_csv(path, rows, compress=False):
    """Rows as CSV with full-precision numbers; compressed with gzip mtime 0 so equal rows give equal bytes."""
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    data = stream.getvalue().encode()
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(gzip.compress(data, mtime=0) if compress else data)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', default=REVISION, help=f'Hugging Face revision of {REPO_ID} (default {REVISION})')
    parser.add_argument('--cache', type=Path, default=ROOT / 'modal_results/regret_rl_20k_continuation/report-inputs',
                        help='Download cache, one folder per revision (default modal_results/regret_rl_20k_continuation/report-inputs)')
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/regret-rl-20k-continuation/data',
                        help='Compact report inputs (default reports/regret-rl-20k-continuation/data)')
    parser.add_argument('--sacct', type=Path, help='sacct -X -P rows (JobID|...|AllocTRES) with jobs 505608 and '
                                                   '505613; without it the existing data/jobs.json is kept')
    args = parser.parse_args()
    entries, twin_entries = tree(args.revision, FOLDER, recursive=True), tree(args.revision, TWIN)
    pattern = lambda expression: sorted(path for path in entries if re.fullmatch(f'{FOLDER}/{expression}', path))
    manifests, backups = pattern(r'checkpoints/step-\d{6}/manifest\.json'), pattern(r'records/receipts/backups/step-\d{6}\.json')
    wanted = ([f'{FOLDER}/{name}' for name in ['metadata/events.jsonl', 'SOURCE.json', 'latest_restart.json',
                                                'metadata/run_config.json']]
              + manifests + [f'{FOLDER}/records/receipts/{name}.json' for name in RECEIPTS] + backups)
    files = {path: fetch(args.cache, args.revision, path, entries[path]) for path in wanted}
    files[f'{TWIN}/events.jsonl'] = fetch(args.cache, args.revision, f'{TWIN}/events.jsonl', twin_entries[f'{TWIN}/events.jsonl'])

    raw = files[f'{FOLDER}/metadata/events.jsonl']
    events, twin = lines(raw), lines(files[f'{TWIN}/events.jsonl'])
    train = train_rows(events)
    receipts = {row['step']: row for row in (json.loads(files[path]) for path in backups)}
    steps = list(range(START + EVERY, STOP + 1, EVERY))
    if sorted(receipts) != steps or len(manifests) != len(steps) or len(backups) != len(steps):
        raise ValueError('Expected one manifest and one upload receipt per new checkpoint, 8,500-20,000')
    checkpoints, checked = checkpoint_rows(entries, files, receipts)
    if json.loads(files[f'{FOLDER}/latest_restart.json']) != receipts[STOP]:
        raise ValueError('latest_restart.json does not name the verified step-20,000 restart')
    handoff_rows = json.loads(files[f'{FOLDER}/SOURCE.json'])['events_at_handoff']['rows']
    prefix = hashlib.sha256(b''.join(raw.splitlines(keepends=True)[:handoff_rows])).hexdigest()
    receipt_summary = summary(files, events, train, prefix)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'updates.csv.gz', [{key: row[key] for key in UPDATES} for row in train], compress=True)
    write_csv(out / 'evals.csv', eval_rows(events, 'regret') + eval_rows(twin, 'raw_bpb', keep_last=True))
    write_csv(out / 'checkpoints.csv', checkpoints)
    atomic_json(out / 'receipts.json', receipt_summary)
    if args.sacct:
        atomic_json(out / 'jobs.json', jobs(args.sacct, receipt_summary['check'], receipt_summary['complete']))
    elif not (out / 'jobs.json').is_file():
        raise SystemExit('Pass --sacct to write data/jobs.json')

    parts = {}
    for entry in entries.values():
        if entry['type'] == 'file':
            rest = entry['path'][len(FOLDER) + 1:]
            part = parts.setdefault(rest.split('/')[0] if '/' in rest else 'top level', Counter())
            part.update(files=1, bytes=entry['size'])
    heldout = (entries[f'{FOLDER}/metadata/heldout_tokenizers.json']['oid']
               == twin_entries[f'{TWIN}/heldout_tokenizers.json']['oid'])
    repeated = sorted(step for step, count in Counter(row['step'] for row in twin if row.get('stage') == 'eval').items() if count > 1)
    source = dict(
        repo_id=REPO_ID, revision=args.revision, url=f'https://huggingface.co/{REPO_ID}/tree/{args.revision}',
        files=[dict(path=path, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                    hf_oid=(entries.get(path) or twin_entries[path])['oid'],
                    lfs=bool((entries.get(path) or twin_entries[path]).get('lfs'))) for path, data in sorted(files.items())],
        folder=dict(path=FOLDER, files=sum(part['files'] for part in parts.values()),
                    bytes=sum(part['bytes'] for part in parts.values()),
                    parts={name: dict(part) for name, part in sorted(parts.items())}),
        checkpoint_payload=dict(checkpoints=len(checkpoints), files=sum(row['files'] for row in checkpoints),
                                bytes=sum(row['bytes'] for row in checkpoints), **checked),
        heldout_tokenizers_same_git_blob=heldout,
        events={f'{FOLDER}/metadata/events.jsonl': dict(rows=len(events)),
                f'{TWIN}/events.jsonl': dict(rows=len(twin), eval_steps_logged_twice=repeated, kept='last row per step')})
    atomic_json(out / 'source.json', source)
    print(json.dumps(dict(revision=args.revision, updates=len(train), checkpoints=len(checkpoints), **checked,
                          downloaded=len(files), bytes=sum(len(data) for data in files.values()))))


if __name__ == '__main__':
    main()
