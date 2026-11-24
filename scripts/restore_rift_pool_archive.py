"""Restore saved Rift results and input proofs for the unfinished checkpoints."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import tempfile


REFERENCE_ONLY = {'run/features-ready.json', 'run/hardware-check.json'}


def read(path):
    return json.loads(path.read_text())


def fingerprint(value):
    content = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(content).hexdigest()


def verify_snapshot(workspace, sources):
    plan = read(workspace / 'plan.json')
    pause = read(workspace / 'run/pause-receipt.json')
    audit = read(workspace / 'run/panel-audit.json')
    if pause['stage'] != 'paused' or (pause['completed'], pause['total']) != (sources['saved'], sources['total']):
        raise ValueError('Wrong or incomplete pause snapshot')
    if audit['plan_sha256'] != fingerprint(plan) or audit['scored_tokens'] != 4992:
        raise ValueError('Paused snapshot has a different scoring plan')
    for name, key in [('scoring-panel.json', 'panel_sha256'), ('native-prefixes.json', 'native_prefixes_sha256')]:
        if fingerprint(read(workspace / 'run' / name)) != audit[key]:
            raise ValueError(f'Changed scoring input: {name}')
    if [r['label'] for r in pause['checkpoints']] != plan['checkpoints']:
        raise ValueError('Paused checkpoint sequence differs from the plan')
    remaining = sources['checkpoints']['items']
    if {r['label'] for r in pause['checkpoints'] if not r['complete']} != set(remaining):
        raise ValueError('Resume source manifest has a different pending set')

    pairs = 0
    for row in pause['checkpoints']:
        label = row['label']
        score_directory = workspace / 'run/scores' / label
        ids = sorted(int(path.stem) for path in score_directory.glob('[0-9][0-9][0-9][0-9].json'))
        if ids != list(range(row['completed'])) or (score_directory / 'complete.json').is_file() != row['complete']:
            raise ValueError(f'Incomplete archived results for {label}')
        pairs += len(ids)
        if label in remaining:
            source = remaining[label]
            original = read(workspace / 'resume-reference/checkpoints' / label / 'ready.json')
            if (source['saved'] != row['completed'] or source['step'] != original['step']
                    or any(original['files'][k] != value for k, value in source['files'].items())):
                raise ValueError(f'Checkpoint source differs from the original scoring weights for {label}')
    if pairs != sources['saved'] or sources['total'] - pairs != sources['remaining']:
        raise ValueError('Incorrect restored score count')
    return dict(saved=pairs, remaining=sources['remaining'], pending_checkpoints=len(remaining),
                plan_sha256=audit['plan_sha256'], scored_tokens=audit['scored_tokens'])


def restore(archive, workspace, sources):
    ref = sources['archive']
    with archive.open('rb') as stream:
        if archive.stat().st_size != ref['bytes'] or hashlib.file_digest(stream, 'sha256').hexdigest() != ref['sha256']:
            raise ValueError('The archive differs from the checksum-verified HF backup')
    workspace.parent.mkdir(parents=True, exist_ok=True)
    if workspace.exists():
        raise FileExistsError(f'Choose an empty destination. {workspace} already exists')
    with tempfile.TemporaryDirectory(dir=workspace.parent) as scratch:
        staged = Path(scratch) / 'sweep'
        staged.mkdir()
        with tarfile.open(archive, 'r|gz') as bundle:
            for member in bundle:
                name = PurePosixPath(member.name)
                if name.is_absolute() or '..' in name.parts or not (member.isreg() or member.isdir()):
                    raise ValueError(f'Unexpected archive member {member.name!r}')
                # Receipt files alone do not make weights, features or a GPU ready.
                if member.name in REFERENCE_ONLY or name.parts[0] == 'checkpoints':
                    member = member.replace(name=f'resume-reference/{member.name}')
                bundle.extract(member, staged, filter='data')
        result = verify_snapshot(staged, sources)
        (staged / 'sources.json').write_text(json.dumps(sources, indent=2) + '\n')
        (staged / 'resume-verified.json').write_text(json.dumps(result, indent=2) + '\n')
        staged.rename(workspace)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('sources', type=Path)
    parser.add_argument('workspace', type=Path)
    args = parser.parse_args()
    print(json.dumps(restore(args.archive, args.workspace.resolve(), read(args.sources))))


if __name__ == '__main__':
    main()
