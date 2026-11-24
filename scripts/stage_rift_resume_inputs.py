"""Download only the approved model, native pool and pending weights on the execution host."""

import argparse
import json
from pathlib import Path
import tarfile
import time

from huggingface_hub import hf_hub_download
from package_training_inputs import atomic_json, sha256


def read(path):
    return json.loads(path.read_text())


def check(path, proof):
    return path.is_file() and path.stat().st_size == proof['bytes'] and sha256(path) == proof['sha256']


def fetch_file(record, cache, *, repo_type, token):
    verified = cache / record['path']
    if check(verified, record):
        return verified
    for attempt in range(4):
        try:
            path = Path(hf_hub_download(record['repo'], record['path'], revision=record['revision'],
                                        repo_type=repo_type, local_dir=cache, token=token,
                                        force_download=attempt > 0))
            if check(path, record):
                return path
            raise ValueError(f'Checksum mismatch for {record["path"]}')
        except (OSError, ValueError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def stage_inputs(root):
    sources = read(root / 'sources.json')
    plan = read(root / 'plan.json')
    receipt = root / 'run/input-ready.json'
    if receipt.is_file():
        return read(receipt)
    bundle = fetch_file(sources['input_bundle'], root / 'downloads/input', repo_type='dataset', token=False)
    with tarfile.open(bundle, 'r|gz') as archive:
        for member in archive:
            if member.name.startswith(('input/model/', 'input/pool/')) and member.isreg():
                archive.extract(member, root, filter='data')
    for name, expected in plan['input_hashes'].items():
        if name.startswith('model/') and sha256(root / 'input' / name) != expected:
            raise ValueError(f'Changed canonical model input {name}')
    index = root / 'input/pool/full-results.json'
    if sha256(index) != plan['pool_index_sha256']:
        raise ValueError('The native tokenizer index differs from Rift')
    candidates = read(index)['tokenizers']
    if {c['config']['index'] for c in candidates} != set(range(1000)):
        raise ValueError('All 1,000 native tokenizers are required')
    for candidate in candidates:
        path = root / 'input/pool/tokenizers' / f'{candidate["config"]["index"]:04d}' / 'tokenizer.json'
        if not check(path, dict(bytes=candidate['tokenizer_bytes'], sha256=candidate['tokenizer_sha256'])):
            raise ValueError(f'Changed native tokenizer {path.parent.name}')
    result = dict(tokenizers=len(candidates), model=plan['input_hashes']['model/model.safetensors'],
                  pool=plan['pool_index_sha256'])
    atomic_json(receipt, result)
    # The exact public bundle stays pinned in HF. Only this verified execution copy is temporary.
    bundle.unlink()
    return result


def stage_checkpoint(root, label):
    sources = read(root / 'sources.json')['checkpoints']
    checkpoint = sources['items'][label]
    folder = root / 'checkpoints' / label
    for name, proof in checkpoint['files'].items():
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if check(target, proof):
            continue
        record = dict(proof, repo=sources['repo'], revision=sources['revision'], path=f'{checkpoint["prefix"]}/{name}')
        source = fetch_file(record, folder / 'downloads', repo_type='model', token=False)
        source.replace(target)
    manifest = read(root / 'resume-reference/checkpoints' / label / 'ready.json')
    if manifest['step'] != checkpoint['step'] or any(manifest['files'][r] != p for r, p in checkpoint['files'].items()):
        raise ValueError(f'Wrong original checkpoint manifest for {label}')
    atomic_json(folder / 'ready.json', manifest)
    return dict(checkpoint=label, step=manifest['step'], bytes=sum(p['bytes'] for p in checkpoint['files'].values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', type=Path)
    parser.add_argument('checkpoint', nargs='?')
    args = parser.parse_args()
    result = stage_checkpoint(args.workspace, args.checkpoint) if args.checkpoint else stage_inputs(args.workspace)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
