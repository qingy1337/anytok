"""Build the pinned accelerated wheel without changing global installations.

    python scripts/build_fast_tokenizers.py --candidate v5 --test

--source may name a local upstream git checkout to avoid downloading source.
Requires cargo and maturin in the current Python environment. No corpora or
trained tokenizer artifacts are read or written by this build command.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'f383101a26663708484cac0727792aad74f78234'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', choices=['v4', 'v5', 'v6'], default='v6')
    parser.add_argument('--source', default='https://github.com/huggingface/tokenizers.git')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    def run(command, **kwargs): return subprocess.run(command, check=True, **kwargs)
    names = ['all-candidate-noise', 'unigram-literal-unk', 'unigram-stack-drop',
             'fast-training-'+args.candidate if args.candidate != 'v4' else 'fast-training']
    patches = [ROOT/'patches'/f'tokenizers-0.22.2-{name}.patch' for name in names]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in patches}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]
    folder = Path('/tmp')/f'anytok-tokenizer-build-{identity}'
    marker = folder/'.anytok-source.json'
    if not folder.exists():
        run(['git', 'clone', '--no-checkout', args.source, str(folder)])
        run(['git', '-C', str(folder), 'checkout', '--detach', REVISION])
        for patch in patches: run(['git', '-C', str(folder), 'apply', str(patch)])
        tracked = subprocess.check_output(['git', '-C', str(folder), 'diff', '--name-only']).decode().splitlines()
        files = {name: hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in tracked}
        marker.write_text(json.dumps({'revision': REVISION, 'patches': hashes, 'files': files}, indent=2)+'\n')
    data = json.loads(marker.read_text())
    if data['patches'] != hashes or data['revision'] != REVISION:
        raise ValueError('Existing source build provenance differs')
    for name, sha in data['files'].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest() != sha:
            raise ValueError('Existing build source was edited: ' + name)
    if args.test:
        run(['cargo', 'test', '--release', '--manifest-path', str(folder/'tokenizers/Cargo.toml'), '--lib'])
    destination = ROOT/'build'/f'fast-{args.candidate}'
    run([sys.executable, '-m', 'maturin', 'build', '--release', '--locked', '--manifest-path',
         str(folder/'bindings/python/Cargo.toml'), '--out', str(destination)])
    wheels = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.glob('*.whl')}
    (destination/'build-manifest.json').write_text(json.dumps({**data, 'wheels': wheels}, indent=2)+'\n')
    print(json.dumps({'candidate': args.candidate, 'output': str(destination), 'wheels': wheels}))


if __name__ == '__main__': main()
