"""Launch 20,000 joint updates with a freshly generated Unigram tokenizer per step."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

from package_training_inputs import atomic_json
from run_raw_bpb_training import run_and_monitor, source_files

ROOT = Path(__file__).resolve().parents[1]
TOTAL_UPDATES = 20000
TRAINER = 'scripts/train_joint_smollm_input.py'


def prepare_code(parent):
    files = source_files()
    patched = subprocess.run([
        'patch', '--batch', '--silent', '--fuzz=0', '--output=-', str(files[TRAINER]),
        str(ROOT / 'patches/online-tokenizer-training.patch'),
    ], check=True, stdout=subprocess.PIPE).stdout
    compile(patched, TRAINER, 'exec')
    staged = {name: path.read_bytes() for name, path in files.items()
              if name not in {TRAINER, 'hf_hypernet/settings_policy.py', 'hf_hypernet/pool_sampling.py'}}
    staged[TRAINER] = patched
    for filename in ('open_unigram_policy.py', 'online_tokenizer_training.py'):
        staged['scripts/' + filename] = (ROOT / 'scripts' / filename).read_bytes()
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in staged.items()}
    revision = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    code = parent / ('.online-tokenizer-code-' + revision[:16])
    code.mkdir(parents=True, exist_ok=True)
    with (code / '.stage.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name, data in staged.items():
            target = code / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.read_bytes() != data:
                    raise ValueError(f'Pinned source has changed: {target}')
            else:
                temporary = target.with_suffix(target.suffix + '.tmp')
                temporary.write_bytes(data)
                temporary.replace(target)
        atomic_json(code / 'source-manifest.json', hashes)
    return code


def command(args, code):
    inputs = args.inputs.resolve()
    result = [
        sys.executable, '-u', str(code / TRAINER),
        '--recipe', str(inputs / 'recipe.json'), '--data-dir', str(inputs / 'corpus'),
        '--data-manifest', str(inputs / 'corpus/manifest.json'), '--eval-dir', str(inputs / 'eval'),
        '--output', str(args.output), '--warmup-checkpoint', str(inputs / 'warmup'),
        '--model', str(inputs / 'model'), '--uniform-config', str(inputs / 'uniform_run_config.json'),
        '--eval-tokenizers', str(inputs / 'uniform_heldout_tokenizers.json'),
        '--online-selector', args.selector, '--reward-objective', 'raw_bpb',
        '--microbatch-size', '32', '--memory-execution', 'optimized',
        '--save-every', '500', '--eval-every', '500', '--seed', '42',
        '--stop-at-update', str(args.stop_at_update),
        '--protected-steps', *map(str, range(500, TOTAL_UPDATES + 1, 500)),
    ]
    if args.resume:
        result.append('--resume')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selector', choices=['reinforce', 'predictor', 'random'], default='reinforce')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-at-update', type=int, default=TOTAL_UPDATES,
                        help='Absolute stop after saving a full restart, for a segment or smoke test')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.stop_at_update <= TOTAL_UPDATES:
        parser.error(f'--stop-at-update must be between 1 and {TOTAL_UPDATES}')
    args.output = args.output.resolve()
    required = ('recipe.json', 'corpus/manifest.json', 'eval', 'model/config.json', 'warmup/config.json',
                'uniform_heldout_tokenizers.json', 'uniform_run_config.json')
    for name in required:
        if not (args.inputs / name).exists():
            parser.error('Missing original input: ' + name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (args.output.parent / f'.{args.output.name}.launcher.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('This output directory already has an active launcher')
        code = prepare_code(args.output.parent)
        cmd = command(args, code)
        print(json.dumps(dict(stage='training_command', code=str(code), command=cmd)), flush=True)
        if args.prepare_only:
            return
        if args.resume and not args.output.is_dir():
            parser.error('--resume requires an existing output directory')
        if not args.resume and args.output.is_dir() and any(args.output.iterdir()):
            parser.error('Choose a new output directory or pass --resume')
        if importlib.metadata.version('torch') != '2.15.0+cu132':
            raise RuntimeError('Use the verified PyTorch 2.15.0+cu132 environment')
        importlib.metadata.version('anytok-online-unigram')
        env = {**os.environ, 'TOKENIZERS_PARALLELISM': 'false', 'PYTHONUNBUFFERED': '1'}
        env.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
        env['LD_LIBRARY_PATH'] = os.pathsep.join(filter(None, [sysconfig.get_config_var('LIBDIR'),
                                                              env.get('LD_LIBRARY_PATH')]))
        raise SystemExit(run_and_monitor(cmd, args, env))


if __name__ == '__main__':
    main()
