"""Run matched-context and 5k/20k hypernetwork/LM swap diagnostics."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import modal

ROOT = Path(__file__).resolve().parents[1]
Z = ROOT / 'zett'
PANELS = Z / 'output_joint_smollm2_32k_zett_retrain_15k/fineweb_edu_length_control'
INDICES = [0, 4, 8, 12, 16, 20, 24, 28]
PAIRS = [(5000, 5000), (10000, 10000), (20000, 20000), (5000, 20000), (20000, 5000)]
app = modal.App('anytok-zett-length-swap')
vol = modal.Volume.from_name('anytok-zett-length-swap', create_if_missing=True, version=2)
image = (modal.Image.from_registry('ubuntu:24.04', add_python='3.12')
    .apt_install('libpython3.12', 'ca-certificates')
    .uv_pip_install('torch==2.15.0', 'transformers==5.15.1', 'tokenizers==0.22.2',
                    'numpy==2.3.5', 'safetensors==0.8.0', 'accelerate==1.14.0',
                    'huggingface_hub==1.28.0')
    .add_local_dir(str(Z / 'hf_hypernet'), '/opt/zett/hf_hypernet')
    .add_local_dir(str(Z / 'scripts'), '/opt/zett/scripts')
    .add_local_file(str(ROOT / 'scripts/evaluate_joint_external_panel.py'), '/opt/zett/scripts/evaluate_joint_external_panel.py')
    .add_local_file(str(ROOT / 'runtime/zett_20k/rust_utils/target/release/librust_utils.so'),
                    '/opt/zett/rust_utils/target/release/librust_utils.so')
    .add_local_file(str(Z / 'output_joint_smollm2_32k_zett_main/expanded_eval_32/tokenizers.json.gz'),
                    '/assets/tokenizers.json.gz'))
for cap in (128, 512, 2048, 8192):
    for name in ('corpus.jsonl.gz', 'examples.jsonl.gz', 'examples.jsonl.summary.json'):
        image = image.add_local_file(str(PANELS / f'positions-{cap:04d}' / name),
                                     f'/length/positions-{cap:04d}/{name}')


@app.function(image=image, gpu='H200', volumes={'/results': vol}, timeout=4 * 3600)
def evaluate(caps):
    from huggingface_hub import hf_hub_download, snapshot_download

    os.environ['PYTHONPATH'] = '/opt/zett'
    work = Path('/work')
    work.mkdir(exist_ok=True)
    model = Path(snapshot_download('HuggingFaceTB/SmolLM2-1.7B',
        revision='effd688a12921b4cc83e3312b6feb579f70f9c71',
        allow_patterns=['config.json', 'model.safetensors', 'tokenizer.json']))
    model_dir = work / 'model'
    model_dir.mkdir(exist_ok=True)
    for name in ('config.json', 'model.safetensors', 'tokenizer.json'):
        (model_dir / name).symlink_to(model / name)
    run = work / 'run'
    run.mkdir(exist_ok=True)
    (run / 'run_config.json').write_text(json.dumps({'model': str(model_dir)}))
    checkpoints = {}
    repo = 'qingy2024/anytok-zett-smollm2-20k'
    for step in (5000, 10000, 20000):
        prefix = f'checkpoints/step-{step:06d}'
        manifest_path = Path(hf_hub_download(repo, f'{prefix}/manifest.json'))
        manifest = json.loads(manifest_path.read_text())
        if manifest['step'] != step:
            raise ValueError('Checkpoint step mismatch')
        files = {}
        for name in ('lm.pt', 'hypernet/config.json', 'hypernet/model.safetensors'):
            path = Path(hf_hub_download(repo, f'{prefix}/{name}'))
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 << 20), b''):
                    digest.update(chunk)
            expected = manifest['files'][name]
            if path.stat().st_size != expected['bytes'] or digest.hexdigest() != expected['sha256']:
                raise ValueError(f'Checkpoint hash mismatch: {step}/{name}')
            files[name] = path
        checkpoints[step] = {'files': files, 'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    for cap in caps:
        panel = Path(f'/length/positions-{cap:04d}')
        for hn_step, lm_step in PAIRS:
            name = f'hn-{hn_step:06d}_lm-{lm_step:06d}'
            output = Path('/results') / f'positions-{cap:04d}' / name
            if (output / 'complete.json').exists():
                continue
            hybrid = run / f'positions-{cap:04d}_{name}'
            hybrid.mkdir(exist_ok=True)
            (hybrid / 'hypernet').mkdir(exist_ok=True)
            for file, source_step in (('lm.pt', lm_step), ('hypernet/config.json', hn_step),
                                      ('hypernet/model.safetensors', hn_step)):
                (hybrid / file).symlink_to(checkpoints[source_step]['files'][file])
            print(json.dumps({'stage': 'evaluating', 'positions': cap,
                              'hypernetwork_step': hn_step, 'lm_step': lm_step}), flush=True)
            subprocess.run(['python', '/opt/zett/scripts/evaluate_joint_external_panel.py',
                '--run', str(run), '--checkpoint', str(hybrid),
                '--corpus', str(panel / 'corpus.jsonl.gz'),
                '--examples', str(panel / 'examples.jsonl.gz'),
                '--specs', '/assets/tokenizers.json.gz', '--output', str(output),
                '--sources', 'fineweb-edu', '--indices', *map(str, INDICES),
                '--method', 'matched 256 documents and 32 canonical continuation tokens; '
                            'shared raw prefix under position cap; swapped hypernetwork and LM'], check=True)
            (output / 'complete.json').write_text(json.dumps({
                'position_cap': cap, 'hypernetwork_step': hn_step, 'lm_step': lm_step,
                'indices': INDICES,
                'hypernetwork_manifest_sha256': checkpoints[hn_step]['manifest_sha256'],
                'lm_manifest_sha256': checkpoints[lm_step]['manifest_sha256']}, indent=2) + '\n')
            vol.commit()
            print(json.dumps({'stage': 'complete', 'positions': cap,
                              'hypernetwork_step': hn_step, 'lm_step': lm_step}), flush=True)


@app.local_entrypoint()
def main(caps: str):
    profile = os.environ.get('MODAL_PROFILE')
    if profile not in ('qingy1337', 'code-group2-evcc'):
        raise ValueError('Use the chosen evaluation profiles')
    values = [int(x) for x in caps.split(',')]
    if not values or not set(values) <= {128, 512, 2048, 8192}:
        raise ValueError('Unexpected position caps')
    evaluate.remote(values)
