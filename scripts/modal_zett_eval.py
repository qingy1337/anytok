"""Evaluate the 1k checkpoint grid on two independent Modal profiles."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import modal

ROOT = Path(__file__).resolve().parents[1]
Z = ROOT / 'zett'
app = modal.App('anytok-zett-grid-eval')
vol = modal.Volume.from_name('anytok-zett-grid-eval', create_if_missing=True, version=2)
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
    .add_local_file(str(ROOT / 'scripts/modal_zett_eval_worker.py'), '/opt/zett/eval_worker.py')
    .add_local_file(str(Z / 'output_joint_smollm2_32k_zett_main/expanded_eval_32/tokenizers.json.gz'), '/assets/tokenizers.json.gz')
    .add_local_file(str(Z / 'output_joint_smollm2_32k_zett_main/fineweb_edu_50b_10mb/corpus.jsonl.gz'), '/assets/corpus.jsonl.gz')
    .add_local_dir(str(Z / 'output_joint_smollm2_32k_zett_retrain_15k/fineweb_edu_next128/panel'), '/assets/panel')
    .add_local_dir(str(ROOT / 'modal_results/tokenizers-specialty-48k'), '/assets/specialty')
    .add_local_dir(str(ROOT / 'modal_results/tokenizers-base-64k/base'), '/assets/base64'))


@app.function(image=image, gpu='H200', volumes={'/results': vol}, timeout=4*3600)
def evaluate(steps):
    from huggingface_hub import hf_hub_download, snapshot_download
    os.environ['PYTHONPATH'] = '/opt/zett'
    repo = 'qingy2024/anytok-zett-smollm2-20k'
    work = Path('/work')
    work.mkdir(exist_ok=True)
    for name in ('tokenizers.json.gz', 'corpus.jsonl.gz', 'panel', 'specialty', 'base64'):
        (work / name).symlink_to(Path('/assets') / name, target_is_directory=name in ('panel', 'specialty', 'base64'))
    base_root = work / 'base_root/modal_results/tokenizers-specialty-48k'
    base_root.mkdir(parents=True, exist_ok=True)
    (base_root / 'base').symlink_to(Path('/assets/specialty/base'), target_is_directory=True)
    model = Path(snapshot_download('HuggingFaceTB/SmolLM2-1.7B', revision='effd688a12921b4cc83e3312b6feb579f70f9c71',
                                   allow_patterns=['config.json', 'model.safetensors', 'tokenizer.json']))
    # Match the exact three-file model snapshot used to create the saved panel.
    model_dir = work / 'model'
    model_dir.mkdir(exist_ok=True)
    for name in ('config.json', 'model.safetensors', 'tokenizer.json'):
        (model_dir / name).symlink_to(model / name)
    run = work / 'run'
    run.mkdir(exist_ok=True)
    (run / 'run_config.json').write_text(json.dumps({'model': str(work / 'model')}))
    for step in steps:
        out = Path('/results') / f'step-{step:06d}'
        if (out / 'complete.json').is_file():
            continue
        name = f'step-{step:06d}'
        prefix = f'checkpoints/{name}'
        manifest_path = hf_hub_download(repo, f'{prefix}/manifest.json')
        manifest = json.loads(Path(manifest_path).read_text())
        if manifest['step'] != step:
            raise ValueError('Checkpoint manifest step mismatch')
        target = run / name
        target.mkdir(exist_ok=True)
        for file in ('hypernet/config.json', 'hypernet/model.safetensors', 'lm.pt'):
            src = Path(hf_hub_download(repo, f'{prefix}/{file}'))
            expected = manifest['files'][file]
            h = hashlib.sha256()
            with src.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 << 20), b''):
                    h.update(chunk)
            if src.stat().st_size != expected['bytes'] or h.hexdigest() != expected['sha256']:
                raise ValueError(f'Checkpoint file mismatch: {file}')
            dest = target / file
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(src)
        (target / 'checksums.json').write_text('{}')
        print(json.dumps({'stage':'evaluating', 'step':step}), flush=True)
        subprocess.run(['python', '/opt/zett/eval_worker.py', str(step)], check=True, env=os.environ)
        (out / 'complete.json').write_text(json.dumps({'step':step, 'manifest_sha256':hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(), 'indices':[0,4,8,12,16,20,24,28]})+'\n')
        vol.commit()
        shutil.rmtree(target)
        print(json.dumps({'stage':'checkpoint_complete', 'step':step}), flush=True)


@app.local_entrypoint()
def main(steps: str):
    profile = os.environ.get('MODAL_PROFILE')
    if profile not in ('qingy1337', 'code-group2-evcc'):
        raise ValueError('Use the requested evaluation profiles')
    selected = [int(x) for x in steps.split(',')]
    if not selected or any(x < 1000 or x > 20000 or x % 1000 for x in selected):
        raise ValueError('Expected 1000-step checkpoints')
    evaluate.remote(selected)
