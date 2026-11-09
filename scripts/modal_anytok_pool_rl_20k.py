"""Adversarial tokenizer selection, using the original immutable pool and inputs.

Default action is a read-only plan. Prepare and transfer directly between Modal
Volumes; no corpus, tokenizer or checkpoint is downloaded to the local caller.
GPU is B300. No CPU or memory reservation is specified.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import uuid

import modal
from anytok_pool_rl_config import APP_NAME, FINAL_STEP, POLICY, PROFILES, VOLUME_NAME
from pool_generalization import save_json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'runtime/anytok_pool_20k'
SOURCE = ROOT/'runtime/anytok_pool_rl_20k'
MOUNT = Path('/mnt/rl')
CODE = Path('/opt/zett')
STATE = ROOT/'modal_results/anytok_pool_rl_20k'
WEIGHTS = {'hypernet/config.json', 'hypernet/model.safetensors', 'lm.pt', 'policy.pt'}
app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)
copy_image = (modal.Image.debian_slim(python_version='3.12')
              .add_local_python_source('anytok_pool_rl_config', 'pool_generalization'))
image = (modal.Image.from_registry('ubuntu:24.04', add_python='3.12')
    .apt_install('libpython3.12')
    .uv_pip_install('torch==2.15.0', 'transformers==5.15.1', 'tokenizers==0.22.2',
                    'numpy==2.3.5', 'safetensors==0.8.0', 'accelerate==1.14.0')
    .add_local_dir(BASE/'hf_hypernet', str(CODE/'hf_hypernet'))
    .add_local_file(SOURCE/'hf_hypernet/bandit_sampling.py', str(CODE/'hf_hypernet/bandit_sampling.py'))
    .add_local_file(SOURCE/'scripts/train_joint_smollm_input.py', str(CODE/'scripts/train_joint_smollm_input.py'))
    .add_local_file(BASE/'rust_utils/target/release/librust_utils.so', str(CODE/'rust_utils/target/release/librust_utils.so'))
    .add_local_python_source('anytok_pool_rl_config', 'anytok_pool_config', 'pool_generalization'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8<<20), b''): h.update(block)
    return h.hexdigest()


def source_volume(name):
    client = modal.Client.from_credentials(os.environ['SOURCE_MODAL_TOKEN_ID'],
                                           os.environ['SOURCE_MODAL_TOKEN_SECRET'])
    return modal.Volume.from_name(name, version=2, client=client)


@app.function(image=copy_image, volumes={str(MOUNT): volume}, max_containers=100, timeout=7200)
def copy_file(source_name, source_path, destination, expected=None):
    path = Path(destination)
    if path.is_absolute() or '..' in path.parts: raise ValueError('Invalid destination')
    path = MOUNT/path
    if expected and path.is_file() and path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256']:
        return {'path': destination, 'reused': True, **expected}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    h = hashlib.sha256()
    with temporary.open('wb') as stream:
        for block in source_volume(source_name).read_file(source_path):
            stream.write(block); h.update(block)
    actual = {'bytes': temporary.stat().st_size, 'sha256': h.hexdigest()}
    if expected and actual != expected: raise ValueError(f'Cloud copy checksum mismatch: {destination}')
    temporary.replace(path); volume.commit()
    return {'path': destination, 'reused': False, **actual}


def copy_worker():
    return copy_file.with_options(secrets=[modal.Secret.from_dict({
        key: os.environ[key] for key in ['SOURCE_MODAL_TOKEN_ID', 'SOURCE_MODAL_TOKEN_SECRET']})])


@app.function(image=copy_image, volumes={str(MOUNT): volume}, timeout=7200)
def prepare_inputs():
    source = source_volume('anytok-pool-20k')
    manifest = json.loads(b''.join(source.read_file('input/manifest.json')))
    index = json.loads(b''.join(source.read_file('input/pool/full-results.json')))
    if index['completed'] != 1000 or len(index['tokenizers']) != 1000:
        raise ValueError('Complete original pool required')
    items = [('anytok-pool-20k', 'input/'+name, 'input/'+name, expected)
             for name, expected in manifest['files'].items()]
    items += [('anytok-pool-20k', f"input/pool/tokenizers/{r['config']['index']:04d}/tokenizer.json",
               f"input/pool/tokenizers/{r['config']['index']:04d}/tokenizer.json",
               {'bytes': r['tokenizer_bytes'], 'sha256': r['tokenizer_sha256']}) for r in index['tokenizers']]
    items += [('anytok-pool-20k', name, name, None) for name in
              ['input/manifest.json', 'input/pool/full-results.json', 'input/pool/plan.json']]
    items += [('anytok-pool-20k', 'run/heldout_tokenizers.json', 'input/uniform_heldout_tokenizers.json', None)]
    items += [('anytok-pool-20k', 'run/run_config.json', 'input/uniform_run_config.json', None)]
    results = []
    for row in copy_worker().starmap(items, order_outputs=False):
        results.append(row)
        if len(results) % 100 == 0: print(json.dumps({'stage':'copying_inputs', 'completed':len(results),'total':len(items)}),flush=True)
    volume.reload()
    save_json(MOUNT/'input/rl-input-audit.json', {'files':results, 'pool_plan_sha256':index['plan_sha256'],
              'pool_index_sha256':sha(MOUNT/'input/pool/full-results.json'), 'tokenizers':1000})
    volume.commit()
    return {'files':len(results), 'tokenizers':1000, 'bytes':sum(r['bytes'] for r in results)}


def verify_checkpoint(run, step, require_restart=False):
    folder = run/f'step-{step:06d}'
    manifest = json.loads((folder/'manifest.json').read_text())
    full = manifest['kind'] == 'full_restart_checkpoint'
    if manifest['kind'] not in ['full_restart_checkpoint','weights_only_checkpoint'] or manifest['step'] != step:
        raise ValueError('Invalid checkpoint manifest')
    if require_restart and not full: raise ValueError('Full restart required')
    names = WEIGHTS | ({'optimizer.pt','state.pt'} if full else set())
    if set(manifest['files']) != names: raise ValueError('Checkpoint file set differs')
    for name, expected in manifest['files'].items():
        path = folder/name
        if not path.is_file() or path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
            raise ValueError(f'Checkpoint integrity failed: {name}')
    return manifest


@app.function(image=copy_image, volumes={str(MOUNT): volume}, timeout=7200)
def transfer_restart(step, grid_checkpoint=False):
    source = source_volume(VOLUME_NAME)
    steps = [step]
    worker = copy_worker()
    for current in steps:
        prefix = f'run/step-{current:06d}'
        raw = b''.join(source.read_file(prefix+'/manifest.json'))
        manifest = json.loads(raw)
        names = WEIGHTS | ({'optimizer.pt','state.pt'} if manifest['kind']=='full_restart_checkpoint' else set())
        if (manifest['step'] != current or set(manifest['files']) != names
                or (not grid_checkpoint and manifest['kind'] != 'full_restart_checkpoint')):
            raise ValueError('Invalid source checkpoint')
        items = [(VOLUME_NAME, prefix+'/'+name, prefix+'/'+name, expected)
                 for name, expected in manifest['files'].items()]
        list(worker.starmap(items, order_outputs=False))
        worker.remote(VOLUME_NAME, prefix+'/manifest.json', prefix+'/manifest.json',
                      {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    if not grid_checkpoint:
        for name in ['run_config.json','heldout_tokenizers.json','initial_eval.json','events.jsonl']:
            worker.remote(VOLUME_NAME, 'run/'+name, 'run/'+name)
    volume.reload()
    verify_checkpoint(MOUNT/'run', step, not grid_checkpoint)
    return {'copied_steps':steps}


@app.function(image=image, gpu='B300', volumes={str(MOUNT):volume}, timeout=4*3600)
def train_segment(max_seconds=12600, stop_step=FINAL_STEP, smoke=False):
    import torch
    if not 60 <= max_seconds <= 12600 or not 1 <= stop_step <= FINAL_STEP:
        raise ValueError('Invalid bounded segment')
    if 'B300' not in torch.cuda.get_device_name(0): raise RuntimeError('Expected B300')
    sys.path.insert(0, str(CODE))
    import scripts.train_joint_smollm_input as trainer
    run = MOUNT/('smoke' if smoke else 'run')
    run.mkdir(exist_ok=True)
    existing = sorted(p for p in run.glob('step-*') if (p/'state.pt').is_file())
    if existing:
        last = int(existing[-1].name[5:])
        verify_checkpoint(run, last, True)
        config_path = run/'run_config.json'
        config = json.loads(config_path.read_text())
        # Only the backing Volume ID changes between workspaces.
        new_path = str((MOUNT/'input/warmup').resolve())
        if config['warmup_checkpoint'] != new_path:
            if not config['warmup_checkpoint'].endswith('/input/warmup') or not new_path.endswith('/input/warmup'):
                raise ValueError('Unexpected warmup path')
            config['warmup_checkpoint'] = new_path
            save_json(config_path, config)
        if last >= stop_step:
            state = torch.load(existing[-1]/'state.pt',map_location='cpu',weights_only=True)
            if state['bandit']['updates'] != last: raise ValueError('Policy/model counter mismatch')
            return {'step':last,'policy_updates':state['bandit']['updates'],'already_complete':True}
        # Replay from the last full restart, preserving any later incomplete
        # trajectory's weights separately rather than overwriting grid files.
        import shutil
        for future in sorted(run.glob('step-*')):
            if int(future.name[5:]) > last:
                archive = run/'interrupted'/str(time.time_ns())/future.name
                archive.parent.mkdir(parents=True,exist_ok=True)
                shutil.move(str(future),str(archive))
    input_audit = json.loads((MOUNT/'input/rl-input-audit.json').read_text())
    if input_audit['tokenizers'] != 1000: raise ValueError('Wrong tokenizer pool')
    original_save = trainer.save_checkpoint
    def save_and_commit(output, step, *args, **kwargs):
        original_save(output, step, *args, **kwargs)
        folder = Path(output)/f'step-{step:06d}'
        files = {p.relative_to(folder).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)}
                 for p in folder.rglob('*') if p.is_file()}
        full = 'state.pt' in files
        save_json(folder/'manifest.json', {'step':step, 'kind':'full_restart_checkpoint' if full else 'weights_only_checkpoint',
                  'files':files, 'trainer_sha256':sha(CODE/'scripts/train_joint_smollm_input.py'),
                  'policy_sha256':sha(CODE/'hf_hypernet/bandit_sampling.py')})
        volume.commit()
        if full:
            # A newer durable restart makes non-milestone restart state obsolete.
            import shutil
            for old in sorted(run.glob('step-*')):
                old_step = int(old.name[5:])
                if old_step >= step or (old_step and old_step % 2000 == 0) or not (old/'state.pt').is_file(): continue
                if old_step == 0 or old_step % 500:
                    shutil.rmtree(old)
                else:
                    prior = json.loads((old/'manifest.json').read_text())
                    prior.update(kind='weights_only_checkpoint', files={n:prior['files'][n] for n in WEIGHTS})
                    save_json(old/'manifest.json',prior)
                    (old/'state.pt').unlink(); (old/'optimizer.pt').unlink()
            volume.commit()
        print(json.dumps({'stage':'committed_checkpoint','step':step,'full_restart':full}),flush=True)
    trainer.save_checkpoint = save_and_commit
    argv = [str(CODE/'scripts/train_joint_smollm_input.py'),
        '--recipe',str(MOUNT/'input/recipe.json'),'--data-dir',str(MOUNT/'input/corpus'),
        '--data-manifest',str(MOUNT/'input/corpus/manifest.json'),'--eval-dir',str(MOUNT/'input/eval'),
        '--output',str(run),'--warmup-checkpoint',str(MOUNT/'input/warmup'),
        '--model',str(MOUNT/'input/model'),'--tokenizer-pool',str(MOUNT/'input/pool'),
        '--eval-tokenizers',str(MOUNT/'input/uniform_heldout_tokenizers.json'),
        '--microbatch-size','32','--save-every','500','--eval-every','500',
        '--stop-at-update',str(stop_step),'--seed','42']
    if not smoke: argv += ['--protected-steps', *map(str,range(500,FINAL_STEP+1,500))]
    if existing: argv += ['--resume']
    timer = threading.Timer(max_seconds, lambda: os.kill(os.getpid(),signal.SIGTERM))
    timer.daemon = True
    sys.argv = argv
    timer.start()
    try: trainer.main()
    finally: timer.cancel(); volume.commit()
    restarts = sorted(p for p in run.glob('step-*') if (p/'state.pt').is_file())
    last = int(restarts[-1].name[5:])
    verify_checkpoint(run,last,True)
    state = torch.load(restarts[-1]/'state.pt',map_location='cpu',weights_only=True)
    if state['bandit']['updates'] != last or int(state['bandit']['counts'].sum()) != last:
        raise ValueError('Policy did not resume with the model')
    return {'step':last,'policy_updates':state['bandit']['updates'],'gpu':torch.cuda.get_device_name(0)}


@app.function(image=copy_image, volumes={str(MOUNT):volume}, timeout=7200)
def audit_final():
    manifests = [verify_checkpoint(MOUNT/'run',step,step%2000==0) for step in range(500,FINAL_STEP+1,500)]
    receipt = {'checkpoints':len(manifests),'final_step':FINAL_STEP,
               'bytes':sum(f['bytes'] for m in manifests for f in m['files'].values())}
    save_json(MOUNT/'run/final_audit.json',receipt); volume.commit()
    return receipt


@app.function(image=copy_image, volumes={str(MOUNT):volume}, timeout=3600)
def compact_handoff(step):
    if step % 2000 == 0: return {'step':step,'compacted':False}
    run = MOUNT/'run'
    manifest = verify_checkpoint(run,step,True)
    folder = run/f'step-{step:06d}'
    manifest.update(kind='weights_only_checkpoint',files={name:manifest['files'][name] for name in WEIGHTS})
    save_json(folder/'manifest.json',manifest)
    (folder/'state.pt').unlink(); (folder/'optimizer.pt').unlink()
    volume.commit()
    return {'step':step,'compacted':True}


@app.local_entrypoint()
def main(action: str='plan', source_profile: str='gildmember', step: int=0,
         max_seconds: int=12600, resume_call: str='', grid_checkpoint: bool=False):
    profile = os.environ.get('MODAL_PROFILE')
    if profile not in PROFILES: raise ValueError('Explicit enabled RL profile required')
    if action == 'plan':
        print(json.dumps({'profile':profile,'volume':VOLUME_NAME,'gpu':'B300','policy':POLICY,
                          'steps':FINAL_STEP,'tokenizers':1000,'positions':128,
                          'weights_every':500,'restart_every':2000,'cpu_request':None,'memory_request':None},indent=2)); return
    STATE.mkdir(parents=True,exist_ok=True)
    if resume_call:
        call = modal.FunctionCall.from_id(resume_call)
        original_pointer = STATE/f'{profile}-{action}-call.json'
        original_app = json.loads(original_pointer.read_text()).get('app_id') if original_pointer.exists() else None
    else:
        if action in ['prepare','transfer']:
            from modal_tokenizer_pool_transfer import source_client
            _,secret = source_client(source_profile)
            if action == 'prepare': call = prepare_inputs.with_options(secrets=[secret]).spawn()
            else: call = transfer_restart.with_options(secrets=[secret]).spawn(step, grid_checkpoint)
        elif action == 'smoke': call = train_segment.spawn(900,step or 2,True)
        elif action == 'train': call = train_segment.spawn(max_seconds,FINAL_STEP,False)
        elif action == 'audit-final': call = audit_final.spawn()
        elif action == 'compact': call = compact_handoff.spawn(step)
        else: raise ValueError('Unknown action')
        pointer = {'at_utc':datetime.now(timezone.utc).isoformat(),'action':action,'profile':profile,
                   'call_id':call.object_id,'step':step,'max_seconds':max_seconds,
                   'source_profile':source_profile,'grid_checkpoint':grid_checkpoint,
                   'app_id':app.app_id}
        save_json(STATE/f'{profile}-{action}-{time.time_ns()}-call.json',pointer)
        save_json(STATE/f'{profile}-{action}-call.json',pointer)
        print(json.dumps({'stage':'submitted',**pointer}),flush=True)
    while True:
        try: result=call.get(timeout=10); break
        except (TimeoutError,modal.exception.TimeoutError): pass
    save_json(STATE/f'{profile}-{action}-result.json',result)
    print(json.dumps({'stage':'result','action':action,'profile':profile,'result':result}),flush=True)
    if resume_call and original_app and original_app != app.app_id:
        # A killed detached caller can leave its completed GPU container idle.
        # The exact root call has returned, so stopping its source app is safe.
        import subprocess
        subprocess.run(['modal','app','stop','-y',original_app],check=True)
