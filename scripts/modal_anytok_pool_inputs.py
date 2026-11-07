"""Install pinned training inputs and the completed pool cloud-to-cloud."""
import hashlib
import json
import os
from pathlib import Path
import uuid
import time

import modal
from grpclib.exceptions import StreamTerminatedError
from anytok_pool_config import PROFILES, VOLUME_NAME

app = modal.App('anytok-pool-20k-inputs')
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)
image = modal.Image.debian_slim(python_version='3.12').add_local_python_source('anytok_pool_config')
POOL = 'pools/1f5d0be379e32101'


def retry_read(operation):
    for attempt in range(4):
        try:
            return operation()
        except (StreamTerminatedError, OSError, TimeoutError, modal.exception.ConnectionError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


@app.function(image=image, volumes={'/work': volume}, max_containers=100, timeout=3600)
def copy_one(item, source_volume):
    client = modal.Client.from_credentials(os.environ['SOURCE_MODAL_TOKEN_ID'],
                                          os.environ['SOURCE_MODAL_TOKEN_SECRET'])
    source = modal.Volume.from_name(source_volume, version=2, client=client)
    relative = Path(item['destination'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Destination must stay inside the mounted Volume')
    destination = Path('/work') / relative
    if (destination.is_file() and destination.stat().st_size == item['bytes']
            and digest(destination) == item['sha256']):
        return {'path': item['destination'], 'reused': True}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.partial')

    def transfer():
        h = hashlib.sha256()
        with temporary.open('wb') as stream:
            for chunk in source.read_file(item['source']):
                stream.write(chunk)
                h.update(chunk)
        if temporary.stat().st_size != item['bytes'] or h.hexdigest() != item['sha256']:
            raise ValueError(f"Input checksum mismatch: {item['destination']}")
    retry_read(transfer)
    temporary.replace(destination)
    volume.commit()
    return {'path': item['destination'], 'reused': False}


def metadata_item(source, path, destination):
    raw = b''.join(source.read_file(path))
    return raw, {'source': path, 'destination': destination, 'bytes': len(raw),
                 'sha256': hashlib.sha256(raw).hexdigest()}


def checkpoint_items(manifest, base):
    step = manifest.get('step')
    names = {'hypernet/config.json', 'hypernet/model.safetensors', 'lm.pt', 'optimizer.pt', 'state.pt'}
    if (manifest.get('kind') != 'full_restart_checkpoint' or not isinstance(step, int)
            or not 1 <= step <= 20000 or set(manifest.get('files', {})) != names
            or base != f'run/step-{step:06d}'):
        raise ValueError('Expected an intact full restart checkpoint')
    return [{'source':f'{base}/{name}', 'destination':f'{base}/{name}', **expected}
            for name, expected in manifest['files'].items()]


def copy_restart(profile):
    from modal_tokenizer_pool_transfer import source_client
    if profile not in PROFILES or profile == os.environ['MODAL_PROFILE']:
        raise ValueError('Distinct enabled checkpoint source required')
    client, secret = source_client(profile)
    source = modal.Volume.from_name(VOLUME_NAME, version=2, client=client)
    entries = list(source.iterdir('run', recursive=True))
    steps = [int(Path(e.path).parent.name.removeprefix('step-')) for e in entries
             if e.path.endswith('/state.pt') and Path(e.path).parent.name.startswith('step-')]
    if not steps:
        raise ValueError('Source has no full restart checkpoint')
    step = max(steps); base = f'run/step-{step:06d}'
    raw, manifest_item = metadata_item(source, base+'/manifest.json', base+'/manifest.json')
    items = checkpoint_items(json.loads(raw), base)
    for name in ['run_config.json','heldout_tokenizers.json','initial_eval.json','events.jsonl']:
        _, item = metadata_item(source, 'run/'+name, 'run/'+name)
        items.append(item)
    worker = copy_one.with_options(secrets=[secret])
    list(worker.starmap([(item,VOLUME_NAME) for item in items],order_outputs=False))
    # Publish the checkpoint manifest only after every file passed its hash check.
    worker.remote(manifest_item,VOLUME_NAME)
    print(json.dumps({'stage':'copied_restart','source':profile,'step':step}),flush=True)


def collect_grid(profiles):
    from modal_tokenizer_pool_transfer import source_client
    selected = {}
    for profile in profiles:
        if profile not in PROFILES:
            raise ValueError('Unknown checkpoint source')
        client, secret = source_client(profile)
        source = modal.Volume.from_name(VOLUME_NAME,version=2,client=client)
        entries = list(source.iterdir('run',recursive=True))
        for entry in entries:
            parent = Path(entry.path).parent.name
            if not entry.path.endswith('/manifest.json') or not parent.startswith('step-'):
                continue
            step = int(parent.removeprefix('step-'))
            if step % 500 or not 500 <= step <= 20000:
                continue
            raw, item = metadata_item(source,entry.path,f'run/step-{step:06d}/manifest.json')
            manifest = json.loads(raw)
            expected_kind = 'full_restart_checkpoint' if step % 2000 == 0 else 'weights_only_checkpoint'
            names = {'hypernet/config.json','hypernet/model.safetensors','lm.pt'}
            if step % 2000 == 0:
                names |= {'optimizer.pt','state.pt'}
            if (manifest['kind'] != expected_kind or manifest['step'] != step
                    or set(manifest.get('files',{})) != names):
                raise ValueError(f'Grid checkpoint needs compaction: {profile}/{step}')
            if step in selected and selected[step][2] != manifest:
                raise ValueError(f'Conflicting checkpoint grid: {step}')
            selected[step] = (secret,item,manifest)
    if set(selected) != set(range(500,20001,500)):
        raise ValueError('Complete 40-checkpoint grid required before collection')
    for step,(secret,item,manifest) in sorted(selected.items()):
        base = f'run/step-{step:06d}'
        worker = copy_one.with_options(secrets=[secret])
        files = [{'source':f'{base}/{name}','destination':f'{base}/{name}',**value}
                 for name,value in manifest['files'].items()]
        list(worker.starmap([(f,VOLUME_NAME) for f in files],order_outputs=False))
        worker.remote(item,VOLUME_NAME)
        print(json.dumps({'stage':'collected_checkpoint','step':step}),flush=True)


@app.local_entrypoint()
def main(checkpoint_source: str = '', collect_from: str = ''):
    from modal_tokenizer_pool_transfer import source_client
    profile = os.environ.get('MODAL_PROFILE')
    if profile not in PROFILES:
        raise ValueError('Explicit authorized training profile required')
    if checkpoint_source and collect_from:
        raise ValueError('Choose checkpoint transfer or final grid collection')
    if checkpoint_source:
        copy_restart(checkpoint_source)
        return
    if collect_from:
        collect_grid(collect_from.split(','))
        return
    client, secret = source_client('projects')
    source = modal.Volume.from_name('anytok-zett-20k', version=2, client=client)
    raw, manifest_item = metadata_item(source, 'input/manifest.json', 'input/manifest.json')
    items = [{'source': f'input/{name}', 'destination': f'input/{name}', **expected}
             for name, expected in json.loads(raw)['files'].items()]
    list(copy_one.with_options(secrets=[secret]).starmap(
        [(item, 'anytok-zett-20k') for item in items], order_outputs=False))
    copy_one.with_options(secrets=[secret]).remote(manifest_item, 'anytok-zett-20k')
    client, secret = source_client('group-cron')
    source = modal.Volume.from_name('anytok-tokenizer-pool', version=2, client=client)
    raw, index_item = metadata_item(source, POOL + '/full-results.json', 'input/pool/full-results.json')
    _, plan_item = metadata_item(source, POOL + '/plan.json', 'input/pool/plan.json')
    index = json.loads(raw)
    if index['completed'] != 1000 or len(index['tokenizers']) != 1000:
        raise ValueError('The complete 1000-tokenizer pool is required')
    items = [{'source': f"{POOL}/tokenizers/{m['config']['index']:04d}/tokenizer.json",
              'destination': f"input/pool/tokenizers/{m['config']['index']:04d}/tokenizer.json",
              'bytes': m['tokenizer_bytes'], 'sha256': m['tokenizer_sha256']}
             for m in index['tokenizers']]
    list(copy_one.with_options(secrets=[secret]).starmap(
        [(item, 'anytok-tokenizer-pool') for item in items], order_outputs=False))
    list(copy_one.with_options(secrets=[secret]).starmap(
        [(item, 'anytok-tokenizer-pool') for item in [plan_item, index_item]], order_outputs=False))
    print(json.dumps({'stage': 'inputs_ready', 'profile': profile, 'tokenizers': 1000,
                      'pool_plan_sha256': index['plan_sha256']}), flush=True)
