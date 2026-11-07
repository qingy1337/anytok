"""Install pinned training inputs and the completed pool cloud-to-cloud."""
import hashlib
import json
import os
from pathlib import Path
import uuid

import modal
from modal_tokenizer_pool_transfer import source_client, retry_read
from anytok_pool_config import PROFILES, VOLUME_NAME

app = modal.App('anytok-pool-20k-inputs')
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)
image = modal.Image.debian_slim(python_version='3.12')
POOL = 'pools/1f5d0be379e32101'


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
    destination = Path('/work') / item['destination']
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


@app.local_entrypoint()
def main():
    profile = os.environ.get('MODAL_PROFILE')
    if profile not in PROFILES:
        raise ValueError('Explicit authorized training profile required')
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
