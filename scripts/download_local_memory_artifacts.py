"""User-approved exact local diagnostic copy; never modifies cloud originals."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import modal
import requests
from modal._utils.async_utils import synchronizer
from modal_proto import api_pb2
from modal_tokenizer_pool_transfer import source_client
from pool_generalization import save_json, file_sha

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'modal_results/local_memory'
DEST = STATE / 'artifacts'
BLOCK = 8 * 2**20


async def _block_urls(volume, path):
    await volume.hydrate()
    response = await asyncio.wait_for(volume.client.stub.VolumeGetFile2(
        api_pb2.VolumeGetFile2Request(volume_id=volume.object_id,path=path)), timeout=30)
    return list(response.get_urls),response.size


block_urls = synchronizer.create_blocking(_block_urls)


def resume_blocks(volume, remote, partial, size, force_all=False):
    """Repair missing sparse blocks with deadlines; verify whole-file SHA later.

    Modal SDK 1.5.4's HTTP session has no read deadline. Retain all verified
    files and all already-written blocks rather than restarting multi-GB files.
    """
    urls,remote_size=block_urls(volume,remote)
    if remote_size!=size or len(urls)!=math.ceil(size/BLOCK):raise ValueError('Source block layout changed')
    with partial.open('r+b' if partial.exists() else 'w+b') as handle:
        fd=handle.fileno();os.ftruncate(fd,size)
        missing=[]
        for index in range(len(urls)):
            start=index*BLOCK;end=min(start+BLOCK,size)
            try:hole=os.lseek(fd,start,os.SEEK_HOLE)
            except OSError:hole=start
            if force_all or hole<end:missing.append(index)
        # An all-allocated partial may still contain an incomplete small write.
        if not missing:missing=list(range(len(urls)))
        print(json.dumps(dict(stage='repairing_download_blocks',path=remote,blocks=len(missing),total_blocks=len(urls))),flush=True)
        def fetch(index):
            for attempt in range(5):
                try:
                    url=urls[index] if attempt==0 else block_urls(volume,remote)[0][index]
                    response=requests.get(url,timeout=(10,30));response.raise_for_status();data=response.content
                    if len(data)!=min(BLOCK,size-index*BLOCK):raise ValueError('Incomplete block')
                    written=0
                    while written<len(data):written+=os.pwrite(fd,data[written:],index*BLOCK+written)
                    return
                except Exception:
                    if attempt==4:raise RuntimeError('Block download failed: '+remote+' block '+str(index)) from None
                    time.sleep(min(8,2**attempt))
        with ThreadPoolExecutor(max_workers=8) as executor:list(executor.map(fetch,missing))


def event(stage, **fields):
    row = dict(at_utc=datetime.now(timezone.utc).isoformat(), stage=stage, completed=0, total=9,
        platform='Local · RTX PRO 6000 96 GB', workers=[dict(id='artifact-copy', profile='local',
        platform='Local · download', state='Copying approved diagnostic artifacts')], **fields)
    save_json(STATE / 'status.json', row)
    with (STATE / 'events.jsonl').open('a') as stream:stream.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def main():
    STATE.mkdir(parents=True, exist_ok=True);DEST.mkdir(parents=True, exist_ok=True)
    client, _ = source_client('queues')
    volume = modal.Volume.from_name('anytok-pool-rl-20k', version=2, client=client)
    audit_raw = b''.join(volume.read_file('input/rl-input-audit.json'));audit = json.loads(audit_raw)
    if audit['tokenizers'] != 1000:raise ValueError('The original full pool is required')
    event('downloading exact approved inputs', files_completed=0, files_total=len(audit['files']))

    def copy(remote, relative, expected):
        path = DEST / relative;path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.stat().st_size == expected['bytes'] and file_sha(path) == expected['sha256']:
            return dict(path=relative, **expected)
        for attempt in range(4):
            partial = path.with_suffix(path.suffix + '.partial')
            try:
                if expected['bytes']>64*2**20:
                    resume_blocks(volume,remote,partial,expected['bytes'],force_all=attempt>0)
                else:
                    with partial.open('wb') as target:
                        async def small():
                            await asyncio.wait_for(volume._read_file_into_fileobj.aio(remote,target,concurrency=2),timeout=120.)
                        asyncio.run(small())
                if partial.stat().st_size != expected['bytes'] or file_sha(partial) != expected['sha256']:
                    raise ValueError('Downloaded artifact checksum mismatch: ' + remote)
                partial.replace(path)
                return dict(path=relative, **expected)
            except Exception:
                if attempt == 3:raise
                time.sleep(2 ** attempt)

    def group(items, stage):
        rows = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(copy, *item) for item in items]
            for future in as_completed(futures):
                rows.append(future.result())
                if len(rows) % 50 == 0 or len(rows) == len(items):
                    event(stage, files_completed=len(rows), files_total=len(items),
                          verified_bytes=sum(r['bytes'] for r in rows))
        return rows

    rows = group([(r['path'], r['path'], {k:r[k] for k in ['bytes', 'sha256']})
                  for r in audit['files']], 'downloading exact approved inputs')
    (DEST / 'input/rl-input-audit.json').write_bytes(audit_raw)
    candidates = sorted((e.path for e in volume.listdir('regret-run') if '/step-' in e.path), reverse=True)
    selected = None
    for prefix in candidates:
        if prefix.endswith('step-000000'):continue
        try:manifest = json.loads(b''.join(volume.read_file(prefix + '/manifest.json')))
        except FileNotFoundError:continue
        if manifest['kind'] == 'full_restart_checkpoint':selected = prefix;break
    if selected is None:
        selected = 'regret-smoke/step-000003'
        manifest = json.loads(b''.join(volume.read_file(selected + '/manifest.json')))
    if manifest['kind'] != 'full_restart_checkpoint' or not {'lm.pt','optimizer.pt','state.pt','hypernet/model.safetensors'} <= manifest['files'].keys():
        raise ValueError('Exact complete trained model, optimizer and RNG state required')
    checkpoint_rows = group([(selected + '/' + name, 'checkpoint/' + name, expected)
                             for name, expected in manifest['files'].items()], 'downloading complete checkpoint')
    save_json(DEST / 'checkpoint/manifest.json', manifest)
    config = b''.join(volume.read_file(selected.split('/')[0] + '/run_config.json'))
    (DEST / 'checkpoint/run_config.json').write_bytes(config)
    receipt = dict(at_utc=datetime.now(timezone.utc).isoformat(), profile='queues', volume='anytok-pool-rl-20k',
        checkpoint_source=selected, step=manifest['step'], files=rows + checkpoint_rows,
        input_audit_sha256=hashlib.sha256(audit_raw).hexdigest(), tokenizers=1000,
        approval='User explicitly approved an exact local diagnostic copy, retaining cloud originals.')
    save_json(DEST / 'download_receipt.json', receipt)
    event('exact artifacts verified; ready for local trials', files_completed=len(rows)+len(checkpoint_rows),
          files_total=len(rows)+len(checkpoint_rows), checkpoint_source=selected)


if __name__ == '__main__':main()
