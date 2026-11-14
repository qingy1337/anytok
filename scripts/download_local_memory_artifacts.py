"""User-approved exact local diagnostic copy; never modifies cloud originals."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import modal
from modal_tokenizer_pool_transfer import source_client
from pool_generalization import save_json, file_sha

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'modal_results/local_memory'
DEST = STATE / 'artifacts'


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
                async def bounded_download(target):
                    # The SDK otherwise prefetches os.cpu_count() blocks per
                    # file. Bound aggregate connections and stalled requests.
                    large = expected['bytes'] > 64 * 2**20
                    await asyncio.wait_for(volume._read_file_into_fileobj.aio(remote, target, concurrency=8 if large else 2),
                        timeout=3600. if large else 120.)
                with partial.open('wb') as target:asyncio.run(bounded_download(target))
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
