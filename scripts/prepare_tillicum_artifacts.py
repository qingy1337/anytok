"""Prepare private signed metadata; artifact bytes go Modal -> Tillicum directly."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import modal
from download_local_memory_artifacts import block_urls
from modal_tokenizer_pool_transfer import source_client
from pool_generalization import file_sha,save_json

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'modal_results/tillicum_smoke'
ARTIFACTS=ROOT/'modal_results/local_memory/artifacts'


def main():
    STATE.mkdir(parents=True,exist_ok=True)
    receipt=json.loads((ARTIFACTS/'download_receipt.json').read_text())
    client,_=source_client(receipt['profile'])
    volume=modal.Volume.from_name(receipt['volume'],version=2,client=client)
    rows=[]
    for item in receipt['files']:
        path=item['path']
        remote=receipt['checkpoint_source']+'/'+path.removeprefix('checkpoint/') if path.startswith('checkpoint/') else path
        rows.append(dict(path=path,remote=remote,bytes=item['bytes'],sha256=item['sha256']))
    for path,remote in [('input/rl-input-audit.json','input/rl-input-audit.json'),
                        ('checkpoint/run_config.json','regret-run/run_config.json')]:
        local=ARTIFACTS/path
        rows.append(dict(path=path,remote=remote,bytes=local.stat().st_size,sha256=file_sha(local)))
    remote=receipt['checkpoint_source']+'/manifest.json'
    raw=b''.join(volume.read_file(remote))  # Small checkpoint metadata only.
    rows.append(dict(path='checkpoint/manifest.json',remote=remote,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    def signed(item):
        urls,size=block_urls(volume,item['remote'])
        if size!=item['bytes']:raise ValueError('Source size changed: '+item['remote'])
        return {**item,'urls':urls}
    with ThreadPoolExecutor(max_workers=12) as pool:files=list(pool.map(signed,rows))
    target=STATE/'signed-files.json'
    save_json(target,dict(profile=receipt['profile'],volume=receipt['volume'],checkpoint_source=receipt['checkpoint_source'],block_bytes=8*2**20,files=files))
    target.chmod(0o600)
    save_json(STATE/'download_receipt.json',receipt)
    print(json.dumps(dict(files=len(files),bytes=sum(r['bytes'] for r in files),metadata_path=str(target))),flush=True)


if __name__=='__main__':main()
