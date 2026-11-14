"""Run on Tillicum login: bounded parallel direct downloads and whole-file hashes."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import requests

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'modal_results/tillicum_smoke'
ARTIFACTS=ROOT/'modal_results/local_memory/artifacts'
THREAD=threading.local()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(8<<20),b''):h.update(data)
    return h.hexdigest()


def main():
    manifest_path=ROOT/'.metadata/signed-files.json';manifest=json.loads(manifest_path.read_text())
    ARTIFACTS.mkdir(parents=True,exist_ok=True);block=manifest['block_bytes'];jobs=[];pending=[];verified=[]
    total=sum(r['bytes'] for r in manifest['files']);completed_bytes=0
    for r in manifest['files']:
        path=ARTIFACTS/r['path'];path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.stat().st_size==r['bytes'] and sha(path)==r['sha256']:
            verified.append(r['path']);completed_bytes+=r['bytes'];continue
        partial=path.with_suffix(path.suffix+'.partial')
        with partial.open('wb') as f:f.truncate(r['bytes'])
        if len(r['urls'])!=(r['bytes']+block-1)//block:raise ValueError('Unexpected source block layout')
        pending.append((r,path,partial))
        jobs.extend((r['path'],partial,i,url,min(block,r['bytes']-i*block)) for i,url in enumerate(r['urls']))
    def status(stage,**fields):
        row=dict(at_utc=datetime.now(timezone.utc).isoformat(),stage=stage,verified_files=len(verified),
                 total_files=len(manifest['files']),completed_bytes=completed_bytes,total_bytes=total,**fields)
        save(STATE/'transfer-status.json',row);print(json.dumps(row),flush=True)
    def fetch(job):
        relative,partial,index,url,length=job
        if not hasattr(THREAD,'session'):THREAD.session=requests.Session()
        for attempt in range(5):
            try:
                response=THREAD.session.get(url,timeout=(15,90));response.raise_for_status();data=response.content
                if len(data)!=length:raise ValueError('Incomplete response')
                fd=os.open(partial,os.O_WRONLY)
                try:
                    offset=0
                    while offset<len(data):offset+=os.pwrite(fd,data[offset:],index*block+offset)
                finally:os.close(fd)
                return length
            except Exception:
                if attempt==4:raise RuntimeError(f'Download failed: {relative}, block {index}') from None
                time.sleep(min(8,2**attempt))
    status('direct cloud download')
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures=[pool.submit(fetch,j) for j in jobs];last=time.monotonic()
        for future in as_completed(futures):
            completed_bytes+=future.result()
            if time.monotonic()-last>=5:status('direct cloud download');last=time.monotonic()
    status('verifying complete source hashes')
    def verify(item):
        r,path,partial=item
        if partial.stat().st_size!=r['bytes'] or sha(partial)!=r['sha256']:
            raise ValueError('Artifact checksum mismatch: '+r['path'])
        partial.replace(path);return r['path']
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name in pool.map(verify,pending):verified.append(name)
    receipt=json.loads((ROOT/'.metadata/download_receipt.json').read_text())
    save(ARTIFACTS/'download_receipt.json',receipt)
    save(STATE/'artifact-receipt.json',dict(source_profile=manifest['profile'],source_volume=manifest['volume'],
        checkpoint_source=manifest['checkpoint_source'],verified_files=len(verified),verified_bytes=total,
        tokenizers=1000,transport='Direct signed block downloads from Modal to Tillicum; no artifact payload through local caller'))
    status('complete');manifest_path.unlink()


if __name__=='__main__':main()
