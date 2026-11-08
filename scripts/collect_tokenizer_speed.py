"""Collect completed benchmark metadata while exact remote calls keep running.

    MODAL_PROFILE=gildmember python scripts/collect_tokenizer_speed.py --run-id RUN_ID --candidate v5

Read-only Modal API calls; no compute, tokenizer files or corpora are downloaded.
This never changes or restarts a remote job, and partial rows are not completion.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re

import modal
from pool_generalization import save_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id', required=True)
    p.add_argument('--candidate', choices=['v4', 'v5', 'v6', 'v7', 'v8'], default='v8')
    a = p.parse_args()
    if os.environ.get('MODAL_PROFILE') != 'gildmember': raise ValueError('Explicit gildmember required')
    if not re.fullmatch('[a-f0-9]{16}(-smoke)?', a.run_id): raise ValueError('Expected exact run ID')
    volume = modal.Volume.from_name('anytok-fast-tokenizers', version=2)
    entries = list(volume.iterdir(a.run_id, recursive=True))
    paths = [e.path for e in entries if e.path.endswith('/complete.json')]
    def read(path):
        row = json.loads(b''.join(volume.read_file(path)))
        if row['provenance']['id'] != a.run_id.removesuffix('-smoke'): raise ValueError('Wrong run provenance')
        return row
    with ThreadPoolExecutor(max_workers=6) as pool: rows = list(pool.map(read, paths))
    state = ROOT/('modal_results/tokenizer_speed'+('_'+a.candidate if a.candidate != 'v4' else ''))
    state.mkdir(parents=True, exist_ok=True)
    save_json(state/f'{a.run_id}-partial.json', rows)
    print(json.dumps({'run_id': a.run_id, 'completed_phases': len(rows), 'expected': 18}))
    for key in sorted({r['case'] for r in rows}):
        pair = {r['variant']: r for r in rows if r['case'] == key}
        if len(pair) == 2:
            b, f = pair['baseline-200mb'], pair['fast-200mb']
            print(json.dumps({'case': key, 'speedup': b['ready_seconds']/f['ready_seconds'],
                              'cpu_speedup': b['manifest']['cpu_seconds']/f['manifest']['cpu_seconds'],
                              'baseline_seconds': b['ready_seconds'], 'fast_seconds': f['ready_seconds'],
                              'roundtrip_failures': f['lossless_audit']['roundtrip_failures']}))


if __name__ == '__main__': main()
