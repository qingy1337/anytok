"""Audit canonical seams in the original full-pool evaluation; no model inference."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import modal
from pool_generalization import digest, file_sha, save_json

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'modal_results/original_pool_boundary_audit'
PREFIX = 'original-pool-boundary-audit-v1'
PROFILE = 'blackpantherx86'
app = modal.App('anytok-original-pool-boundary-audit')
volume = modal.Volume.from_name('anytok-rl-all-pool-20k', version=2)
image = (modal.Image.debian_slim(python_version='3.12')
    .uv_pip_install('tokenizers==0.22.2')
    .add_local_python_source('pool_generalization'))


def status(stage, completed=0, **fields):
    row = dict(at_utc=datetime.now(timezone.utc).isoformat(), stage=stage,
               completed=completed, total=576, profile=PROFILE, **fields)
    save_json(STATE / 'status.json', row)
    with (STATE / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


@app.function(image=image, volumes={'/sweep': volume}, timeout=120, scaledown_window=2)
def audit(expected_spec):
    from tokenizers import Tokenizer
    root = Path('/sweep');volume.reload()
    spec = json.loads((root / 'spec.json').read_text())
    if digest(spec) != digest(expected_spec):
        raise ValueError('Original evaluation specification changed')
    panel = json.loads((root / 'panel.json').read_text())
    if digest(panel) != spec['panel_sha256']:
        raise ValueError('Original evaluation panel changed')
    path = root / 'input/model/tokenizer.json'
    if file_sha(path) != spec['input_hashes']['model/tokenizer.json']:
        raise ValueError('Canonical tokenizer changed')
    canonical = Tokenizer.from_file(str(path))
    splits = {}
    for split in ['screen', 'confirm']:
        rows = []
        for row in panel[split]:
            prefix = canonical.encode(row['prefix'], add_special_tokens=False).ids
            suffix = canonical.decode(row['canonical_continuation_ids'], skip_special_tokens=False)
            combined = canonical.encode(row['prefix'] + suffix, add_special_tokens=False).ids
            if prefix != row['canonical_prefix_ids']:
                raise ValueError('Saved prefix encoding changed')
            if len(row['canonical_continuation_ids']) != 32 or len(suffix.encode()) != row['continuation_bytes']:
                raise ValueError('Saved continuation changed')
            rows.append(dict(id=row['id'], domain=row['domain'],
                canonical_joint_encoding_matches=combined == prefix + row['canonical_continuation_ids'],
                cut_inside_alphanumeric=bool(row['prefix'][-1].isalnum() and suffix[0].isalnum()),
                continuation_bytes=row['continuation_bytes']))
        by_domain = {}
        for domain in spec['domains']:
            sub = [r for r in rows if r['domain'] == domain]
            by_domain[domain] = dict(passages=len(sub), valid_seams=sum(r['canonical_joint_encoding_matches'] for r in sub))
        splits[split] = dict(rows=rows, passages=len(rows),
            valid_seams=sum(r['canonical_joint_encoding_matches'] for r in rows), by_domain=by_domain)
    result = dict(panel_sha256=digest(panel), source_spec_sha256=digest(spec), splits=splits,
        purpose='Read-only encoding audit of the original 1000-tokenizer screen and confirmation panels. No cuts, targets, texts or scores changed.')
    save_json(root / PREFIX / 'audit.json', result);volume.commit()
    return result


@app.local_entrypoint()
def main():
    if os.environ.get('MODAL_PROFILE') != PROFILE:
        raise ValueError('Use ' + PROFILE)
    STATE.mkdir(parents=True, exist_ok=True)
    status('auditing original canonical token boundaries')
    try:
        spec = json.loads((ROOT / 'reports/rl-pool-sweep-20k/spec.json').read_text())
        result = audit.remote(spec)
        save_json(STATE / 'audit.json', result)
        status('complete', 576, valid_seams={k: v['valid_seams'] for k, v in result['splits'].items()})
    except Exception as exc:
        status('failed', error=str(exc));raise
