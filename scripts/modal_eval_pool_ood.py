"""Run the fixed 11-tokenizer held-out test on H200, using cloud-staged weights."""
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import sys

import modal
from pool_generalization import load_plan, file_sha, save_json
from pool_ood_eval import LABELS, STEPS, EXPECTED

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runtime/anytok_pool_20k'
PLAN = ROOT / 'plans/pool-ood-48k-v1.json'
STATE = ROOT / 'modal_results/pool_ood_48k'
NAME = 'anytok-pool-ood-48k'
app = modal.App(NAME)
# Reuse the already completed cloud-to-cloud asset staging. Results stay separate.
volume = modal.Volume.from_name('anytok-pool-generalization', version=2)
image = (modal.Image.from_registry('ubuntu:24.04', add_python='3.12')
    .apt_install('libpython3.12')
    .uv_pip_install('torch==2.15.0', 'transformers==5.15.1', 'tokenizers==0.22.2',
                    'numpy==2.3.5', 'safetensors==0.8.0', 'accelerate==1.14.0')
    .add_local_dir(str(SOURCE / 'hf_hypernet'), '/opt/zett/hf_hypernet')
    .add_local_file(str(SOURCE / 'rust_utils/target/release/librust_utils.so'),
                    '/opt/zett/rust_utils/target/release/librust_utils.so')
    .add_local_python_source('pool_ood_eval', 'pool_generalization'))


def timestamp():
    return datetime.now(timezone.utc).isoformat()


@app.function(image=image, gpu='H200', volumes={'/eval': volume}, timeout=3600)
def evaluate():
    sys.path.insert(0, '/opt/zett')
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from hf_hypernet.modeling_hypernet import ZettHypernet
    from hf_hypernet.smollm_setup import separate_input_embeddings_and_freeze_output
    from hf_hypernet.tokenizer_inputs import build_surface_forms, enable_byte_fallbacks
    from pool_ood_eval import examples_for, score_documents, summarize

    root = Path('/eval'); out = root / 'ood48-v1'; out.mkdir(exist_ok=True)
    plan = load_plan(out / 'plan.json'); panel = plan['panel']
    def event(stage, completed=0, **details):
        value = {'at_utc': timestamp(), 'stage': stage, 'completed': completed,
                 'total': 33, 'profile': 'trenton', **details}
        save_json(out / 'status.json', value)
        with (out / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(value) + '\n')
        volume.commit(); print(json.dumps(value), flush=True)
    try:
        event('verifying inputs')
        canonical = AutoTokenizer.from_pretrained(root / 'model', local_files_only=True)
        tokenizers = {}; tokens = {}
        for name in LABELS:
            path = out / 'tokenizers' / name.lower() / 'tokenizer.json'
            if file_sha(path) != plan['tokenizer_sha256'][name]:
                raise ValueError('Held-out tokenizer checksum changed')
            tokenizer = Tokenizer.from_file(str(path)); vocab = tokenizer.get_vocab()
            if len(vocab) != 48000 or set(vocab.values()) != set(range(48000)) or vocab['<|endoftext|>'] != 0:
                raise ValueError('Unexpected native tokenizer IDs')
            ordered = [None] * 48000
            for token, index in vocab.items(): ordered[index] = token
            tokenizers[name] = tokenizer; tokens[name] = ordered
        if (len(panel), sum(len(r['canonical_continuation_ids']) for r in panel),
                sum(r['continuation_bytes'] for r in panel)) != EXPECTED:
            raise ValueError('Scored cohort changed')
        canonical_examples = examples_for(panel)
        for r in panel:
            if canonical(r['prefix'], add_special_tokens=False).input_ids != r['canonical_prefix_ids']:
                raise ValueError('Canonical prefix changed')
            if len(canonical.decode(r['canonical_continuation_ids'], clean_up_tokenization_spaces=False).encode('utf-8')) != r['continuation_bytes']:
                raise ValueError('Continuation bytes changed')
        target_examples = {n: examples_for(panel, t) for n, t in tokenizers.items()}
        # Verify checkpoint files before any scoring; only these immutable files were copied.
        for step in STEPS:
            cp = root / 'checkpoints' / f'step-{step:06d}'
            manifest = json.loads((cp / 'manifest.json').read_text())
            if manifest['step'] != step: raise ValueError('Wrong checkpoint')
            for name in ['lm.pt', 'hypernet/config.json', 'hypernet/model.safetensors']:
                p = cp / name; expected = manifest['files'][name]
                if p.stat().st_size != expected['bytes'] or file_sha(p) != expected['sha256']:
                    raise ValueError(f'Corrupt checkpoint: {step}/{name}')
        model = AutoModelForCausalLM.from_pretrained(root / 'model', local_files_only=True,
                    dtype=torch.float32, low_cpu_mem_usage=True).to('cuda')
        separate_input_embeddings_and_freeze_output(model); model.eval()
        reference = model.get_input_embeddings().weight.detach().clone()
        results = {}
        for step in STEPS:
            cp = root / 'checkpoints' / f'step-{step:06d}'
            path = out / f'step-{step:06d}.json'
            if path.exists():
                saved = json.loads(path.read_text())
                if saved['plan_sha256'] != plan['plan_sha256']: raise ValueError('Previous output has different inputs')
                results[str(step)] = saved
                continue
            model.load_state_dict(torch.load(cp / 'lm.pt', map_location='cpu', weights_only=True))
            table = model.get_input_embeddings().weight.detach()
            if not torch.equal(table, reference) or not torch.equal(model.get_output_embeddings().weight, reference):
                raise ValueError('Canonical embedding table or output head changed')
            event('scoring canonical baseline', len(results) * 11, step=step)
            canonical_nats = score_documents(model, canonical_examples, table, table)
            hypernet = ZettHypernet.from_pretrained(cp / 'hypernet', local_files_only=True).to('cuda').eval()
            fallbacks = enable_byte_fallbacks(hypernet, canonical)
            scores = {}
            for name in LABELS:
                event('evaluating', len(results) * 11 + len(scores), step=step, tokenizer=name)
                forms_np, truncated = build_surface_forms(tokens[name], canonical, 10, fallbacks, {'<|endoftext|>': 0})
                forms = torch.from_numpy(forms_np.astype(np.int64)).to('cuda')
                priors = torch.zeros(48000, device='cuda', dtype=torch.float32)
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    generated, output_embeddings, _ = hypernet(forms, priors, table, None, deterministic=True)
                if output_embeddings is not None or generated.shape != (48000, table.shape[1]) or not torch.isfinite(generated).all():
                    raise ValueError('Invalid transferred input embeddings')
                scores[name] = {'nats': score_documents(model, target_examples[name], generated, table),
                                'truncated_surface_forms': truncated,
                                'prefix_tokens': sum(len(r[0]) for r in target_examples[name])}
                del generated, forms, priors; torch.cuda.empty_cache()
            result = {'step': step, 'plan_sha256': plan['plan_sha256'], 'ids': [r['id'] for r in panel],
                      'bytes': [r['continuation_bytes'] for r in panel],
                      'canonical_nats': canonical_nats, 'tokenizers': scores}
            save_json(path, result); results[str(step)] = result; volume.commit()
            del hypernet; torch.cuda.empty_cache()
        report = summarize(results, panel)
        report.update(plan_sha256=plan['plan_sha256'], documents=256, continuation_tokens=8192,
                      continuation_bytes=38534, max_positions=max(r['maximum_positions'] for r in panel),
                      model_run='uniform generated pool', position_cap=128, suffix_tokens=32)
        save_json(out / 'report.json', report); event('complete', 33)
        return report
    except BaseException as exc:
        event('failed', error=str(exc)); raise


@app.local_entrypoint()
def main(collect_only: bool = False):
    if os.environ.get('MODAL_PROFILE') != 'trenton': raise ValueError('Use the staged trenton workspace')
    from modal_credits import billing_row
    balance = billing_row('trenton', 'this month', Decimal(30))
    if not collect_only and (balance['status'] != 'ok' or float(balance['estimated_remaining_usd']) < 6):
        raise RuntimeError('Insufficient verified credit margin')
    plan = load_plan(PLAN)
    STATE.mkdir(parents=True, exist_ok=True)
    if collect_only:
        saved = json.loads((STATE / 'call.json').read_text())
        if saved['plan_sha256'] != plan['plan_sha256']: raise ValueError('Different saved evaluation plan')
        call = modal.FunctionCall.from_id(saved['call_id'])
    else:
        with volume.batch_upload(force=True) as batch:
            batch.put_file(PLAN, 'ood48-v1/plan.json')
            # Saved artifacts are being persisted, not generated again.
            for name in LABELS:
                folder = ROOT / 'modal_results/tokenizers-specialty-48k' / name.lower()
                if file_sha(folder / 'tokenizer.json') != plan['tokenizer_sha256'][name]:
                    raise ValueError('Local tokenizer changed after preflight')
                for filename in ['tokenizer.json', 'manifest.json']:
                    batch.put_file(folder / filename, f'ood48-v1/tokenizers/{name.lower()}/{filename}')
        save_json(STATE / 'status.json', {'at_utc': timestamp(), 'stage': 'starting H200', 'completed': 0, 'total': 33, 'profile': 'trenton'})
        call = evaluate.spawn()
        save_json(STATE / 'call.json', {'call_id': call.object_id, 'plan_sha256': plan['plan_sha256']})
    while True:
        try:
            report = call.get(timeout=5)
            break
        except (TimeoutError, modal.exception.TimeoutError):
            try:
                state = json.loads(b''.join(volume.read_file('ood48-v1/status.json')))
                save_json(STATE / 'status.json', state)
                # Small score/metadata files only; model weights never pass through the caller.
                if state.get('stage') == 'failed': raise RuntimeError(state['error'])
            except FileNotFoundError:
                pass
    save_json(STATE / 'report.json', report)
    for step in STEPS:
        raw = b''.join(volume.read_file(f'ood48-v1/step-{step:06d}.json'))
        save_json(STATE / f'step-{step:06d}.json', json.loads(raw))
    save_json(STATE / 'status.json', {'at_utc': timestamp(), 'stage': 'complete', 'completed': 33, 'total': 33, 'profile': 'trenton'})
    import subprocess
    subprocess.run([sys.executable, str(ROOT / 'plotting_scripts/plot_pool_ood_48k.py')], check=True)
    print(json.dumps(report), flush=True)
