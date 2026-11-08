"""Search existing useful native tokenizers, then confirm failures on new documents."""
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request

import modal
from pool_generalization import digest, file_sha, load_plan, save_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'plans/uniform-weakness-v1.json'
STATE = ROOT / 'modal_results/uniform_weakness'
NAME = 'anytok-uniform-weakness'
PREFIX = 'weakness-v1'
app = modal.App(NAME)
volume = modal.Volume.from_name('anytok-pool-generalization', version=2)
cpu_image = modal.Image.debian_slim(python_version='3.11').add_local_python_source('pool_generalization')
SOURCE = ROOT / 'runtime/anytok_pool_20k'
gpu_image = (modal.Image.from_registry('ubuntu:24.04', add_python='3.12')
    .apt_install('libpython3.12')
    .uv_pip_install('torch==2.15.0', 'transformers==5.15.1', 'tokenizers==0.22.2',
                    'numpy==2.3.5', 'safetensors==0.8.0', 'accelerate==1.14.0')
    .add_local_dir(str(SOURCE / 'hf_hypernet'), '/opt/zett/hf_hypernet')
    .add_local_file(str(SOURCE / 'rust_utils/target/release/librust_utils.so'),
                    '/opt/zett/rust_utils/target/release/librust_utils.so')
    .add_local_python_source('uniform_weakness', 'pool_ood_eval', 'pool_generalization'))
audit_image = (modal.Image.debian_slim(python_version='3.11')
               .uv_pip_install('tokenizers==0.22.2')
               .add_local_python_source('pool_generalization'))


def now(): return datetime.now(timezone.utc).isoformat()


@app.function(image=cpu_image, volumes={'/eval': volume}, max_containers=100, timeout=1800)
def copy_file(item, source_volume):
    import time
    client = modal.Client.from_credentials(os.environ['SOURCE_MODAL_TOKEN_ID'], os.environ['SOURCE_MODAL_TOKEN_SECRET'])
    source = modal.Volume.from_name(source_volume, client=client)
    target = Path('/eval') / PREFIX / item['destination']
    if target.exists() and target.stat().st_size == item['bytes'] and file_sha(target) == item['sha256']:
        return {'destination': item['destination'], 'reused': True}
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix('.partial')
    for attempt in range(4):
        try:
            with partial.open('wb') as stream:
                for block in source.read_file(item['source']): stream.write(block)
            if partial.stat().st_size != item['bytes'] or file_sha(partial) != item['sha256']:
                raise ValueError('Asset checksum mismatch')
            partial.replace(target); volume.commit()
            return {'destination': item['destination'], 'bytes': item['bytes']}
        except (OSError, modal.exception.ConnectionError):
            if attempt == 3: raise
            time.sleep(2 ** attempt)


@app.function(image=cpu_image, volumes={'/eval': volume}, max_containers=100, timeout=600)
def fetch_published(candidate):
    target = Path('/eval') / PREFIX / 'tokenizers' / candidate['id'] / 'tokenizer.json'
    manifest_path = target.with_name('manifest.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old['revision'] != candidate['revision'] or file_sha(target) != old['sha256']:
            raise ValueError('Existing published tokenizer differs')
        return old
    url = f"https://huggingface.co/{candidate['repo']}/resolve/{candidate['revision']}/tokenizer.json"
    partial = target.with_suffix('.partial')
    with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as stream:
        while block := response.read(1 << 20): stream.write(block)
    parsed = json.loads(partial.read_text())
    manifest = {**candidate, 'sha256': file_sha(partial), 'bytes': partial.stat().st_size,
                'model_type': parsed['model']['type'], 'decoder': parsed.get('decoder'),
                'normalizer': parsed.get('normalizer')}
    partial.replace(target); save_json(manifest_path, manifest); volume.commit()
    return manifest


@app.function(image=gpu_image, gpu='H200', volumes={'/eval': volume}, timeout=3600)
def search():
    sys.path.insert(0, '/opt/zett')
    import gzip
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from hf_hypernet.modeling_hypernet import ZettHypernet
    from hf_hypernet.smollm_setup import separate_input_embeddings_and_freeze_output
    from hf_hypernet.tokenizer_inputs import build_surface_forms, enable_byte_fallbacks
    from pool_ood_eval import score_documents, examples_for
    from uniform_weakness import load_tokenizer, make_panels, compare_scores, compression, classify

    root = Path('/eval'); out = root / PREFIX
    plan = load_plan(out / 'plan.json'); total = len(plan['candidates'])
    def emit(stage, completed=0, **details):
        value = {'at_utc': now(), 'stage': stage, 'completed': completed, 'total': total,
                 'profile': 'trenton', **details}
        save_json(out / 'status.json', value)
        with (out / 'events.jsonl').open('a') as stream: stream.write(json.dumps(value) + '\n')
        volume.commit(); print(json.dumps(value), flush=True)
    try:
        emit('validating native tokenizer interfaces')
        canonical = AutoTokenizer.from_pretrained(root / 'model', local_files_only=True)
        candidates = []; rejected = []
        for item in plan['candidates']:
            path = out / 'tokenizers' / item['id'] / 'tokenizer.json'
            expected = (item['sha256'] if item['kind'] == 'saved-native' else
                        json.loads(path.with_name('manifest.json').read_text())['sha256'])
            if file_sha(path) != expected: raise ValueError('Tokenizer checksum changed')
            try: native = load_tokenizer(path)
            except ValueError as exc:
                rejected.append({'id': item['id'], 'reason': str(exc)}); continue
            candidates.append({**item, 'sha256': expected, 'native': native,
                               'vocab_size': len(native[1]),
                               'absent_from_training_pool': expected not in plan['pool_tokenizer_sha256']})
        targets = [c['native'][0] for c in candidates]
        special_ids = [c['native'][4] for c in candidates]
        panels = {}
        for domain in plan['domains']:
            emit('preparing fixed screen/confirmation documents', domain=domain)
            with gzip.open(out / 'corpus' / f'{domain}.jsonl.gz', 'rt', encoding='utf-8') as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
            panels[domain] = make_panels(rows, domain, canonical, targets, special_ids,
                                        plan['screen_documents_per_domain'], plan['confirm_documents_per_domain'])
        save_json(out / 'panels.json', panels); volume.commit()
        cp = root / 'checkpoints' / 'step-012000'
        manifest = json.loads((cp / 'manifest.json').read_text())
        if manifest['step'] != 12000: raise ValueError('Wrong uniform checkpoint')
        for name in ['lm.pt', 'hypernet/config.json', 'hypernet/model.safetensors']:
            if file_sha(cp / name) != manifest['files'][name]['sha256']: raise ValueError('Checkpoint checksum changed')
        emit('loading uniform 12k checkpoint')
        model = AutoModelForCausalLM.from_pretrained(root / 'model', local_files_only=True,
                    dtype=torch.float32, low_cpu_mem_usage=True).to('cuda')
        separate_input_embeddings_and_freeze_output(model); model.eval()
        reference = model.get_input_embeddings().weight.detach().clone()
        model.load_state_dict(torch.load(cp / 'lm.pt', map_location='cpu', weights_only=True))
        table = model.get_input_embeddings().weight.detach()
        if not torch.equal(table, reference) or not torch.equal(model.get_output_embeddings().weight, reference):
            raise ValueError('Canonical input/output tables changed')
        del reference
        hypernet = ZettHypernet.from_pretrained(cp / 'hypernet', local_files_only=True).to('cuda').eval()
        fallbacks = enable_byte_fallbacks(hypernet, canonical)
        baselines = {}
        for domain, cohorts in panels.items():
            baselines[domain] = {}
            for split in ['screen', 'confirm']:
                emit('scoring canonical controls', domain=domain, split=split)
                baselines[domain][split] = score_documents(model, examples_for(cohorts[split]), table, table)
        save_json(out / 'canonical-nats.json', baselines); volume.commit()
        def generate(candidate):
            tokenizer, tokens, native_priors, specials, _ = candidate['native']
            forms_np, truncated = build_surface_forms(tokens, canonical, 10, fallbacks, specials)
            forms = torch.from_numpy(forms_np.astype(np.int64)).to('cuda')
            priors = torch.tensor(native_priors, dtype=torch.float32, device='cuda')
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                generated, output, _ = hypernet(forms, priors, table, None, deterministic=True)
            if output is not None or generated.shape != (len(tokens), table.shape[1]) or not torch.isfinite(generated).all():
                raise ValueError('Invalid generated input embeddings')
            del forms, priors
            return generated, truncated
        screen = []; errors = []; confirm = []
        for index, candidate in enumerate(candidates):
            name = candidate['id']; tokenizer = candidate['native'][0]
            emit('screening', index, tokenizer=name)
            try:
                generated, truncated = generate(candidate)
                for domain, cohorts in panels.items():
                    panel = cohorts['screen']
                    nats = score_documents(model, examples_for(panel, tokenizer), generated, table)
                    score = compare_scores(nats, baselines[domain]['screen'], [r['continuation_bytes'] for r in panel])
                    try: utility = compression(panel, tokenizer, canonical)
                    except ValueError as exc: utility = {'error': str(exc), 'token_reduction_vs_canonical_percent': -100.}
                    row = {'tokenizer': name, 'domain': domain, 'score': score, 'utility': utility,
                           'truncated_surface_forms': truncated}
                    screen.append(row)
                    save_json(out / 'screen.json', screen); volume.commit()
                del generated; torch.cuda.empty_cache()
            except (ValueError, RuntimeError) as exc:
                errors.append({'tokenizer': name, 'reason': str(exc), 'stage': 'screen'})
                if 'generated' in locals(): del generated
                torch.cuda.empty_cache()
        shortlist = [r for r in screen if r['score']['gap_bpb'] >= plan['screen_gap_threshold']]
        names = {r['tokenizer'] for r in shortlist}
        total = len(candidates) + len(names)
        for index, candidate in enumerate(c for c in candidates if c['id'] in names):
            name = candidate['id']; tokenizer = candidate['native'][0]
            emit('independent confirmation', len(candidates) + index, tokenizer=name,
                 screen_pairs_above_threshold=len(shortlist))
            generated, truncated = generate(candidate)
            for selected in (r for r in shortlist if r['tokenizer'] == name):
                domain = selected['domain']; panel = panels[domain]['confirm']
                nats = score_documents(model, examples_for(panel, tokenizer), generated, table)
                score = compare_scores(nats, baselines[domain]['confirm'], [r['continuation_bytes'] for r in panel])
                try: utility = compression(panel, tokenizer, canonical)
                except ValueError as exc: utility = {'error': str(exc), 'token_reduction_vs_canonical_percent': -100.}
                row = {'tokenizer': name, 'domain': domain, 'score': score, 'utility': utility,
                       'classification': classify(score, utility),
                       'document_ids': [r['id'] for r in panel], 'nats': nats,
                       'canonical_nats': baselines[domain]['confirm'],
                       'continuation_bytes': [r['continuation_bytes'] for r in panel],
                       'maximum_positions': max(r['maximum_positions'] for r in panel),
                       'canonical_prefix_tokens': sum(len(r['canonical_prefix_ids']) for r in panel),
                       'target_prefix_tokens': sum(len(r[0]) for r in examples_for(panel, tokenizer)),
                       'truncated_surface_forms': truncated}
                confirm.append(row); save_json(out / 'confirmation.json', confirm); volume.commit()
            del generated; torch.cuda.empty_cache()
        report = {'plan_sha256': plan['plan_sha256'], 'checkpoint_step': 12000,
                  'candidates': [{k: v for k, v in c.items() if k != 'native'} for c in candidates],
                  'rejected_interfaces': rejected, 'evaluation_errors': errors,
                  'screen': screen, 'confirmation': confirm,
                  'qualified': [r for r in confirm if r['classification']['confirmed_weak_and_useful']],
                  'panel_counts': {d: {s: len(p[s]) for s in ['screen', 'confirm']} for d, p in panels.items()},
                  'position_cap': 128, 'suffix_tokens': 32,
                  'warning': '12k checkpoint while training continues; >=0.1 on independent confirmation is provisional until repeated at20k. Native published/150k tokenizers may be outside training support.'}
        save_json(out / 'report.json', report); emit('complete', total)
        return report
    except BaseException as exc:
        emit('failed', error=str(exc)); raise


def local_status(stage, **details):
    STATE.mkdir(parents=True, exist_ok=True)
    save_json(STATE / 'status.json', {'at_utc': now(), 'stage': stage, 'completed': 0,
                                    'total': 26, 'profile': 'trenton', **details})


@app.function(image=audit_image, volumes={'/eval': volume}, timeout=600)
def audit():
    """Recheck actual raw text, native encodings, independent IDs and score pairing."""
    from tokenizers import Tokenizer
    root = Path('/eval'); out = root / PREFIX
    plan = load_plan(out / 'plan.json')
    report = json.loads((out / 'report.json').read_text())
    panels = json.loads((out / 'panels.json').read_text())
    base = json.loads((out / 'canonical-nats.json').read_text())
    if report['plan_sha256'] != plan['plan_sha256']: raise ValueError('Wrong report plan')
    canonical = Tokenizer.from_file(str(root / 'model/tokenizer.json'))
    tokenizers = {c['id']: Tokenizer.from_file(str(out / 'tokenizers' / c['id'] / 'tokenizer.json'))
                  for c in report['candidates']}
    for domain, cohorts in panels.items():
        if {r['id'] for r in cohorts['screen']} & {r['id'] for r in cohorts['confirm']}:
            raise ValueError('Confirmation reuses screening documents')
        for split in ['screen', 'confirm']:
            for row in cohorts[split]:
                source_ids = canonical.encode(row['prefix'], add_special_tokens=False).ids
                suffix_ids = row['canonical_continuation_ids']
                suffix = canonical.decode(suffix_ids, skip_special_tokens=False)
                if (source_ids != row['canonical_prefix_ids'] or len(suffix_ids) != 32
                        or len(suffix.encode('utf-8')) != row['continuation_bytes']
                        or not row['compression_text'][:row['cut_chars']].endswith(row['prefix'])
                        or not row['compression_text'][row['cut_chars']:].startswith(suffix)):
                    raise ValueError('Raw prefix or scored suffix changed')
                for tokenizer in tokenizers.values():
                    ids = tokenizer.encode(row['prefix'], add_special_tokens=False).ids
                    if tokenizer.decode(ids, skip_special_tokens=False) != row['prefix'] or 1 + len(ids) + 32 > 128:
                        raise ValueError('Target context differs or exceeds the cap')
                if 1 + len(source_ids) + 32 > 128: raise ValueError('Canonical context exceeds the cap')
    checked = []
    for row in report['confirmation']:
        cohort = panels[row['domain']]['confirm']
        if (row['document_ids'] != [r['id'] for r in cohort]
                or row['continuation_bytes'] != [r['continuation_bytes'] for r in cohort]
                or row['canonical_nats'] != base[row['domain']]['confirm']
                or len(row['nats']) != len(cohort)):
            raise ValueError('Per-document score pairing differs')
        checked.append({'tokenizer': row['tokenizer'], 'domain': row['domain'],
                        'documents': len(cohort), 'suffix_tokens': 32 * len(cohort),
                        'suffix_bytes': sum(row['continuation_bytes']),
                        'canonical_prefix_tokens_min': min(len(r['canonical_prefix_ids']) for r in cohort),
                        'canonical_prefix_tokens_max': max(len(r['canonical_prefix_ids']) for r in cohort)})
    value = {'at_utc': now(), 'passed': True, 'plan_sha256': plan['plan_sha256'],
             'candidate_count': len(tokenizers), 'domains': len(panels),
             'confirmed_pair_count': len(checked), 'checked': checked}
    save_json(out / 'audit.json', value); volume.commit(); return value


@app.function(image=gpu_image, gpu='H200', volumes={'/eval': volume}, timeout=1800)
def rerun_selected(spec):
    """Reuse the original confirmation contexts, changing only checkpoint weights."""
    sys.path.insert(0, '/opt/zett')
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from hf_hypernet.modeling_hypernet import ZettHypernet
    from hf_hypernet.smollm_setup import separate_input_embeddings_and_freeze_output
    from hf_hypernet.tokenizer_inputs import build_surface_forms, enable_byte_fallbacks
    from pool_ood_eval import score_documents, examples_for
    from uniform_weakness import load_tokenizer, compare_checkpoints

    root = Path('/eval'); source = root / PREFIX
    out = source / 'reruns' / f"step-{spec['step']:06d}"
    def emit(stage, completed=0, **details):
        value = {'at_utc': now(), 'stage': stage, 'step': spec['step'], 'completed': completed,
                 'total': len(spec['cases']), 'profile': 'trenton', **details}
        save_json(out / 'status.json', value)
        with (out / 'events.jsonl').open('a') as stream: stream.write(json.dumps(value) + '\n')
        volume.commit(); print(json.dumps(value), flush=True)
    try:
        emit('verifying saved confirmation panel and checkpoint')
        previous = json.loads((source / 'report.json').read_text())
        audit_result = json.loads((source / 'audit.json').read_text())
        if (digest(previous) != spec['previous_report_digest'] or not audit_result['passed']
                or audit_result['plan_sha256'] != previous['plan_sha256']):
            raise ValueError('Original results or panel audit changed')
        panels = json.loads((source / 'panels.json').read_text())
        cp = out / 'checkpoint'
        manifest = json.loads((cp / 'manifest.json').read_text())
        if manifest != spec['checkpoint_manifest']: raise ValueError('Checkpoint manifest changed')
        for name in ['lm.pt', 'hypernet/config.json', 'hypernet/model.safetensors']:
            expected = manifest['files'][name]
            if (cp / name).stat().st_size != expected['bytes'] or file_sha(cp / name) != expected['sha256']:
                raise ValueError(f'Checkpoint checksum mismatch: {name}')
        canonical = AutoTokenizer.from_pretrained(root / 'model', local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(root / 'model', local_files_only=True,
                    dtype=torch.float32, low_cpu_mem_usage=True).to('cuda')
        separate_input_embeddings_and_freeze_output(model); model.eval()
        reference = model.get_input_embeddings().weight.detach().clone()
        model.load_state_dict(torch.load(cp / 'lm.pt', map_location='cpu', weights_only=True))
        table = model.get_input_embeddings().weight.detach()
        if not torch.equal(table, reference) or not torch.equal(model.get_output_embeddings().weight, reference):
            raise ValueError('Canonical embedding tables changed')
        del reference
        hypernet = ZettHypernet.from_pretrained(cp / 'hypernet', local_files_only=True).to('cuda').eval()
        fallbacks = enable_byte_fallbacks(hypernet, canonical)
        rows = []
        for case in spec['cases']:
            name, domain = case['tokenizer'], case['domain']
            emit('evaluating fixed confirmation cases', len(rows), tokenizer=name, domain=domain)
            old = next(r for r in previous['confirmation'] if r['tokenizer'] == name and r['domain'] == domain)
            if digest(old) != case['previous_case_digest']: raise ValueError('Previous case changed')
            panel = panels[domain]['confirm']
            path = source / 'tokenizers' / name / 'tokenizer.json'
            if file_sha(path) != case['tokenizer_sha256']: raise ValueError('Tokenizer changed')
            tokenizer, tokens, native_priors, specials, _ = load_tokenizer(path)
            target_examples = examples_for(panel, tokenizer)
            if (sum(len(r[0]) for r in target_examples) != old['target_prefix_tokens']
                    or sum(len(r['canonical_prefix_ids']) for r in panel) != old['canonical_prefix_tokens']):
                raise ValueError('Prefix token counts changed')
            for row in panel:
                suffix = canonical.decode(row['canonical_continuation_ids'], clean_up_tokenization_spaces=False)
                if (canonical(row['prefix'], add_special_tokens=False).input_ids != row['canonical_prefix_ids']
                        or len(row['canonical_continuation_ids']) != 32
                        or len(suffix.encode('utf-8')) != row['continuation_bytes']
                        or not row['compression_text'][row['cut_chars']:].startswith(suffix)
                        or not row['compression_text'][:row['cut_chars']].endswith(row['prefix'])):
                    raise ValueError('Source text or scored suffix changed')
            canonical_nats = score_documents(model, examples_for(panel), table, table)
            forms_np, truncated = build_surface_forms(tokens, canonical, 10, fallbacks, specials)
            forms = torch.from_numpy(forms_np.astype(np.int64)).to('cuda')
            priors = torch.tensor(native_priors, dtype=torch.float32, device='cuda')
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                generated, output, _ = hypernet(forms, priors, table, None, deterministic=True)
            if output is not None or generated.shape != (len(tokens), table.shape[1]) or not torch.isfinite(generated).all():
                raise ValueError('Invalid generated embeddings')
            current = {'document_ids': [r['id'] for r in panel],
                       'continuation_bytes': [r['continuation_bytes'] for r in panel],
                       'canonical_nats': canonical_nats,
                       'nats': score_documents(model, target_examples, generated, table)}
            comparison = compare_checkpoints(old, current)
            rows.append({'tokenizer': name, 'domain': domain, 'comparison': comparison,
                         'current_scores': current, 'utility': old['utility'],
                         'tokenizer_sha256': case['tokenizer_sha256'],
                         'context_sha256': digest(panel), 'truncated_surface_forms': truncated,
                         'maximum_positions': max(r['maximum_positions'] for r in panel)})
            del forms, priors, generated; torch.cuda.empty_cache()
            save_json(out / 'results.json', rows); volume.commit()
        report = {'previous_step': previous['checkpoint_step'], 'step': spec['step'],
                  'source_profile': spec['source_profile'], 'spec_digest': digest(spec),
                  'previous_report_digest': spec['previous_report_digest'], 'rows': rows,
                  'position_cap': 128, 'suffix_tokens': 32, 'document_alignment_verified': True}
        save_json(out / 'report.json', report); emit('complete', len(rows)); return report
    except BaseException as exc:
        emit('failed', error=str(exc)); raise


def rerun_latest_main(collect_only=False):
    """Freeze the newest committed checkpoint; copy weights directly into eval storage."""
    from modal_tokenizer_pool_transfer import source_client
    import io
    pointer = STATE / 'latest-rerun.json'
    if collect_only:
        saved = json.loads(pointer.read_text()); spec = saved['spec']
        call = modal.FunctionCall.from_id(saved['call_id'])
    else:
        from modal_credits import billing_row
        balance = billing_row('trenton', 'this month', Decimal(30))
        if balance['status'] != 'ok' or float(balance['estimated_remaining_usd']) < 6:
            raise RuntimeError('Insufficient verified credit margin')
        training = json.loads((ROOT / 'modal_results/anytok_pool_20k/status.json').read_text())
        candidates = []
        for profile in dict.fromkeys([training.get('profile'), training.get('resume_source')]):
            if not profile: continue
            client, secret = source_client(profile)
            src = modal.Volume.from_name('anytok-pool-20k', client=client)
            entries = list(src.iterdir('run', recursive=True))
            for entry in entries:
                if not entry.path.endswith('/manifest.json') or '/step-' not in entry.path: continue
                step = int(Path(entry.path).parent.name.removeprefix('step-'))
                candidates.append((step, profile, src, secret, entry.path))
        if not candidates: raise ValueError('No committed checkpoint found')
        step, profile, src, secret, path = max(candidates, key=lambda c: c[0])
        if step <= 12000: raise ValueError('No newer checkpoint than the 12k reference')
        manifest = json.loads(b''.join(src.read_file(path)))
        if manifest['step'] != step: raise ValueError('Wrong checkpoint manifest step')
        previous = json.loads((STATE / 'report.json').read_text())
        cases = []
        for name, domain in [('ood48-chinese35', 'web-zh'), ('ood48-japanese35', 'web-ja'), ('base64', 'medical-en')]:
            old = next(r for r in previous['confirmation'] if r['tokenizer'] == name and r['domain'] == domain)
            native = next(c for c in previous['candidates'] if c['id'] == name)
            cases.append({'tokenizer': name, 'domain': domain, 'previous_case_digest': digest(old),
                          'tokenizer_sha256': native['sha256']})
        spec = {'step': step, 'source_profile': profile, 'checkpoint_manifest': manifest,
                'previous_report_digest': digest(previous), 'cases': cases,
                'position_cap': 128, 'suffix_tokens': 32}
        prefix = f'reruns/step-{step:06d}'
        local = STATE / prefix; local.mkdir(parents=True, exist_ok=True)
        remote = PREFIX + '/' + prefix
        save_json(local / 'spec.json', spec)
        save_json(STATE / 'rerun-status.json', {'at_utc': now(), 'stage': 'copying newest checkpoint cloud-to-cloud',
                  'step': step, 'completed': 0, 'total': 3, 'profile': 'trenton', 'source_profile': profile})
        with volume.batch_upload(force=True) as batch:
            batch.put_file(local / 'spec.json', remote + '/spec.json')
            batch.put_file(io.BytesIO(json.dumps(manifest).encode()), remote + '/checkpoint/manifest.json')
        items = [{'source': str(Path(path).parent / name), 'destination': prefix + '/checkpoint/' + name,
                  **manifest['files'][name]} for name in ['lm.pt', 'hypernet/config.json', 'hypernet/model.safetensors']]
        list(copy_file.with_options(secrets=[secret]).starmap([(i, 'anytok-pool-20k') for i in items], order_outputs=False))
        call = rerun_selected.spawn(spec)
        save_json(pointer, {'spec': spec, 'call_id': call.object_id})
    while True:
        try:
            report = call.get(timeout=5); break
        except (TimeoutError, modal.exception.TimeoutError):
            try:
                status = json.loads(b''.join(volume.read_file(f"{PREFIX}/reruns/step-{spec['step']:06d}/status.json")))
                save_json(STATE / 'rerun-status.json', status)
            except FileNotFoundError: pass
    local = STATE / 'reruns' / f"step-{spec['step']:06d}"; local.mkdir(parents=True, exist_ok=True)
    save_json(local / 'report.json', report)
    save_json(STATE / 'rerun-status.json', {'at_utc': now(), 'stage': 'complete', 'step': spec['step'],
              'completed': len(report['rows']), 'total': 3, 'profile': 'trenton'})
    print(json.dumps({'step': spec['step'], 'rows': [{k: v for k, v in r.items() if k != 'current_scores'} for r in report['rows']]}), flush=True)


@app.local_entrypoint()
def main(collect_only: bool = False, audit_only: bool = False, rerun_latest: bool = False):
    if os.environ.get('MODAL_PROFILE') != 'trenton': raise ValueError('Use the staged trenton workspace')
    if rerun_latest: return rerun_latest_main(collect_only)
    plan = load_plan(PLAN); STATE.mkdir(parents=True, exist_ok=True)
    if audit_only:
        result = audit.remote(); save_json(STATE / 'audit.json', result)
        print(json.dumps(result), flush=True); return
    if collect_only:
        saved = json.loads((STATE / 'call.json').read_text())
        if saved['plan_sha256'] != plan['plan_sha256']: raise ValueError('Different saved plan')
        call = modal.FunctionCall.from_id(saved['call_id'])
    else:
        from modal_credits import billing_row
        balance = billing_row('trenton', 'this month', Decimal(30))
        if balance['status'] != 'ok' or float(balance['estimated_remaining_usd']) < 6:
            raise RuntimeError('Insufficient verified credit margin')
        from modal_tokenizer_pool_transfer import source_client
        local_status('staging existing tokenizers cloud-to-cloud')
        with volume.batch_upload(force=True) as batch: batch.put_file(PLAN, PREFIX + '/plan.json')
        _, secret = source_client('group-cron')
        list(copy_file.with_options(secrets=[secret]).starmap([(i, 'anytok-corpus') for i in plan['native_files']], order_outputs=False))
        client, secret = source_client('csproject')
        src = modal.Volume.from_name('anytok-pool-20k', client=client)
        manifest = json.loads(b''.join(src.read_file('input/manifest.json')))
        files = [{'source': 'input/eval/' + d + '.jsonl.gz', 'destination': 'corpus/' + d + '.jsonl.gz',
                  **manifest['files']['eval/' + d + '.jsonl.gz']} for d in plan['domains']]
        list(copy_file.with_options(secrets=[secret]).starmap([(i, 'anytok-pool-20k') for i in files], order_outputs=False))
        published = list(fetch_published.map([c for c in plan['candidates'] if c['kind'] == 'published-native']))
        save_json(STATE / 'published.json', published)
        local_status('starting H200 screen')
        call = search.spawn()
        save_json(STATE / 'call.json', {'call_id': call.object_id, 'plan_sha256': plan['plan_sha256']})
    while True:
        try:
            report = call.get(timeout=5); break
        except (TimeoutError, modal.exception.TimeoutError):
            try:
                state = json.loads(b''.join(volume.read_file(PREFIX + '/status.json')))
                save_json(STATE / 'status.json', state)
            except FileNotFoundError: pass
    save_json(STATE / 'report.json', report)
    local_status('complete', completed=len(report['candidates']), total=len(report['candidates']))
    print(json.dumps({'qualified': [{k: v for k, v in r.items() if k not in ['nats', 'canonical_nats', 'document_ids', 'continuation_bytes']} for r in report['qualified']],
                      'confirmation_pairs': len(report['confirmation']), 'errors': report['evaluation_errors']}), flush=True)
