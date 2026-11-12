"""Evaluate RL20k on all 1,000 native training-pool tokenizers, using B300s.

Only metadata traverses the local caller. Original corpora, tokenizers and
weights are copied directly from gildmember to this persistent cloud volume.
No CPU or memory reservations; no new training or tokenizer generation.
"""
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import uuid

import modal
from pool_generalization import digest, file_sha, save_json
from uniform_trajectory import PROFILE_DOMAINS

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'modal_results/rl_pool_20k'
PROFILE = 'blackpantherx86'
GPU = 'B300'
NAME = 'anytok-rl-all-pool-20k'
app = modal.App(NAME)
volume = modal.Volume.from_name(NAME, create_if_missing=True, version=2)
MOUNT = Path('/sweep')
SOURCE = ROOT / 'runtime/anytok_pool_20k'
cpu_image = (modal.Image.debian_slim(python_version='3.12')
    .uv_pip_install('tokenizers==0.22.2', 'numpy==2.3.5')
    .env({'TOKENIZERS_PARALLELISM': 'false'})
    .add_local_dir(SOURCE / 'hf_hypernet', '/opt/zett/hf_hypernet')
    .add_local_python_source('pool_generalization', 'uniform_trajectory',
                             'uniform_weakness', 'pool_rl_sweep'))
gpu_image = (modal.Image.from_registry('ubuntu:24.04', add_python='3.12')
    .apt_install('libpython3.12')
    .uv_pip_install('torch==2.15.0', 'transformers==5.15.1', 'tokenizers==0.22.2',
                    'numpy==2.3.5', 'safetensors==0.8.0', 'accelerate==1.14.0')
    .env({'TOKENIZERS_PARALLELISM': 'false'})
    .add_local_dir(SOURCE / 'hf_hypernet', '/opt/zett/hf_hypernet')
    .add_local_file(SOURCE / 'rust_utils/target/release/librust_utils.so',
                    '/opt/zett/rust_utils/target/release/librust_utils.so')
    .add_local_python_source('pool_generalization', 'uniform_trajectory',
                             'uniform_weakness', 'pool_ood_eval', 'pool_rl_sweep'))


def now():
    return datetime.now(timezone.utc).isoformat()


def status(stage, completed=0, **fields):
    row = {'at_utc': now(), 'stage': stage, 'completed': completed, 'total': 1000,
           'profile': PROFILE, **fields}
    save_json(STATE / 'status.json', row)
    with (STATE / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def read_json(path):
    return json.loads(Path(path).read_text())


@app.function(image=cpu_image, volumes={'/sweep': volume}, max_containers=40,
              scaledown_window=2, timeout=1800)
def copy_batch(items):
    volume.reload()
    from grpclib.exceptions import StreamTerminatedError
    client = modal.Client.from_credentials(os.environ['SOURCE_MODAL_TOKEN_ID'],
                                           os.environ['SOURCE_MODAL_TOKEN_SECRET'])
    sources = {}
    results = []
    for item in items:
        name, source_path, destination, expected = item
        if name not in sources:
            sources[name] = modal.Volume.from_name(name, version=2, client=client)
        target = MOUNT / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        if (target.is_file() and target.stat().st_size == expected['bytes']
                and file_sha(target) == expected['sha256']):
            results.append({'path': destination, **expected, 'reused': True})
            continue
        partial = target.with_name(target.name + '.' + uuid.uuid4().hex + '.partial')
        for attempt in range(4):
            try:
                h = hashlib.sha256()
                with partial.open('wb') as stream:
                    for block in sources[name].read_file(source_path):
                        stream.write(block)
                        h.update(block)
                actual = {'bytes': partial.stat().st_size, 'sha256': h.hexdigest()}
                if actual != expected:
                    raise ValueError('Cloud transfer checksum mismatch: ' + source_path)
                partial.replace(target)
                break
            except (OSError, StreamTerminatedError, modal.exception.ConnectionError, ValueError):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        results.append({'path': destination, **expected, 'reused': False})
    volume.commit()
    return results


@app.function(image=cpu_image, volumes={'/sweep': volume}, timeout=300)
def initial_panel(spec):
    volume.reload()
    import gzip
    from pool_rl_sweep import make_panel
    from uniform_trajectory import CanonicalTokenizer
    canonical = CanonicalTokenizer(MOUNT / 'input/model/tokenizer.json')
    sources = {}
    for domain in PROFILE_DOMAINS:
        path = MOUNT / f'input/eval/{domain}.jsonl.gz'
        if file_sha(path) != spec['input_hashes'][f'eval/{domain}.jsonl.gz']:
            raise ValueError('Heldout corpus changed')
        with gzip.open(path, 'rt') as stream:
            sources[domain] = [json.loads(line) for line in stream if line.strip()]
    panel = make_panel(sources, canonical, count=spec['documents_per_domain'])
    save_json(MOUNT / 'initial-panel.json', panel)
    volume.commit()
    return {'panel_sha256': digest(panel), 'documents_per_split': len(panel['screen'])}


@app.function(image=cpu_image, volumes={'/sweep': volume}, max_containers=100,
              scaledown_window=2, timeout=1800)
def prepare_features(indices, panel_path, spec):
    """Full vocab features, plus loss-independent common-context constraints."""
    volume.reload()
    sys.path.insert(0, '/opt/zett')
    import numpy as np
    from hf_hypernet.tokenizer_inputs import build_surface_forms, missing_byte_symbols
    from pool_rl_sweep import flat_panel, fit_start
    from uniform_trajectory import CanonicalTokenizer, load_candidate
    canonical = CanonicalTokenizer(MOUNT / 'input/model/tokenizer.json')
    config = read_json(MOUNT / 'checkpoint/hypernet/config.json')
    symbols = missing_byte_symbols(canonical)
    if symbols != config['hn_byte_fallback_symbols']:
        raise ValueError('Byte fallback configuration changed')
    fallbacks = {s: len(canonical) + 1 + i for i, s in enumerate(symbols)}
    panel = read_json(MOUNT / panel_path)
    rows = flat_panel(panel)
    starts = [0] * len(rows)
    maximum_tokens = 0
    for index in indices:
        expected = spec['tokenizers'][index]
        path = MOUNT / f'input/pool/tokenizers/{index:04d}/tokenizer.json'
        if file_sha(path) != expected['sha256']:
            raise ValueError('Pool tokenizer changed')
        tokenizer, tokens, priors, specials, forbidden = load_candidate(path, 'pool')
        if len(tokens) != expected['vocab_size']:
            raise ValueError('Native vocabulary size mismatch')
        folder = MOUNT / f'features/{index:04d}'
        manifest_path = folder / 'manifest.json'
        if not manifest_path.exists():
            folder.mkdir(parents=True, exist_ok=True)
            forms, truncated = build_surface_forms(tokens, canonical, 10, fallbacks, specials)
            np.save(folder / 'forms.npy', forms)
            np.save(folder / 'priors.npy', np.asarray(priors, dtype=np.float32))
            save_json(manifest_path, {'tokenizer_sha256': expected['sha256'],
                'forms_sha256': file_sha(folder / 'forms.npy'),
                'priors_sha256': file_sha(folder / 'priors.npy'), 'truncated': truncated,
                'max_pieces': 10, 'fallbacks': fallbacks})
        for i, row in enumerate(rows):
            start = fit_start(row, canonical, tokenizer, forbidden)
            starts[i] = max(starts[i], start)
            if start == 0:
                maximum_tokens = max(maximum_tokens,
                    len(tokenizer.encode(row['prefix'], add_special_tokens=False).ids))
    volume.commit()
    return {'indices': indices, 'panel_sha256': digest(panel), 'required_starts': starts,
            'maximum_target_prefix_tokens': maximum_tokens}


@app.function(image=cpu_image, volumes={'/sweep': volume}, timeout=120)
def reduce_constraints(panel_path, starts, destination):
    volume.reload()
    from pool_rl_sweep import trim_panel
    from uniform_trajectory import CanonicalTokenizer
    canonical = CanonicalTokenizer(MOUNT / 'input/model/tokenizer.json')
    panel = trim_panel(read_json(MOUNT / panel_path), starts, canonical)
    save_json(MOUNT / destination, panel)
    volume.commit()
    return {'panel_sha256': digest(panel), 'path': destination}


@app.cls(image=gpu_image, gpu=GPU, volumes={'/sweep': volume},
         max_containers=4, scaledown_window=2, timeout=1800)
class Scorer:
    @modal.enter()
    def load(self):
        volume.reload()
        sys.path.insert(0, '/opt/zett')
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from hf_hypernet.modeling_hypernet import ZettHypernet
        from hf_hypernet.smollm_setup import separate_input_embeddings_and_freeze_output
        from hf_hypernet.tokenizer_inputs import enable_byte_fallbacks
        torch.set_num_threads(1)
        manifest = read_json(MOUNT / 'checkpoint/manifest.json')
        spec = read_json(MOUNT / 'spec.json')
        if manifest != spec['checkpoint_manifest']:
            raise ValueError('Checkpoint manifest changed')
        for name in ['lm.pt', 'hypernet/config.json', 'hypernet/model.safetensors']:
            if file_sha(MOUNT / 'checkpoint' / name) != manifest['files'][name]['sha256']:
                raise ValueError('Checkpoint weights changed')
        self.canonical = AutoTokenizer.from_pretrained(MOUNT / 'input/model', local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(MOUNT / 'input/model', local_files_only=True,
            dtype=torch.float32, low_cpu_mem_usage=True).to('cuda')
        separate_input_embeddings_and_freeze_output(self.model)
        reference = self.model.get_input_embeddings().weight.detach().clone()
        self.model.load_state_dict(torch.load(MOUNT / 'checkpoint/lm.pt', map_location='cpu', weights_only=True))
        self.model.eval()
        self.table = self.model.get_input_embeddings().weight.detach()
        if not torch.equal(self.table, reference) or not torch.equal(self.model.get_output_embeddings().weight, reference):
            raise ValueError('Canonical embedding tables changed')
        del reference
        self.hypernet = ZettHypernet.from_pretrained(MOUNT / 'checkpoint/hypernet', local_files_only=True).to('cuda').eval()
        self.fallbacks = enable_byte_fallbacks(self.hypernet, self.canonical)
        self.canonical_cache = {}
        self.device_name = torch.cuda.get_device_name()
        print(json.dumps({'at_utc': now(), 'stage': 'RL20k loaded',
            'gpu': self.device_name, 'cuda': torch.version.cuda}), flush=True)

    @modal.method()
    def score(self, indices, split, spec):
        volume.reload()
        import numpy as np
        import torch
        from pool_rl_sweep import summarize_candidate
        from pool_ood_eval import examples_for, score_documents
        from uniform_trajectory import load_candidate
        panel = read_json(MOUNT / 'panel.json')[split]
        signature = digest(panel)
        if signature != spec['panel_hashes'][split]:
            raise ValueError('Frozen panel changed')
        control = examples_for(panel)
        if signature not in self.canonical_cache:
            self.canonical_cache[signature] = score_documents(self.model, control, self.table, self.table, batch_size=16)
        baseline = self.canonical_cache[signature]
        results = []
        for index in indices:
            target_path = MOUNT / f'{split}/{index:04d}.json'
            if target_path.exists():
                old = read_json(target_path)
                if old['evaluation_sha256'] != digest(spec) or old['panel_sha256'] != signature:
                    raise ValueError('Existing scores belong to a different evaluation')
                results.append(old)
                continue
            begin = time.monotonic()
            expected = spec['tokenizers'][index]
            path = MOUNT / f'input/pool/tokenizers/{index:04d}/tokenizer.json'
            if file_sha(path) != expected['sha256']:
                raise ValueError('Native tokenizer changed')
            tokenizer, _, _, _, forbidden = load_candidate(path, 'pool')
            examples = examples_for(panel, tokenizer)
            if any(forbidden.intersection(e[0]) for e in examples):
                raise ValueError('Special/unknown token in prefix')
            folder = MOUNT / f'features/{index:04d}'
            manifest = read_json(folder / 'manifest.json')
            if manifest['fallbacks'] != self.fallbacks or manifest['tokenizer_sha256'] != expected['sha256']:
                raise ValueError('Native feature mapping differs')
            for name in ['forms', 'priors']:
                if file_sha(folder / (name + '.npy')) != manifest[name + '_sha256']:
                    raise ValueError('Feature arrays changed')
            forms = torch.from_numpy(np.load(folder / 'forms.npy').astype(np.int64)).to('cuda')
            priors = torch.from_numpy(np.load(folder / 'priors.npy')).to('cuda')
            # Inter-token attention requires the entire vocabulary, even when
            # only a few native token IDs occur in these evaluation prefixes.
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                generated, output, _ = self.hypernet(forms, priors, self.table, None, deterministic=True)
            if output is not None or not torch.isfinite(generated).all():
                raise ValueError('Invalid transferred embedding output')
            nats = score_documents(self.model, examples, generated, self.table, batch_size=16)
            result = summarize_candidate(index, expected['config'], panel, nats, baseline)
            result.update(evaluation_sha256=digest(spec), tokenizer_sha256=expected['sha256'],
                gpu=self.device_name,
                elapsed_seconds=time.monotonic() - begin,
                prefix_tokens=sum(len(e[0]) for e in examples),
                maximum_positions=max(1 + len(e[0]) + len(e[1]) for e in examples),
                target_examples_sha256=digest(examples))
            save_json(target_path, result)
            volume.commit()
            results.append(result)
            print(json.dumps({'at_utc': now(), 'stage': 'scored tokenizer', 'index': index,
                'split': split, 'mean_domain_bpb': result['mean_domain_bpb'],
                'mean_domain_gap_bpb': result['mean_domain_gap_bpb'],
                'elapsed_seconds': result['elapsed_seconds']}), flush=True)
            del forms, priors, generated
            torch.cuda.empty_cache()
        return results


def upload_json(name, value):
    with volume.batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO((json.dumps(value, indent=2) + '\n').encode()), name)


def plan_inputs():
    """Read only metadata locally; every large transfer runs in Modal."""
    from modal_tokenizer_pool_transfer import source_client
    client, secret = source_client('gildmember')
    training = modal.Volume.from_name('anytok-pool-20k', version=2, client=client)
    rl = modal.Volume.from_name('anytok-rl-ood-20k', version=2, client=client)
    def get(volume, name):
        raw = b''.join(volume.read_file(name))
        return json.loads(raw), {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    original, original_meta = get(training, 'input/manifest.json')
    pool, pool_meta = get(training, 'input/pool/full-results.json')
    base = read_json(ROOT / 'modal_results/rl_ood_20k/evaluation-spec.json')
    prefix = base['run_id'] + '/checkpoint/'
    checkpoint, checkpoint_meta = get(rl, prefix + 'manifest.json')
    if checkpoint != base['checkpoint_manifest'] or checkpoint['step'] != 20000:
        raise ValueError('Wrong RL checkpoint')
    if len(pool['tokenizers']) != 1000 or pool['completed'] != 1000:
        raise ValueError('Original 1000-member pool required')
    files = ['model/config.json', 'model/model.safetensors', 'model/tokenizer.json']
    files += [f'eval/{domain}.jsonl.gz' for domain in PROFILE_DOMAINS]
    items = [('anytok-pool-20k', 'input/' + name, 'input/' + name, original['files'][name]) for name in files]
    items += [('anytok-pool-20k', 'input/manifest.json', 'input/manifest.json', original_meta),
              ('anytok-pool-20k', 'input/pool/full-results.json', 'input/pool/full-results.json', pool_meta),
              ('anytok-rl-ood-20k', prefix + 'manifest.json', 'checkpoint/manifest.json', checkpoint_meta)]
    items += [('anytok-rl-ood-20k', prefix + name, 'checkpoint/' + name, checkpoint['files'][name])
              for name in ['lm.pt', 'hypernet/config.json', 'hypernet/model.safetensors']]
    tokenizers = sorted(pool['tokenizers'], key=lambda r: r['config']['index'])
    if [r['config']['index'] for r in tokenizers] != list(range(1000)):
        raise ValueError('Pool index mapping changed')
    for row in tokenizers:
        name = f"input/pool/tokenizers/{row['config']['index']:04d}/tokenizer.json"
        items.append(('anytok-pool-20k', name, name,
                      {'bytes': row['tokenizer_bytes'], 'sha256': row['tokenizer_sha256']}))
    spec = {'version': 1, 'checkpoint_manifest': checkpoint, 'documents_per_domain': 32,
            'position_cap': 128, 'continuation_tokens': 32, 'domains': list(PROFILE_DOMAINS),
            'pool_index_sha256': pool_meta['sha256'], 'pool_plan_sha256': pool['plan_sha256'],
            'input_hashes': {name: original['files'][name]['sha256'] for name in files},
            'tokenizers': [{'index': r['config']['index'], 'sha256': r['tokenizer_sha256'],
                'vocab_size': r['vocab_size_actual'], 'config': r['config']} for r in tokenizers],
            'primary': 'Equal-weight mean of nine domain BPBs; largest is worst.',
            'confirmation': 'Independent 32 passages/domain for top10 overall and each domain winner.',
            'context': 'Identical raw prefix, document IDs, 32 canonical suffix IDs and bytes for all1000; prefixes trimmed jointly until every tokenizer fits.'}
    return spec, items, secret


def run(action):
    if os.environ.get('MODAL_PROFILE') != PROFILE:
        raise ValueError('Use ' + PROFILE)
    STATE.mkdir(parents=True, exist_ok=True)
    if action == 'plan':
        print(json.dumps({'profile': PROFILE, 'gpu': 'B300', 'workers': 4,
            'tokenizers': 1000, 'screen_passages_per_tokenizer': 288,
            'confirmation_passages_per_finalist': 288, 'cpu': None, 'memory': None}))
        return
    from modal_credits import billing_row
    balance = billing_row(PROFILE, 'this month', Decimal(30))
    save_json(STATE / 'credits-before.json', balance)
    if balance['status'] != 'ok' or Decimal(balance['estimated_remaining_usd']) < Decimal('5'):
        raise ValueError('Insufficient verified evaluation credit')
    spec_path = STATE / 'spec.json'
    if action in ['prepare', 'contexts']:
        spec, items, secret = plan_inputs()
        save_json(spec_path, spec)
        save_json(STATE / 'copy-plan.json', items)
        total = len(items)
        status('copying saved assets directly between cloud volumes', files_completed=0, files_total=total)
        copied = []
        # Large model files get their own stream; small tokenizer files travel
        # in batches to avoid starting 1,000 transfer containers.
        batches = [[item] for item in items if item[3]['bytes'] > 100_000_000]
        small = [item for item in items if item[3]['bytes'] <= 100_000_000]
        batches += [small[i:i + 25] for i in range(0, len(small), 25)]
        if action == 'contexts':
            copied = read_json(STATE / 'copied.json')
            if {r['path'] for r in copied} != {i[2] for i in items}:
                raise ValueError('Complete verified transfer required before context recovery')
        else:
            for result in copy_batch.with_options(secrets=[secret]).map(batches, order_outputs=False):
                copied.extend(result)
                status('copying saved assets directly between cloud volumes', files_completed=len(copied), files_total=total)
            save_json(STATE / 'copied.json', copied)
        upload_json('spec.json', spec)
        metadata = initial_panel.remote(spec)
        save_json(STATE / 'initial-panel-metadata.json', metadata)
        groups = [list(range(i, min(i + 10, 1000))) for i in range(0, 1000, 10)]
        panel_path = 'initial-panel.json'
        for round_index in range(8):
            constraints = []
            done = 0
            for result in prepare_features.map(groups, [panel_path] * len(groups), [spec] * len(groups), order_outputs=False):
                constraints.append(result)
                done += len(result['indices'])
                status('preparing features / checking common contexts', round=round_index, validated_tokenizers=done)
            save_json(STATE / f'constraints-{round_index}.json', constraints)
            starts = [max(values) for values in zip(*(r['required_starts'] for r in constraints))]
            destination = 'panel.json' if not any(starts) else f'panel-round-{round_index}.json'
            reduced = reduce_constraints.remote(panel_path, starts, destination)
            panel_path = destination
            if not any(starts):
                break
        else:
            raise ValueError('Common prefixes did not converge; no model scores launched')
        # Only IDs/counts/hash summaries return locally; raw heldout text stays cloud-side.
        spec['panel_sha256'] = reduced['panel_sha256']
        # The CPU helper reports split hashes without transferring passage text.
        hashes = panel_metadata.remote()
        spec.update(hashes)
        upload_json('spec.json', spec)
        save_json(spec_path, spec)
        status('ready for B300 scoring')
        return
    spec = read_json(spec_path)
    if action not in ['pilot', 'run', 'confirm']:
        raise ValueError('Unknown action')
    if 'panel_hashes' not in spec:
        raise ValueError('Prepare frozen contexts first')
    if spec.get('execution_gpu') != GPU:
        raise ValueError('Frozen evaluation GPU differs from requested GPU')
    scorer = Scorer()
    if action == 'pilot':
        status('loading RL20k / scoring B300 pilot')
        started = time.monotonic()
        result = scorer.score.remote([0, 616, 822], 'screen', spec)
        save_json(STATE / 'pilot.json', {'rows': result, 'gpu': GPU,
            'elapsed_seconds': time.monotonic() - started})
        status('B300 pilot complete', len(result), seconds_per_tokenizer=sum(r['elapsed_seconds'] for r in result) / len(result))
        return
    if action == 'run':
        from pool_rl_sweep import rank_results
        rows = []
        groups = [list(range(i, min(i + 10, 1000))) for i in range(0, 1000, 10)]
        started = time.monotonic()
        status('scoring all training-pool tokenizers')
        for batch in scorer.score.map(groups, ['screen'] * len(groups), [spec] * len(groups), order_outputs=False):
            rows.extend(batch)
            save_json(STATE / 'screen-partial.json', {'rows': rows, 'evaluation_sha256': digest(spec)})
            elapsed = time.monotonic() - started
            status('scoring all training-pool tokenizers', len(rows),
                elapsed_seconds=elapsed, eta_seconds=elapsed / len(rows) * (1000 - len(rows)))
        ranked = rank_results(rows, range(1000))
        result = {'evaluation_sha256': digest(spec), 'rows': ranked,
                  'primary': spec['primary'], 'at_utc': now()}
        save_json(STATE / 'screen.json', result)
        upload_json('screen.json', result)
        selected = {r['index'] for r in ranked[:10]}
        for domain in PROFILE_DOMAINS:
            selected.add(max(ranked, key=lambda r: r['domains'][domain]['gap_bpb'])['index'])
        save_json(STATE / 'finalists.json', sorted(selected))
        status('screen complete; confirming worst candidates', 1000, state='running')
        # Confirmation is small and uses the already loaded model containers.
        confirmed = []
        selected = sorted(selected)
        groups = [selected[i:i + 5] for i in range(0, len(selected), 5)]
        for batch in scorer.score.map(groups, ['confirm'] * len(groups), [spec] * len(groups), order_outputs=False):
            confirmed.extend(batch)
        confirmation = {'evaluation_sha256': digest(spec),
            'rows': rank_results(confirmed, selected), 'selected_by_screen': selected}
        save_json(STATE / 'confirmation.json', confirmation)
        upload_json('confirmation.json', confirmation)
        upload_json('complete.json', {'screen_tokenizers': 1000, 'confirmed_tokenizers': len(confirmed),
            'screen_sha256': digest(result), 'confirmation_sha256': digest(confirmation),
            'evaluation_sha256': digest(spec), 'at_utc': now()})
        status('complete', 1000)
        return
    # Recover confirmation without rerunning the full pool if a caller disconnects.
    from pool_rl_sweep import rank_results
    selected = read_json(STATE / 'finalists.json')
    confirmed = scorer.score.remote(selected, 'confirm', spec)
    result = {'evaluation_sha256': digest(spec), 'rows': rank_results(confirmed, selected),
              'selected_by_screen': selected}
    save_json(STATE / 'confirmation.json', result)
    upload_json('confirmation.json', result)
    status('complete', 1000)


@app.function(image=cpu_image, volumes={'/sweep': volume}, timeout=120)
def panel_metadata():
    volume.reload()
    panel = read_json(MOUNT / 'panel.json')
    summary = {}
    for split in ['screen', 'confirm']:
        rows = panel[split]
        summary[split] = {'documents': len(rows), 'continuation_tokens': 32 * len(rows),
            'continuation_bytes': sum(r['continuation_bytes'] for r in rows),
            'document_ids': [r['id'] for r in rows], 'domains': [r['domain'] for r in rows],
            'byte_counts': [r['continuation_bytes'] for r in rows],
            'canonical_prefix_tokens': [len(r['canonical_prefix_ids']) for r in rows]}
    return {'panel_hashes': {s: digest(panel[s]) for s in ['screen', 'confirm']}, 'panels': summary}


@app.local_entrypoint()
def main(action: str = 'plan'):
    try:
        run(action)
    except BaseException as error:
        previous = read_json(STATE / 'status.json') if (STATE / 'status.json').exists() else {}
        STATE.mkdir(parents=True, exist_ok=True)
        status('failed', previous.get('completed', 0), state='failed', error=str(error))
        raise
