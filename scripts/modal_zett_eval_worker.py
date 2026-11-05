"""Score one saved checkpoint on the fixed FineWeb-Edu tokenizer panel."""
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from transformers import AutoTokenizer
from scripts import run_fineweb_specialty_48k_retrain_eval as ood
from scripts import run_fineweb_base_64k_eval as base
from scripts.evaluate_joint_external_32 import digest

step = int(sys.argv[1])
root = Path('/work')
run = root / 'run'
out = Path('/results') / f'step-{step:06d}'
corpus_path = root / 'corpus.jsonl.gz'
examples = root / 'panel/transfer_examples.jsonl.gz'
specs = root / 'tokenizers.json.gz'
model = root / 'model'
baseline = out / 'zett32'
out.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, '/opt/zett/scripts/evaluate_joint_external_panel.py',
    '--run', str(run), '--checkpoint', str(run / f'step-{step:06d}'),
    '--corpus', str(corpus_path), '--examples', str(examples), '--specs', str(specs),
    '--output', str(baseline / f'step-{step:06d}'), '--sources', 'fineweb-edu',
    '--indices', '0', '4', '8', '12', '16', '20', '24', '28'], check=True)
ood.RUN = run
ood.CORPUS = corpus_path
ood.EXAMPLES = examples
ood.MODELS = root / 'specialty'
ood.OUTPUT = out / 'ood48'
ood.BASELINES = baseline
ood.EXPECTED = (1936, 235503, 1088789)
ood.OUTPUT.mkdir(parents=True, exist_ok=True)
source = AutoTokenizer.from_pretrained(model, local_files_only=True)
corpus, prepared = ood.load_panel(source)
paths = ood.tokenizer_paths()
hashes = {name: digest(path) for name, path in paths.items()}
ood.evaluate_step(step, corpus, prepared, source, paths, digest(corpus_path),
                  digest(examples), hashes, str(model))
base.RUN = run
base.OUTPUT = out / 'base64'
base.MODEL_DIR = root / 'base64'
base.EXPECTED = ood.EXPECTED
base.ROOT = root
base.OUTPUT.mkdir(parents=True, exist_ok=True)
# validate_tokenizer compares the original 48k and 64k training manifests.
base.ROOT = root / 'base_root'
tok, tokens, tok_hash, manifest_hash = base.validate_tokenizer()
target = base.prepared_target(tok, corpus, prepared)
base.score_step(step, {'model': str(model)}, source, tokens, target,
                digest(corpus_path), digest(examples), tok_hash, manifest_hash)
print(json.dumps({'stage': 'evaluation_complete', 'step': step}), flush=True)
