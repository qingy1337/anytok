# Held-out full-pool eval · 4,992 scored tokens

Use this protocol to compare checkpoints on the **same held-out text across all 1,000 tokenizers**. It is the sweep run on Rift and Eagle2. The earlier [Regret RL 6.5k training-reward study](../regret-pool-ranking-6500/README.md) used 32 training batches per tokenizer and measured a different quantity.

Each checkpoint scores 156 frozen passages across nine domains. Each passage has 32 scored canonical continuation tokens, for **4,992 tokens and 14,429 bytes per tokenizer**. The prefix uses the selected tokenizer. The length cap is 128 positions including BOS, prefix, and continuation. The canonical continuation, output head, and scored bytes are shared. The hypernetwork runs with dropout off and generates one full embedding table per tokenizer. We score the canonical baseline once per checkpoint and report the mean of the nine domain BPB gaps.

The frozen plan hash is `d1269e6fee0c5875d475cfda7584b65f3f0c0cc9cb8a6161b518b756627a6964`. The [scorer](../../scripts/run_rift_pool_sweep.py) validates the panel, native prefix IDs, tokenizer hashes, and run identity. Use the checked-in [H200 Slurm script](../../scripts/heldout_pool_h200.sbatch) for Tillicum.

## Prepare on the login node

Use Python 3.12 and the pinned environment below. Stage files directly on Tillicum project storage. The [private Rift archive](https://huggingface.co/datasets/qingy2024/anytok-rift-eval-resume/tree/38176d7d654b33f513dcc7e4f3f19e6bfc142d85/paused-20261122T143018Z) requires read access. It contains the frozen panel, the exact model code, checkpoint file receipts, and the saved Rift scores.

```bash
repo="$PWD"                       # AnyTok checkout
stage=/path/to/project/anytok-heldout
work="$stage/sweep"               # fresh directory, created by restore
venv="$stage/venv"
prefix=paused-20261122T143018Z
mkdir -p "$stage"
uv venv --python 3.12 "$venv"
uv pip install --python "$venv/bin/python" \
  --extra-index-url https://download.pytorch.org/whl/cu132 --index-strategy unsafe-best-match \
  torch==2.15.0+cu132 transformers==5.15.1 tokenizers==0.22.2 numpy==2.3.5 \
  safetensors==0.8.0 huggingface-hub==1.33.0 requests

"$venv/bin/hf" download qingy2024/anytok-rift-eval-resume \
  "$prefix/sources.json" "$prefix/rift-sweep-resume.tar.gz" \
  --repo-type dataset --revision 38176d7d654b33f513dcc7e4f3f19e6bfc142d85 --local-dir "$stage/hf"
"$venv/bin/python" "$repo/scripts/restore_rift_pool_archive.py" \
  "$stage/hf/$prefix/rift-sweep-resume.tar.gz" "$stage/hf/$prefix/sources.json" "$work"
cp "$repo/scripts/run_rift_pool_sweep.py" "$repo/scripts/rift_eval_memory.py" "$work/"
"$venv/bin/python" "$repo/scripts/stage_rift_resume_inputs.py" "$work"
TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=1 \
  "$venv/bin/python" "$work/run_rift_pool_sweep.py" features
```

Save the checkpoint labels you want in `$stage/checkpoints.txt`, one per line, using the exact names in `$work/plan.json`. For labels present in `sources.json` under `checkpoints.items`, stage them on the login node as follows.

```bash
labels="$stage/checkpoints.txt"
while read -r label || [ -n "$label" ]; do
  "$venv/bin/python" "$repo/scripts/stage_rift_resume_inputs.py" "$work" "$label" || break
done < "$labels"
```

For other labels, use the original cloud-stored checkpoint files and matching receipt from `$work/resume-reference/checkpoints/<label>/ready.json`. Put the files and receipt under `$work/checkpoints/<label>/`. The preflight reads and verifies every checkpoint file against the archived receipt before any GPU submission.

```bash
"$venv/bin/python" "$work/run_rift_pool_sweep.py" verify-launch --checkpoint-list "$labels"
```

For 20 checkpoints the preflight should report `protocol: heldout-pool-4992`, `checkpoints: 20`, `pairs: 20000`, and `scored_tokens: 4992`. A failed preflight or a different count means the inputs need attention before submitting.

## Run on Tillicum H200

Run the memory smoke test once. It checks the largest 128,000-token vocabulary and leaves a GPU-specific receipt.

```bash
mkdir -p "$work/logs"
first=$(head -n 1 "$labels")
sbatch --account=zlab --time=00:05:00 --output="$work/logs/%x-smoke-%j.log" \
  "$repo/scripts/heldout_pool_h200.sbatch" smoke "$work" "$venv/bin/python" "$first"
```

Once the smoke job completes successfully, submit one array entry per checkpoint. Replace `19` with the last zero-based index if your list has a different length.

```bash
sbatch --account=zlab --array=0-19 --output="$work/logs/%x-%A_%a.log" \
  "$repo/scripts/heldout_pool_h200.sbatch" score "$work" "$venv/bin/python" "$labels"
```

H200 scores go to `$work/run/heldout-h200/<label>/`, separate from the archived Rift scores. Each checkpoint writes its per-passage losses, `ranking.csv`, and a `complete.json` receipt after all 1,000 tokenizers finish. The same protocol on a different GPU type can shift BPB slightly. For a checkpoint already started on Rift, the H200 job scores the entire checkpoint in its own output directory.
