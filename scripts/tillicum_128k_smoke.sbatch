#!/bin/bash
#SBATCH --account=zlab
#SBATCH --partition=gpu-h200
#SBATCH --qos=debug
#SBATCH --job-name=anytok-128k-smoke
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=/gpfs/scrubbed/qxli2/anytok-h200-smoke/modal_results/tillicum_smoke/slurm-%j.log

set -euo pipefail
cd /gpfs/scrubbed/qxli2/anytok-h200-smoke
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="/gpfs/home/qxli2/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib:${LD_LIBRARY_PATH:-}"
exec .venv-memory/bin/python -u scripts/tillicum_128k_smoke.py
