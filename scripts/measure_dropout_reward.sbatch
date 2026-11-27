#!/bin/bash
#SBATCH --partition=gpu-h200
#SBATCH --job-name=anytok-dropout-check
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00

set -euo pipefail
runner=${ANYTOK_PYTHON:-python}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$($runner -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'):${LD_LIBRARY_PATH:-}"
exec "$runner" -u scripts/measure_dropout_reward.py "$@"
