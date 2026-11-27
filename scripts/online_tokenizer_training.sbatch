#!/bin/bash
#SBATCH --partition=gpu-h200
#SBATCH --job-name=anytok-direct-joint
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@300

set -euo pipefail
runner=${ANYTOK_PYTHON:-python}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$("$runner" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'):${LD_LIBRARY_PATH:-}"
exec "$runner" -u scripts/run_online_tokenizer_training.py "$@"
