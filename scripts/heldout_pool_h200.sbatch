#!/usr/bin/env bash
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --job-name=anytok-heldout-4992
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00

set -euo pipefail
action=${1:?Pass smoke or score}
workspace=${2:?Pass the absolute path to the prepared held-out workspace}
python=${3:?Pass the absolute path to the pinned Python executable}
selection=${4:?Pass a checkpoint label for smoke or a checkpoint-list path for score}

cd "$workspace"
export ANYTOK_POOL_GPU_NAME=H200
export ANYTOK_POOL_VERIFY_RESUME=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$("$python" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'):${LD_LIBRARY_PATH:-}"

case "$action" in
    smoke)
        exec "$python" -u run_rift_pool_sweep.py smoke --smoke-checkpoint "$selection"
        ;;
    score)
        mapfile -t labels < "$selection"
        index=${SLURM_ARRAY_TASK_ID:?Submit score as a Slurm array}
        label=${labels[index]:?Array index is missing from the checkpoint list}
        exec "$python" -u run_rift_pool_sweep.py score "h200-${SLURM_ARRAY_JOB_ID}-${index}" "$label" \
            --score-dir "$workspace/run/heldout-h200"
        ;;
    *)
        printf '%s\n' "Expected smoke or score, got $action" >&2
        exit 2
        ;;
esac
