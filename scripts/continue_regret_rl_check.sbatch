#!/bin/bash
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=anytok-regret-check
#SBATCH --output=/gpfs/scrubbed/%u/anytok-runs/regret-rl-20k/slurm-check-%j.log
#
# Resume check for the regret continuation: resume a hard-linked copy of the step-8,000 run folder
# to update 8,002, with no uploads and no resubmission (continue_regret_rl.py --check), and verify
# that update 8,001 reproduces the source run's logged tokenizer and batch.
# Before sbatch: fetch_regret_continuation_restart.py has filled run/, and code/ holds
# continue_regret_rl.py, resume_regret_continuation.py and both sbatch files.
# No CPU or memory reservations (project rule).

set -euo pipefail
USER=${USER:-$(id -un)}
P=/gpfs/projects/stf/$USER
STATE=/gpfs/scrubbed/$USER/anytok-runs/regret-rl-20k
ENV=$P/anytok-data/env/train

# As the training jobs ran: no login-shell Python or numerics overrides, and no token in the job.
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE PYTHONOPTIMIZE TORCH_ALLOW_TF32_CUBLAS_OVERRIDE LD_PRELOAD
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH="$ENV/hyak-libpython:${LD_LIBRARY_PATH:-}"

date -u
nvidia-smi -L
cd "$STATE"
exec "$ENV/bin/python" -u "$STATE/code/continue_regret_rl.py" --check
