#!/bin/bash
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@900
#SBATCH --job-name=anytok-regret-20k
#SBATCH --output=/gpfs/scrubbed/%u/anytok-runs/regret-rl-20k/slurm-%j.log
#
# Train the regret continuation from its latest full restart to update 20,000
# (continue_regret_rl.py). Fifteen minutes before the walltime the trainer gets SIGTERM, saves a full
# restart and exits; the driver then queues one follow-up allocation (at most two in total) before
# the final uploads. Requires a passed resume check (check-receipt.json).
# No CPU or memory reservations (project rule).

set -euo pipefail
USER=${USER:-$(id -un)}
P=/gpfs/projects/stf/$USER
STATE=/gpfs/scrubbed/$USER/anytok-runs/regret-rl-20k
ENV=$P/anytok-data/env/train

# As the training jobs ran: no login-shell Python or numerics overrides. The driver reads the
# upload token from its file and keeps it out of the trainer's environment.
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE PYTHONOPTIMIZE TORCH_ALLOW_TF32_CUBLAS_OVERRIDE LD_PRELOAD
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HOME=$STATE/.hf-home HF_XET_CACHE=$STATE/.hf-home/xet HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export LD_LIBRARY_PATH="$ENV/hyak-libpython:${LD_LIBRARY_PATH:-}"

date -u
nvidia-smi -L
cd "$STATE"
exec "$ENV/bin/python" -u "$STATE/code/continue_regret_rl.py"
