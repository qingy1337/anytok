#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
    echo "Usage: $0 PROJECTS_APP_ID|--recover-cancelled" >&2
    exit 2
fi

cd /home/zli/anytok
source /home/zli/.venv/bin/activate
mkdir -p modal_results/zett_20k_controller
export PYTHONUNBUFFERED=1
if [[ "$1" == "--recover-cancelled" ]]; then
    python -u scripts/modal_zett_20k_controller.py --recover-cancelled \
        2>&1 | tee -a modal_results/zett_20k_controller/controller.log
else
    python -u scripts/modal_zett_20k_controller.py --wait-app "$1" \
        2>&1 | tee -a modal_results/zett_20k_controller/controller.log
fi
