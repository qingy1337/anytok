#!/usr/bin/env bash
# Archive each finished Modal profile's 500-step grid to the public HF model.
set -euo pipefail

cd "$(dirname "$0")/.."
source /home/zli/.venv/bin/activate
export PYTHONUNBUFFERED=1

archive_profile() {
    local profile="$1"
    echo "{\"stage\":\"archiving_${profile}\"}"
    MODAL_PROFILE="$profile" modal run --detach scripts/modal_zett_20k_hf.py --action archive
}

wait_for_stage() {
    local wanted="$1"
    local stage
    for ((attempt=0; attempt<720; attempt++)); do
        stage=$(python -c 'import json; from pathlib import Path; p=Path("modal_results/zett_20k_controller/status.json"); s=json.loads(p.read_text()); print(s.get("stage",""), s.get("profile",""))')
        if [[ "$stage" == "failed " ]]; then
            echo "Training controller failed before ${wanted}" >&2
            return 1
        fi
        if [[ "$stage" == "$wanted" || "$stage" == "training_complete " ]]; then
            return 0
        fi
        sleep 60
    done
    echo "Timed out waiting for ${wanted}" >&2
    return 1
}

archive_profile projects
wait_for_stage 'segment_start test-joining'
archive_profile grading
wait_for_stage 'training_complete '
archive_profile test-joining
# A recovery may resume the last steps on grading after test-joining.
archive_profile grading

python -c 'import json; from huggingface_hub import HfApi,hf_hub_download; r="qingy2024/anytok-zett-smollm2-20k"; a=HfApi(); files=set(a.list_repo_files(r)); missing=[s for s in range(500,20001,500) if f"checkpoints/step-{s:06d}/manifest.json" not in files]; latest=json.load(open(hf_hub_download(r,"latest.json"))); assert not missing,missing; assert latest["step"]==20000,latest; print(json.dumps({"stage":"archive_complete","checkpoint_count":40,"latest_step":20000}))'
