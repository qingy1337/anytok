#!/usr/bin/env bash
# Collect results once both detached diagnostic workers have finished.
set -euo pipefail
cd /home/zli/anytok
source /home/zli/.venv/bin/activate
for ((attempt=0; attempt<240; attempt++)); do
    if ! tmux has-session -t zett-length-qingy-v2 2>/dev/null && \
       ! tmux has-session -t zett-length-code-v2 2>/dev/null; then
        break
    fi
    sleep 60
done
if (( attempt == 240 )); then
    echo 'Timed out waiting for the length diagnostic workers' >&2
    exit 1
fi
python -c 'import json; from pathlib import Path
for filename,caps in (("modal_results/zett_length_qingy.log",{128,2048}),("modal_results/zett_length_code.log",{512,8192})):
    got=set()
    for line in Path(filename).read_text(errors="replace").splitlines():
        try: x=json.loads(line)
        except json.JSONDecodeError: continue
        if isinstance(x,dict) and x.get("stage")=="complete": got.add((x["positions"],x["hypernetwork_step"],x["lm_step"]))
    expected={(cap,hn,lm) for cap in caps for hn,lm in ((5000,5000),(10000,10000),(20000,20000),(5000,20000),(20000,5000))}
    if got != expected: raise SystemExit(f"Incomplete diagnostic in {filename}: missing {sorted(expected-got)}")'
python plotting_scripts/plot_zett_length_swap.py --sync
