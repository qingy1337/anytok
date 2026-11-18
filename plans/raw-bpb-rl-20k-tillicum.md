# Raw-BPB RL run on Tillicum (entropy weight 0, 20k)

Run the verified raw-BPB handoff for its default 20,000 updates on UW Hyak
Tillicum: trainer `runtime/anytok_pool_raw_bpb_rl_20k`, launcher
`scripts/run_raw_bpb_training.py` and the input bundle pinned in
`docs/raw-bpb-handoff/` (commit 670993b). The policy reward is the raw
transferred continuation BPB of each update's 128 paired training examples,
measured before the model update, with policy entropy weight 0. Everything else
is the handoff's: identity-warmed hypernetwork, corrected paired boundaries,
the registered 1,000-tokenizer pool and corpus, batch 128 and the GPU-only
memory path. It is the raw-BPB counterpart of the regret run, with the same
sanity evaluation, so the two can be compared through update 8,000, where the
regret run is paused. The original uniform run remains the baseline, and no
new uniform run is trained.

Status: complete on 2026-11-17 UTC, 20,000 of 20,000 updates, 39.9 GPU-hours.
Each job used one H200, 4 CPUs and 120G in chained 24-hour normal-QoS Slurm
jobs that resume from the newest full restart. The first attempt crashed while
saving update 5,000 because the cluster's shared scratch file system was over
its quota; the run resumed from the update-4,000 restart, copied to project
storage. The replay was not bit-identical: it matched through update 4,031 and
drew different tokenizers from 4,091. The event log keeps both versions, and
analyses keep the last row per update. The first restart job ended with a
graceful stop and full restart at update 16,409.

Outputs: a checkpoint every 500 updates plus 16,409, the event log, the final
status and the run configuration, on Hugging Face (`Dekiofsa/RLtokenizers`,
folder `raw-bpb-rl-20k`, revision 2e2cf342). Full restarts (state and
optimizer) are at 0, 2,000, …, 20,000 and 16,409. All 239 run files were
downloaded again and matched the cluster copy; the redacted run_config.json
was uploaded separately. The cluster wrappers are on the `hyak-tillicum-port`
branch of the fork Dekiofsa/anytok.

Report: `reports/raw-bpb-rl-20k-tillicum/`, built by
`scripts/report_raw_bpb_rl_20k_tillicum.py` from tables extracted read-only on
the cluster by `scripts/extract_raw_bpb_rl_20k_tillicum.py`. Dashboard run:
`raw-bpb-entropy0-training`, Training; it reads the final status from the
report folder.

Not done: no final evaluation of this checkpoint and no full-pool hardness
measurement of its model. Its sanity evaluation is not directly comparable with
the uniform run's, which cut its sanity passages differently.
