# Tillicum native 128k GPU-only fit smoke

User explicitly authorized a very short Tillicum smoke test to check whether
the GPU-only optimized execution fits the larger GPU. Do not start a training
migration or modify the existing Modal run.

Use one full H200, account zlab (after default stf rejection), debug QoS, five-minute wall limit.
No explicit CPU or system-memory requests. Stage code and environment before
requesting GPU time. The exact step 2000 checkpoint (including populated Adam
and RNG) and original corpus/baseLM/full 1,000-tokenizer pool transfer directly
from queues/anytok-pool-rl-20k to an isolated Tillicum scratch directory. Only
signed metadata and small code packages pass through the local caller. Verify
every source checksum before submission. Cloud originals remain authoritative.

Use PyTorch 2.15.0+cu132, transformers 5.15.1, tokenizers 0.22.2 and Python 3.12
in an isolated environment. The existing Rust helper is compatible with the
cluster's glibc 2.34 and its Python library; import preflight passed.

Run scripts/local_memory_trial.py with variant chunked_norm_mlp, complete
native tokenizer #989 (128000 entries), batch 128, microbatch 32, maximum 128 LM
positions and unchanged losses/canonical head. One full warmup, then reload
the exact checkpoint/corpus/RNG for one measured optimizer update. Disable
allocation stack tracing for timing, retain peak counters. No optimizer,
LM-gradient or activation CPU offload; no vocabulary truncation. Record OOM
as a valid negative fit outcome rather than silently changing the workload.

Track staging, Slurm job ID/state, phase logs and final allocated/reserved
GPU peaks in plans/run-dashboard.json. Stop after this one smoke job. Preserve
its result, hardware/runtime metadata and source provenance in a report.

The complete 1,070-file transfer was checksum-verified. Initial submission to
stf was rejected because the user's budget dates are invalid/expired; no GPU
allocation was created. Use the user's existing zlab account (73 prior jobs
in the current cycle) for the same single five-minute job. It has no prepaid
credits, so this is normal usage billing, bounded to about $0.075 at the
published $0.90/GPU-hour rate. No other resource or workload change.

Completed job 469536 on g006 (H200), account zlab; one GPU, 77 seconds elapsed,
exit 0. Full 128k GPU-only warmup and measured update both completed with exact
same-device input/loss/RNG equality. Peak allocated 115.87 GiB, reserved 136.19 GiB,
capacity 139.81 GiB; measured model step 6.34 s. No CPU offload. Scheduler defaults
assigned 4 CPUs/120G RAM; the sbatch script has no CPU or memory directives.
No additional GPU job or training migration was launched.
