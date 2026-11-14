# Tillicum 128k GPU-only smoke test

**Fits on NVIDIA H200.** Outcome: `complete`. Slurm job `469536`; account `zlab`.

Peak PyTorch allocated GPU memory: **115.87 GiB**. Peak reserved: **136.19 GiB**. Device-reported capacity: **139.81 GiB**.

Timed model/optimizer phases: **6.34s** (complete measured update).

## Exact workload

Same step 2000 trained LM/hypernetwork, populated Adam state, original corpora and complete 1,000-tokenizer pool as the local diagnostics. Native tokenizer #989 has 128,000 entries; batch 128, microbatch 32, maximum 128 LM positions, unchanged paired boundaries, canonical output head and lexical loss.

Execution variant: `chunked_norm_mlp`, the GPU-only source/normalization/recomputation/MLP-buffer optimization. No optimizer-state, LM-gradient or activation CPU offload. No tokenizer truncation or replacement inputs.

One full warmup update, then complete checkpoint/corpus/RNG reload for one measured update. Allocation stack tracing disabled, peak counters enabled. This is a very short capacity smoke, not a long-run throughput or training migration.

Warmup/reload equality check: `True`. Scored continuation tokens: `7396`; bytes: `23365`.

## Transfer and runtime

All 1070 files (36,910,437,064 bytes) were downloaded directly from Modal to Tillicum and checksum-verified before GPU submission. Only signed metadata and small code passed through the local caller. Cloud originals remain authoritative.

An isolated Python 3.12 environment uses PyTorch 2.15.0+cu132, transformers 5.15.1, tokenizers 0.22.2 and the same model code. The existing Rust helper passed its import preflight. Source hashes, actual GPU identity/capacity, package versions, all measured phases and error details are in result.json.

The single debug job requested one full H200 with a five-minute wall limit and no explicit CPU/system-memory reservation. Slurm elapsed time was 77 seconds, including startup and checkpoint loading. The scheduler assigned its defaults of 4 CPUs and 120G RAM. Preparation and transfer occurred before requesting GPU time. Existing Modal training was not modified.

Cross-device numerical parity against the local Blackwell GPU is not asserted; the smoke checks capacity and same-device warmup/reload fidelity. Prior local losslessness comparisons remain in [the implementation report](../chunked-memory/README.md).
