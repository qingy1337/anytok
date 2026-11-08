# Faster lossless tokenizer generation

Goal: at least 5–10× faster generation, with exact Unicode encode/decode round trips. Retain BPE, WordPiece and Unigram, all three pretokenizers, existing vocabulary/noise ranges and unconstrained mixture sampling. Do not achieve the target by reporting only the fast regex subset or silently reducing training data.

Primary benchmark: the nine algorithm/pretokenizer cells in tokenizer-speed-v1.json. Each uses a saved representative configuration and exactly the same 200 MB source prefix quotas. Both variants execute sequentially on the same Modal worker with four Rayon threads, alternating order by case. Source text hashes, settings and audit fixtures must match. Report individual cells, algorithm totals and total matched workload time; historical timings are context, not the paired speedup denominator.

The implementation-only candidate includes:

- BPE compaction in one pass instead of repeated vector insert/remove operations.
- One changed-pair presence scan per affected segment instead of one scan per change event in the custom all-candidate noise trainer.
- Unigram forward/backward reductions once per lattice position instead of repeating identical sums for every incoming/outgoing node pair. Reduction order is retained; an independent reference checks bit-for-bit marginal results.

Patch order against tokenizers upstream commit f383101a26663708484cac0727792aad74f78234: all-candidate-noise, unigram-literal-unk, unigram-stack-drop, fast-training. The optional disable-top patch is not part of this experiment. Accelerated wheel is built in build/fast-v2, baseline in build/pool-v2. Do not replace the global tokenizer installation or the immutable runtime/anytok_pool_20k source.

Cloud: profile gildmember, input Volume anytok-pool-20k mounted read-only; output Volume anytok-fast-tokenizers. Every run is named by a digest of plan, wheels and source hashes. Save tokenizers, full logs, provenance and results remotely. Only metadata returns locally. No GPU, CPU reservation or memory reservation is requested. Nine workers maximum. Check billing before runs; maintain a margin and inspect live calls rather than retrying after an observation timeout.

Commands:

```
MODAL_PROFILE=gildmember modal run scripts/modal_tokenizer_speed.py
MODAL_PROFILE=gildmember modal run scripts/modal_tokenizer_speed.py --action smoke
MODAL_PROFILE=gildmember modal run --detach scripts/modal_tokenizer_speed.py --action benchmark
```

The default command prints the plan without compute. Smoke tests use 1 MB and 1,024 vocabulary entries in all nine cells and are not speedup evidence. Benchmark uses the original data and vocabularies. Local call pointers under modal_results/tokenizer_speed permit reattachment to the exact coordinator. The dashboard tracks generation phases in plans/run-dashboard.json.

Lossless means exact reconstruction of valid Unicode with add_special_tokens=False and decode(skip_special_tokens=False). Audit all 256 byte alphabet entries, absence of normalization/lossy postprocessing, randomized Unicode across all planes, control characters, combining marks, literal special-token strings and persistent held-out text. Default decoding with skip_special_tokens=True intentionally deletes special strings and is outside this contract. Finite tests are not a universal proof; also inspect model coverage and runtime limits, especially WordPiece's long-input limit.

A smaller-corpus candidate, if used, must be labeled separately and checked for requested vocabulary size, losslessness, compression and transfer quality. It must not be called an identical-output implementation optimization.
