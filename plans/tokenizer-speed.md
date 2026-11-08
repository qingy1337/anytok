# Faster lossless tokenizer generation

Goal: at least 5–10× faster generation, with exact Unicode encode/decode round trips. Retain BPE, WordPiece and Unigram, all three pretokenizers, existing vocabulary/noise ranges and unconstrained mixture sampling. Do not achieve the target by reporting only the fast regex subset or silently reducing training data.

The user explicitly requires keeping the same 200 MB training data and optimizing implementation only. No reduced-data candidate is authorized. One-megabyte smoke tests are solely integration checks, not performance evidence or a replacement generation recipe.

Primary benchmark: the nine algorithm/pretokenizer cells in tokenizer-speed-v1.json. Each uses a saved representative configuration and exactly the same 200 MB source prefix quotas. Both variants execute sequentially on the same Modal worker with four Rayon threads, alternating order by case. Source text hashes, settings and audit fixtures must match. Report individual cells, algorithm totals and total matched workload time; historical timings are context, not the paired speedup denominator.

The implementation-only candidate includes:

- BPE compaction in one pass instead of repeated vector insert/remove operations.
- One changed-pair presence scan per affected segment instead of one scan per change event in the custom all-candidate noise trainer.
- Unigram forward/backward reductions once per lattice position instead of repeating identical sums for every incoming/outgoing node pair. Reduction order is retained; an independent reference checks bit-for-bit marginal results.
- Unigram pruning computes each candidate's alternative segmentation independently in parallel, preserving candidate order. The EM loop avoids Viterbi counts used only by disabled debug logging.
- WordPiece searches only up to the longest possible vocabulary entry, avoiding impossible whole-document prefix probes. A reference oracle checks token IDs and offsets, including multibyte text and continuation prefixes.

The subsequent v5 candidate additionally uses vocabulary IDs directly from the Unigram trie and compact edge/predecessor arrays for EM, preserving node insertion and floating-point reduction order. A bit-for-bit lattice oracle covers Unicode, empty inputs and unknown fallbacks. Unique candidate-pruning alternatives use flat dynamic programming; tied and near-tied cases retain the original A* path to preserve tie behavior. Independent candidate tests compare pruning results against the original n-best implementation.

V6 adds a training-only occurrence index for BPE and WordPiece segments of at least 128 symbols. Stable linked positions locate only actual occurrences of the chosen pair; adjacent-pair counts update incrementally. Short pretokens keep the existing compact representation. Sorted occurrence positions preserve left-to-right overlap handling, and merge change events and active symbols are checked against the pinned reference through nested/duplicate-surface merge sequences. This addresses the full-segment scans that remain after linear compaction; it does not change the allowed merges, their probabilities or training data. Select it with ANYTOK_SPEED_CANDIDATE=v6 and build with --candidate v6.

Before launching the full v6 run, its timing order is explicitly changed to accelerated-first for every cell, followed by its baseline on the same worker. This provides accelerated timings promptly while the slow controls finish. Any cold-source/cache penalty falls on the accelerated candidate. V4/v5 retain their original alternating order. Do not pool those protocols silently or call unfinished controls completed measurements; ongoing baseline runtime can provide a labeled speedup lower bound only.

Patch order against tokenizers upstream commit f383101a26663708484cac0727792aad74f78234: all-candidate-noise, unigram-literal-unk, unigram-stack-drop, fast-training. The optional disable-top patch is not part of this experiment. Accelerated wheel is built in build/fast-v4, baseline in build/pool-v2. Do not replace the global tokenizer installation or the immutable runtime/anytok_pool_20k source.

Cloud: profile gildmember, input Volume anytok-pool-20k mounted read-only; output Volume anytok-fast-tokenizers. Every run is named by a digest of plan, wheels and source hashes. Save tokenizers, full logs, provenance and results remotely. Only metadata returns locally. No GPU, CPU reservation or memory reservation is requested. Nine workers maximum. Check billing before runs; maintain a margin and inspect live calls rather than retrying after an observation timeout.

Commands:

```
MODAL_PROFILE=gildmember modal run scripts/modal_tokenizer_speed.py
MODAL_PROFILE=gildmember modal run scripts/modal_tokenizer_speed.py --action smoke
MODAL_PROFILE=gildmember modal run --detach scripts/modal_tokenizer_speed.py --action benchmark
```

Build v5 from pinned source, then plan/smoke/benchmark it separately:

```
python scripts/build_fast_tokenizers.py --candidate v5 --test
ANYTOK_SPEED_CANDIDATE=v5 MODAL_PROFILE=gildmember modal run scripts/modal_tokenizer_speed.py
ANYTOK_SPEED_CANDIDATE=v5 MODAL_PROFILE=gildmember modal run --detach scripts/modal_tokenizer_speed.py --action smoke
ANYTOK_SPEED_CANDIDATE=v5 MODAL_PROFILE=gildmember modal run --detach scripts/modal_tokenizer_speed.py --action benchmark
```

V5 uses the full replacement optimization patch fast-training-v5 after the three original pool patches, not after fast-training. Its plan explicitly records the unchanged-data requirement. Results are separate under modal_results/tokenizer_speed_v5 and the same persistent output Volume. V4 continues independently; never restart it just because a later source revision exists. To reattach an old launched coordinator after launcher edits, use `--action benchmark --resume-from PATH_TO_ITS_CALL_JSON` with the matching candidate environment. Resume is read-only and does not require a new credit margin.

The default command prints the plan without compute. Smoke tests use 1 MB and 1,024 vocabulary entries in all nine cells and are not speedup evidence. Benchmark uses the original data and vocabularies. Local call pointers under modal_results/tokenizer_speed permit reattachment to the exact coordinator. The dashboard tracks generation phases in plans/run-dashboard.json.

Lossless means exact reconstruction of valid Unicode with add_special_tokens=False and decode(skip_special_tokens=False). Audit all 256 byte alphabet entries, absence of normalization/lossy postprocessing, randomized Unicode across all planes, control characters, combining marks, literal special-token strings and persistent held-out text. Default decoding with skip_special_tokens=True intentionally deletes special strings and is outside this contract. The worker removes WordPiece's one-million-character UNK cutoff by setting its runtime limit to sys.maxsize; this applies to both benchmark variants and does not change the learned vocabulary. Finite tests are not a universal proof; also inspect byte coverage and runtime limits.

Completion requires measured full-size speedups and exact round-trip checks, not only Rust unit tests, a theoretical complexity improvement or the tiny smoke-test timings. Report ready-to-use generation time including the existing serialization/validation path, CPU time and extended audit time separately. Keep all nine cells visible, including regressions.
