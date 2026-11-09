# Faster lossless tokenizer generation

Target: at least 5× aggregate speedup, preferably 10×, on the complete nine-cell workload. The user requires **the same 200 MB training text**, with implementation changes only. Keep all BPE, WordPiece and Unigram settings, the three pretokenizers, vocabulary/noise ranges, mixture weights and all Unigram iterations.

## Current benchmark

Candidate v8 is running as `ed3e9f11892aca02`, Modal app `ap-merQ8USN2jSIWD154iPiZn`, coordinator `fc-01M6YR4P7HWJDCZC60C8DZQM6M`. All nine smoke cases passed (18 phases), as did 202 native tests and 20 Python checks. All nine optimized full-size tokenizers have finished. Full-size speedup remains unqualified until the paired controls finish.

Inputs: profile `gildmember`, Volume `anytok-pool-20k/input`, mounted read-only. Outputs: `gildmember/anytok-fast-tokenizers/ed3e9f11892aca02`. Tokenizers, process logs, completion metadata and wheels persist there. Only metadata returns locally. There are nine CPU workers; no explicit CPU or memory reservations, and no GPUs. Rayon uses four threads for both variants.

Each cell uses a saved representative configuration in `tokenizer-speed-v8.json`. The accelerated trainer runs first, followed by the original trainer on the same worker, assigning the cold-cache penalty to the candidate. Compare source-document hashes and boundaries, per-source byte quotas, generator settings, output vocabulary size and audit fixture hashes. All nine optimized models' source hashes also match the original pool's saved manifests. The report checks this original reference as well as the paired variant, preventing both variants from silently using different text. Require matching nonempty container IDs. After provider preemption, preserve completed artifacts and replay the unmatched timing on the current worker.

Primary timing includes training, serialization and existing validation. Extended Unicode/held-out audits are reported separately. Report all nine cells, algorithm totals, CPU time and peak RSS. The aggregate is the ratio of summed times for nine equally weighted cells; it is not a minimum speedup guarantee for every tokenizer. Historical pool timings and unfinished controls are not the denominator.

## Implementation

- Compact BPE/WordPiece merges in one pass instead of repeated vector shifts. Compute changed-pair presence once per affected segment.
- Index actual pair occurrences in noisy-BPE segments of at least 2,048 symbols. Retain linear compaction for shorter segments and WordPiece, where the index regressed in the paired measurements.
- Bound WordPiece prefix searches by the longest vocabulary entry.
- Use vocabulary IDs directly from the Unigram trie. Cache initial substring matches and map surviving vocabulary entries to current IDs each pass.
- Compute all EM probabilities and expectations using compact arrays, retaining floating-point reduction order and the dynamic unknown fallback.
- Parallelize independent candidate alternatives. Keep the original A* fallback for tied alternatives. Use compact predecessor records for corpus Viterbi paths, preserving arrival order and floating-point rounding ties.

Native reference oracles compare merge events, final symbols, token IDs/offsets, exact EM values and pruning paths, including Unicode, removed pieces, overlaps, equal scores, signed zeros and large-magnitude rounding cases. No approximate objective, reduced vocabulary, smaller corpus or fewer iterations are used.

## Losslessness

The contract is exact reconstruction of valid Unicode using `encode(add_special_tokens=False)` followed by `decode(skip_special_tokens=False)`. Default decoding that skips special tokens intentionally deletes literal special strings and is outside this contract.

Audit all 256 ByteLevel symbols, no normalization/postprocessing/truncation/padding, no whitespace-stripping special tokens, finite Unigram scores and exact round trips on 472 fixed randomized/held-out fixtures per model. WordPiece's runtime UNK length cutoff is raised to `sys.maxsize` in both variants, without changing its learned vocabulary. Finite fixtures alone are not a universal proof; byte coverage and runtime settings support the losslessness argument.

The construction preserves text because ByteLevel maps each UTF-8 byte to one of 256 symbols, and every model retains all 256 single-symbol tokens. BPE therefore starts with a complete representation. WordPiece always has a matching single-symbol prefix at every remaining position, with no practical runtime length cutoff. Unigram always has a complete finite-score single-symbol path. The pretokenizers preserve all text, the runtime adds/drops no text, and the byte decoder inverts the mapping. Keeping special tokens during decoding also preserves literal special-token strings.

After generation, the nine saved optimized models were reloaded in the idle coordinator container. Every actual file hash matched its completion record, actual vocabulary sizes matched the plan, and all nine passed another 264 Unicode fixtures each, with full byte coverage and finite scores. The persistent receipt is `ed3e9f11892aca02/posthoc-fast-verification.json`. This separate audit did not run in a timed training worker.

Modal preempted the no-regex Unigram control at 2026-11-09 00:35:06 UTC and automatically restarted the same input in `ta-01M6YVNV5B45XX39AG9M8N4VJR`. Its saved fast artifact survived. The benchmark will repeat that fast timing on the replacement worker after its control finishes. Cross-container timing pairs are excluded from reports and plots while this replay is pending; 18 completion markers alone do not establish a matched grid.

The observer reloads **all 18 current artifacts** after timing finishes using `modal_verify_tokenizer_speed.py`. It checks their actual file hashes, vocabulary sizes, wheel/worker provenance and all 472 fixtures each, then verifies that the audit refers to the exact artifact paths used in the final timings. This covers replayed artifacts rather than relying on the earlier nine-model receipt. A cloud smoke audit of the current nine optimized artifacts passed; final receipts persist under `verification/current-models-all.json` and in the archived report. The Unigram cache increases peak RSS (line256: 6.54 → 17.86 GiB); per-case memory is reported alongside speed.

## Reproduction

Pinned upstream: `f383101a26663708484cac0727792aad74f78234` (tokenizers 0.22.2). Apply the three original pool patches (all-candidate-noise, unigram-literal-unk, unigram-stack-drop), then **one** full optimization patch `fast-training-v8`. Do not stack optimization versions or apply the unrelated disable-top-noise patch. The builder checks source and patch hashes and leaves global installations unchanged.

```sh
python scripts/build_fast_tokenizers.py --candidate v8 --test
ANYTOK_SPEED_CANDIDATE=v8 MODAL_PROFILE=gildmember modal run scripts/modal_tokenizer_speed.py
ANYTOK_SPEED_CANDIDATE=v8 MODAL_PROFILE=gildmember modal run --detach scripts/modal_tokenizer_speed.py --action smoke
ANYTOK_SPEED_CANDIDATE=v8 MODAL_PROFILE=gildmember modal run --detach scripts/modal_tokenizer_speed.py --action benchmark
```

The default action prints the plan. Smoke uses 1 MB and vocabulary 1,024 solely to check integration; it is never performance evidence. Check credits before launching. To observe the existing benchmark after launcher edits, supply `--action benchmark --resume-from modal_results/tokenizer_speed_v8/ed3e9f11892aca02-benchmark-call.json` with the matching candidate environment. This reattaches to the exact coordinator; do not launch duplicates after an observation timeout.

```sh
MODAL_PROFILE=gildmember python scripts/collect_tokenizer_speed.py --candidate v8 --run-id ed3e9f11892aca02
python scripts/analyze_tokenizer_speed.py --candidate v8 --run-id ed3e9f11892aca02 --partial
python plotting_scripts/plot_tokenizer_speed.py --analysis modal_results/tokenizer_speed_v8/ed3e9f11892aca02-analysis.json
```

Omit `--partial` only after the consolidated `-results.json` exists. Partial grids cannot qualify. Monitoring uses `plans/run-dashboard.json`, timestamped JSONL events and the existing web dashboard on port 1337. Generation can also use the isolated wheel with `scripts/tokenizer_speed.py --config ... --recipe ... --destination ... --heldout-root ...` inside the persistent cloud worker.

The read-only observer is running in tmux `anytok-speed-observer`. It polls the existing coordinator every minute, refreshes analysis/plots, and archives final results, CSV, findings and PNG/PDF with a verified checksum manifest when the call completes. It launches no compute or replacement jobs:

```sh
MODAL_PROFILE=gildmember python scripts/observe_tokenizer_speed.py --pointer modal_results/tokenizer_speed_v8/ed3e9f11892aca02-benchmark-call.json --watch
```

## Generate new configurations

Use the same audited worker outside the benchmark. This entry point preserves the 200 MB quotas, fingerprints the persistent input recipe/corpus manifest, saves model files and logs in the output Volume, and supports up to 100 containers without explicit CPU/memory reservations. The default action prints the selected plan; generation is explicit:

```sh
MODAL_PROFILE=gildmember modal run scripts/modal_generate_fast_tokenizers.py --indices 86
MODAL_PROFILE=gildmember modal run --detach scripts/modal_generate_fast_tokenizers.py --indices 86 --action generate
```

`--plan-path` accepts another saved generator plan; `--indices` selects configurations within it. Completed outputs are cached by runtime and complete configuration hashes, separately from the old pool. Existing call pointers reattach to the same coordinator after observation timeouts. The `generate_one` worker is also callable directly from Modal orchestration.

The production entry point was verified with full-size configuration 86: identical source hashes and serialized tokenizer to its benchmark counterpart, 472 passing fixtures, and the hardened WordPiece runtime limit. Its persistent artifact is `generated/ffac5c30d0686d89/bfaa4213a1123fe0ef95ecc0ef001e9b9ff7a53e305ad526984a9bfa5a689b8a/tokenizer.json`. Repeating the command returned the same call/result without launching another training task.

## Archived tuning runs

V4 and v5 stopped deliberately after 11/18 and 12/18 phases; their implementations were superseded and two workers suffered provider preemptions. V6 and v7 stopped deliberately after 14/18 phases each once v8 smoke passed, avoiding duplicate expensive controls. All completed metadata/models persist. Their incomplete grids cannot qualify the target. V6's broad occurrence index caused regressions and is not the selected implementation. V7 retained expensive pruning lattices, replaced in v8. Never restart these archived apps.
