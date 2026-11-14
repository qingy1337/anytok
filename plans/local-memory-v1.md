# Local memory diagnostics alongside unchanged Modal training

User explicitly requested local memory experiments. The local RTX PRO 6000
has approximately 96 GB GPU RAM. No Tillicum job, Modal experiment or live
training-code change is authorized by this diagnostic study.

The user explicitly approved downloading an exact local diagnostic copy of
weights, optimizer state, original inputs and the full tokenizer pool, retaining
all cloud originals. Files are hash-verified and isolated under
modal_results/local_memory/artifacts/. No production artifacts are modified.

Use the exact model architecture, real checkpoint tensors and optimizer state,
original corpus, and complete 1,000-tokenizer pool after an approved copy.
Use production helpers for source embeddings, surface forms, prefixes, labels,
byte counts, batch 128, microbatch 32, the 128-position cap, canonical control,
lexical loss weight .5, backpropagation, clipping and optimizer settings.
Local PyTorch will match Modal's 2.15.0 in a separate environment.

Measure GPU allocated/reserved peaks separately for canonical scoring,
hypernetwork forward, LM microbatch forward/backward, lexical loss,
hypernetwork backward, and optimizer update. Use independent processes for
trials; explicitly report out-of-memory failures rather than dropping them.

Candidate variants are opt-in patches used only by the diagnostic process:
- Current training implementation, unchanged.
- Bounded CPU offloading of saved activations, keeping parameters resident.
- Finer activation checkpoint boundaries with preserved dropout RNG.
- Both finer checkpoints and bounded offloading.

Run the complete 128k-vocabulary workload; the baseline may OOM on this GPU.
Also compare complete 43k-tokenizer steps, which fit the unchanged baseline,
for loss, outputs, all parameter gradients and RNG state. Repeat the baseline
to measure its own numerical reproducibility. These are memory diagnostics,
not quality evaluation, and no small-vocabulary result will stand in for a
128k memory-fit result. Preserve full-vocabulary attention in every variant.
Record host RAM, runtime and numerical differences; do not claim bitwise
identity unless verified. Any production use requires the user's review.

## Explicit storage exception approved

The user approved downloading the exact local diagnostic copy, keeping cloud
originals. The matrix has nine trials: baseline/repeat/offload/fine/fine+offload
on the complete 43k tokenizer #652, then all four variants on complete 128k
tokenizer #989. #989 produced the highest observed Modal peak, 200.55 GiB.

Two additional trials isolate normalization from attention and MLP
recomputation, still preserving the original operators and full vocabulary.
The complete matrix now has eleven trials. CPU offloading preserves original
strides and respects a 64 GiB live saved-activation budget; it is never an
unbounded host-memory allocation. All seven semantic unit/CUDA checks pass.

The original eleven trials completed. Combined recomputation/offload reduced
43k peak from 86.01 to 59.30 GiB, but all 128k cases OOM on the local GPU.
Four follow-ups test the documented expandable_segments allocator setting,
including complete 43k numerical checks for both candidate implementations.
These retain all original trial results and all experimental model settings.

All four allocator follow-ups completed; 128k still OOM. Two additional
trials test normalization-split checkpointing with expandable segments and a
bounded NVMe-backed activation cache. They preserve the exact operators and
restore tensor values, shapes, strides and offsets. Saved host activations
remain bounded to 64 GiB (48 GiB steady + 16 GiB transient reserve); disk
spill is capped at 144 GiB with 16 GiB free-disk reserve. Temporary activation
files are deleted when autograd releases them. Numerical/CUDA tests cover
strided views, gradients and temporary-file lifecycle. No production change.

The original disk-spill trials completed; 128k still failed. A bug in the
diagnostic-only normalization wrappers retained whole GPU tensors in Python
closures merely to read their dtype. Replace those references with scalar
dtype values. Retain all previous results and run four distinctly named
fixed-wrapper trials, with full 43k numerical checks and full 128k workloads.
This bug was introduced in the diagnostic candidate, not the live trainer.

The fixed 128k NVMe trial completed forward and all four LM backward passes,
but OOMed in HN backward after saturating its 144 GiB disk-spill cap. Two final
cache-budget tests use 80 GiB host saved-activation allowance (64 steady plus
16 transient), within the 123 GiB local machine, with identical batch/model
computation. Record this operational budget separately in input metadata.

The 80 GiB cache did not complete 128k backward, despite no budget skips;
the remaining failure requested 5.37 GiB near the local device limit. Three
additional trials temporarily put the real Adam tensors on CPU during
forward/backward, restoring them on CUDA before the unchanged fused Adam
update. One isolates this on the original 43k implementation; two combine
it with the fixed activation strategy at 43k and 128k. Unit tests verify
exact fused Adam parameter/state updates after the round-trip.

The full 128k step completed at 87.24 GiB after inactive optimizer-state
staging was combined with the fixed memory strategy. Add a dedicated
forward-equivalence check on all 128,000 × 2,048 generated embedding values:
original versus optimized operators, same trained checkpoint and dropout RNG.
Both equivalence passes disable autograd, so this check is separate from the
completed full-training memory trial and does not establish 128k gradient parity.

Completed all 26 full-workload trials plus the full-table forward check. All
262,144,000 output elements and dropout RNG match exactly at 128k under the
matching PyTorch 2.15.0 runtime. The final full-step peak is 87.24 GiB, with
139.13 seconds in measured training phases. CPU saved activations peaked at
66.92 GiB (plus 13.82 GiB Adam state), and NVMe spill at 134.17 GiB. This is a
memory-fit result with substantial I/O cost, not a production throughput win.
The report preserves every failure and superseded diagnostic source version.
The running Modal trainer remains unchanged; no Tillicum job was submitted.
