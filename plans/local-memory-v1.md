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
