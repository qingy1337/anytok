# Vocabulary scaling: original versus lower-memory methods

The user requested fitted speed curves through43k and an estimate of the
relative slowdown at128k. This is a local performance study, not a quality
evaluation or production change. Reuse the specifically authorized diagnostic
copy of the exact step2000 checkpoint, inputs and complete tokenizer pool.

Benchmark six complete native BPE/no-regex tokenizers at32k,34k,36k,37k,40k,
and43k. All methods use each native tokenizer without truncation, the same
checkpoint/corpus/RNG, full128-example batch, microbatch32 and128-position
cap. Preserve all losses and full native vocabulary attention.

Three methods: original high-memory implementation; the new GPU-only native
operator implementation; the exact low-memory128k recipe with selected CPU
staging and GradientEdge release. The third is needed to distinguish rewrite
cost from offloading cost. Three repetitions at each point, shuffled point
order and rotating method order (54 timed measurements). Add three128k
staged measurements held out from fitting. No new cloud jobs.

Each independent process completes one full warmup update, releases all its
models/tensors, then reloads the exact original checkpoint/RNG/corpus before
the timed update. Assert matching warmup/reload losses, RNG and input proof.
Disable allocation stack tracing consistently for timing; keep CUDA peak
counters. Check exact example identity, losses and RNG between all methods
at each native vocabulary. Loading/preparation/warmup remain outside timing.

Fit affine time versus vocabulary to32k–43k measurements only, separately
for total timed model/optimizer phases and hypernetwork work. Display measured
points, dashed extrapolations to128k and timing-noise resampling intervals.
Compare predicted original128k with measured staged128k; show the GPU-only
rewrite extrapolation separately. Native operations have linear-in-vocabulary
work at fixed source-piece length, width and latent count; this motivates a
linear fit but does not guarantee extrapolation accuracy. The observed range
is narrow and far below128k. Report leave-one-size-out sensitivity and label
all128k original/GPU-only predictions as extrapolations, not measurements or
an H200/B300 throughput forecast. Preserve failures instead of replacing them.

Completed57 timed updates and57 full-workload warmups. All checkpoint/reload
checks and matched-method input/loss/RNG checks passed. Fits use54 complete
32k–43k measurements; all three128k results are held out. Final original128k
projection9.67s, GPU-only rewrite11.37s, staged projection19.44s. Held-out
staged measurements19.23/19.21/19.31s, mean19.25s. Relative penalties:1.176x
GPU-only,1.990x measured staging over projected original. Bootstrap intervals
are conditional on an affine model, not a guarantee of extrapolation accuracy.

A setup failure on the first warmup/reload attempt exposed a PyO3 one-import
restriction. Cache the immutable trainer module while recreating all model,
optimizer, corpus and RNG state. Retain the setup failure log. No failed run
was counted as a timing point. Kernels are warmed; allocator cache is cleared
before exact checkpoint reload, so this is not a long-run steady-state claim.
