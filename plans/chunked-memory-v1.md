# Lossless chunking diagnostics

The user authorized an isolated fast, lossless alternative to the original
full-vocabulary hypernetwork. Use the approved local exact checkpoint/input
copy, retaining cloud originals. Do not change the running Modal trainer.

Track full native 43k and 128k memory/step trials on the dashboard. Preserve
all failed candidates. Compare against original full43k output/gradient/RNG
references and repeat-baseline numerical noise; compare full128k forwards
separately because the original training step does not fit this local GPU.

Whole-entry chunking preserves the full vocabulary and global latent attention,
and uses packed native dropout masks. It completed real43k and128k steps, but
changed real outputs and gradients; it is rejected as lossless. Unit tests
retain this rejection explicitly instead of relaxing the acceptance criterion.

Follow-up: chunk only LayerNorm forward. Its native CUDA operation sees the
same FP32 values as before. Save the exact original BF16 input instead of an
expanded FP32 copy; reconstruct it in backward. Keep one original full-shape
native backward call, preserving affine-parameter reduction order. All other
operations and global vocabulary attention remain unchanged. Eight focused
CUDA tests pass, including exact input/affine gradients at native width2048
and43k rows, plus full-model output/gradient/RNG checks.

This candidate does not claim constant total memory: full layer states and
native backward temporaries still scale with vocabulary. Its purpose is to
find a fast computation-preserving reduction in the actual peak. Full streaming
is only acceptable if it meets the same correctness checks.

The native-width43k source/normalization candidate completed in4.84s at
61.08GiB, with exact outputs/losses/RNG and all349 parameter gradients within
repeat-baseline noise (relativeL2 1.33e-8). Full128k still OOMed in backward,
including a CPU-Adam variant that missed a10.74GiB allocation by0.51GiB.

Next split the native LayerNorm backward into its existing independent
subcomputations: full original affine-parameter reduction, followed by
row-chunked input gradients. This removes the full FP32 input-gradient
allocation while preserving the full weight/bias reduction. Twelve CUDA
checks pass, including exact2048-wide input/affine gradients.

Further exact native-operator trials tested finer MLP checkpoints, compact
projection inputs, direct generated-embedding gradient injection and bounded
CPU activation staging. Whole-entry chunking remains rejected; no tolerance
was relaxed. A6GiB activation cache still OOMs at128k, so test18GiB, with CPU
Adam staging and no NVMe. All43k candidates continue to match exact outputs,
losses and RNG with gradient differences at repeat-baseline noise.

The complete128k step finally completed at93.24GiB in20.25s, with selected
normalization inputs, inactive Adam state and completed LM gradients staged
on CPU, and the unused generated-output allocation released through a public
GradientEdge before backward. No NVMe. The matched43k run retains exact
outputs/losses/RNG and repeat-baseline gradient agreement.

Validate128k gradients against the previously verified original-operator
checkpoint/offload implementation, saving its complete output table and all
349 parameter gradients as a reference. This is not an unmodified single-GPU
baseline: the original unmodified128k training graph cannot fit locally.
Afterwards repeat the new full128k step with the same reference. Do not claim
128k gradient parity before this succeeds. No production changes or new
cloud jobs have been launched.

Validation complete:43 full-workload records retained, plus the separate128k
forward check. Current GPU-only43k:57.47GiB,4.92s, exact outputs/losses/RNG,
gradient relativeL2 1.32e-8. Validated128k:93.24GiB,20.51s, all262,144,000
embedding values/losses/RNG exact, all349 gradients relativeL2 8.21e-9 versus
the original-operator checkpoint/offload reference. The unmodified128k
no-grad forward also matches bit-for-bit.35 tests passed;4 expected failures
retain the rejected whole-entry chunking. CPU staging:13.82GiB Adam,
6.00GiB LM gradients,16.11GiB peak normalization inputs; no NVMe.

Total memory still scales with vocabulary. Fully bounded streaming did not
meet the original numerical-equivalence criterion; do not describe this as
constant-memory training. All work remains isolated from production. No
Tillicum allocation or new paid cloud job was made.
