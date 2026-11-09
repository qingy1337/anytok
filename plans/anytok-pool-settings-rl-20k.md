# Shared generator-settings policy, 20k

User requested a shared network that understands tokenizer-generator settings.
It scores each of the1,000 existing tokenizers from its settings, then samples
from the softmax over those scores. No tokenizer is retrained. The ID-only
experiment was stopped cleanly at335; its complete model/policy restart remains
at trenton / anytok-pool-rl-20k / run/step-000335.

Start with `python scripts/run_anytok_pool_rl_20k.py --run` under tmux. Without
`--run`, it prints the plan. This experiment starts fresh from the same original
LM and identity-warmed hypernetwork as uniform training. It does not continue
the335-step ID-only model.

## Network inputs

Eighteen features, using fixed scales rather than fitted pool statistics:

- Algorithm: three one-hot features (BPE, WordPiece, Unigram).
- Pretokenizer: three one-hot features (regex, line256, no-regex).
- Vocabulary: `log2(vocab_size / 32000) / 2` (32k→0,128k→1).
- Merge noise: applicability flag plus `noise_per_million / 750000`; non-BPE
  entries get0 for both. No added restrictions on the existing pool settings.
- Nine domain fractions: web, wikipedia, code, papers, math, qa, books,
  medical, legal. Normalize the saved domain-byte quotas by their sum.

Tokenizer ID, seed, config hash, file size and measured loss are not input
features. Bucket/language ratios within each domain are fixed by the original
recipe and do not vary independently in this pool. All varying generator knobs
are represented. Identical settings receive identical scores in the same batch,
even when tokenizer IDs/seeds differ. No per-ID residual or embedding is used.

A CPU float64 MLP maps18→64→64→1 with Tanh hidden activations:5,441 shared
parameters independent of pool size. Initialize hidden weights with private
seed43 and zero the output layer, giving exactly uniform initial probabilities.
Preserve both the LM's CPU RNG state and CUDA/dropout seeds during construction.

CPU Adam lr0.001, EMA reward baseline decay0.99 and entropy bonus0.01. REINFORCE
maximizes pre-model-update continuation BPB; the LM/hypernetwork minimize the
same original loss. The first policy advantage is zero. There is no reward
clipping, model-state input or learned per-tokenizer parameter. The network can
score another configuration, but actual training samples only the fixed pool;
this is not yet evidence of generalization to unseen generator settings.

## Model, checkpoints and comparison

Keep the128-position cap,128-example updates, microbatch32, frozen canonical
input/output tables, original model optimizer/schedule, lexical loss weight0.5
and seed42. Check these against the saved uniform configuration. Periodic sanity
evaluation uses the uniform run's two fixed tokenizers. The fixed pool/OOD panels
remain the tests of whether this curriculum improves on uniform training.
Adaptive training BPB alone cannot establish that comparison.

The new source snapshot is runtime/anytok_pool_settings_rl_20k/. The old ID-only
snapshot remains unchanged. Both reuse original frozen helper modules and the
Rust library from runtime/anytok_pool_20k. Policy checkpoints contain network
weights, feature schema/digest, settings, exposure counts and update count;
full restarts also include its optimizer, baseline and RNG. Changed settings
features fail resume validation. Model-only policy weights are saved alongside
every500-step model; full restarts every2k, initial0 and segment boundaries.

Use the existing anytok-pool-rl-20k Volume and input/ data, with separate
**settings-run/** and **settings-smoke/** output directories in each profile.
Old run/ and smoke/ artifacts are preserved. State/logs are separate under
modal_results/anytok_pool_settings_rl_20k/. App name:
anytok-pool-settings-rl-20k-b300. Dashboard ID:anytok-pool-settings-rl-training.

B300 only, no explicit CPU/memory reservations. Enabled profiles, credit checks,
$2 forecast margin, $8.25/hour estimate, three-hour maximum segments, direct
hashed cloud-to-cloud copies and final40-checkpoint collection follow the prior
controller. Disabled general and research-everettcc never participate. The
controller records exact calls and reattaches after observation timeouts.

Before full training, run2 GPU updates and resume on another container for
update3. Local checks cover all feature knobs, absence of ID/seed features,
reward sharing across unsampled identical configurations, reward direction,
RNG isolation, fixed parameter count and exact serialized optimizer/RNG recovery.
