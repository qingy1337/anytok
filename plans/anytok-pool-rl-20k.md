# Adversarial selection from the existing tokenizer pool

Run `python scripts/run_anytok_pool_rl_20k.py --run` under tmux. Without `--run`,
the controller prints the plan. User authorized a 20k-step RL run on the existing
1,000 tokenizers. No tokenizer training is involved.

The model starts from the exact original LM and 10k identity-warmed hypernetwork
inputs used by the completed uniform-pool run, not its final20k checkpoint.
Native saved tokenizers and their hashes are preserved. The model optimizer,
batch128, microbatch32, seed42, 128 total positions, canonical continuation/output
head, lexical weight0.5 and original190k learning-rate schedule remain the same.
The runner checks these settings against the uniform run configuration. Its two
periodic sanity tokenizers are copied from the uniform run for paired evaluation.

## Policy and reward

One categorical action per model update selects a tokenizer for all128 examples.
The policy has1,000 logits initialized to zero. CPU Adam uses lr0.01, no weight
decay. Its independent NumPy RNG uses seed43; it cannot consume the model's
dropout RNG or corpus RNG. The model and hypernetwork minimize their existing
objective. The policy maximizes pre-model-update continuation BPB:

`reward = sum(continuation negative log likelihood) / scored UTF-8 bytes / ln(2)`

REINFORCE uses reward minus the previous EMA baseline (decay0.99) and entropy
bonus0.01 in BPB units. On the first update, initialize the baseline to the reward
and use zero advantage. No reward clipping, learned reward model, tokenizer
regeneration, evaluation reward, or restrictions by algorithm/domain are added.
The lexical auxiliary loss is excluded from the policy reward.

Each event records the selected tokenizer/probability, raw reward, advantage,
baseline, entropy/effective pool size, maximum probability, exposure count,
scored tokens/bytes, model loss and timing. `policy.pt` records logits, settings
and exposure counts with every model save. Full restart `state.pt` additionally
contains the policy Adam/RNG/baseline state, model RNGs, corpus cursor and queue.

## Persistence, budget and recovery

Input source: gildmember / anytok-pool-20k / input, with frozen sanity specs and
reference configuration from run/. Destination in each enabled profile:
**anytok-pool-rl-20k**, input/ and run/. This does not modify the uniform run.
Only metadata passes through the local caller; corpora, tokenizers and weights
copy directly between cloud services with per-file hashes and sizes.

GPU B300, no explicit CPU or memory reservations. Profiles are listed in
scripts/anytok_pool_rl_config.py; disabled general and research-everettcc never
participate. Check credits before preparation and training, allow a$2 forecast
margin, estimate$8.25/hour including overhead and cap each segment at3 hours.
Billing can lag; these are conservative estimates, not a hard billing cap.
The controller stops with a valid checkpoint if all usable credits are exhausted.

Weights every500 steps, full restarts every2k and at segment boundaries. A full
initial step0 checkpoint permits safe recovery before the first2k milestone.
Obsolete non-milestone restart state is removed only after a newer full save is
committed or a destination verifies the handoff. Interrupted future grid saves
are retained under interrupted/ rather than silently reused after replay.
Before full training, the GPU smoke executes2 updates and resumes for a third,
checking policy/model counters and serialized restart integrity. The smoke
directory is separate from the experiment. A pointer records every remote call;
observation timeouts wait on that call and do not restart training.

At completion, gather the40 grid checkpoints into the final profile's Volume
and audit actual file hashes. Existing private HF storage quota prevents using
HF as the reliable handoff store; use the already established direct Modal
Volume path without changing repository visibility or downloading locally.

## Comparison

Use the existing fixed pool/held-out skewed panels against the uniform run's
matching checkpoints. Higher policy reward is not success: success is lower
fixed-panel loss/transfer gap. Keep document IDs, continuation bytes/token counts
and context settings matched in each comparison. The policy's training loss is
adaptive and cannot be compared as an ordinary learning curve to uniform loss.
The128-position training split can choose different raw continuation spans for
different tokenizers; rawBPB may reward data/length effects as well as embedding
weakness. Record scored-byte/token counts; do not claim selection isolates
embedding quality. This is a single-seed curriculum experiment.

Progress: modal_results/anytok_pool_rl_20k/status.json, events.jsonl and per-profile
logs. Dashboard entry anytok-pool-rl-training, read-only web UI on port1337.
Unit checks: `python -m unittest discover -s tests -p test_pool_rl_bandit.py -v`.
Training source: runtime/anytok_pool_rl_20k, reusing the pinned original helper
modules and Rust library from runtime/anytok_pool_20k. Original runtime remains
unchanged. Run provenance records Git revision and input/helper/source hashes.
