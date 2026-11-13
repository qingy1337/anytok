# Frozen-model adversarial search over all 1,000 tokenizers

The user authorized testing whether the existing shared-settings adversarial
policy can find high-BPB tokenizers when the trained LM and hypernetwork are
frozen. The user explicitly rejected reducing the text to the 160 passages
whose old cuts were already valid. No subset experiment was run.

Use the final RL20k LM and hypernetwork. Repair the original token-boundary
problem on **all 288 original screen passages and all 288 original confirmation
passages**. Keep every source document and split, with 32 documents in each of
nine domains per split. Keep already-valid contexts exactly unchanged. Repair
invalid cuts by the established nearest canonical-boundary rule, with a maximum
64-character shift. Validate shared prefixes against **all 1,000 native
tokenizers**, iterating common prefix trims until every sequence fits 128
positions. Score exactly 32 canonical continuation targets. No context-benefit
filtering, proxy weighting or loss-based exclusions. If any passage cannot be
validated, stop and report it; do not silently discard or replace it.

Compute a full per-passage loss table once using the verified inference scorer,
complete native vocabularies and checksum-verified RL checkpoint. Run the
upfront evaluation where artifacts reside in cloud storage, staging directly
between cloud volumes if another credit profile is needed. Local caller receives
only metadata and prediction losses. No CPU/memory reservations. B300 scoring:
two shards of 500, each bounded to 2,400 seconds; verify at least $11 remaining
before launch. Never use the disabled Modal profiles.

Train the policy locally for 20,000 updates with unchanged 18-feature shared
settings network, seed 43, Adam lr 0.001, EMA baseline decay 0.99 and entropy
bonus 0.01. Start exactly uniform. Each selected action receives its **exact
mean-domain BPB on the full 288-passage screen**, without reward noise from
changing text batches. This controlled test checks learning under a stationary
reward; it does not reproduce the minibatch noise of joint model training.
No LM or hypernetwork optimizer is instantiated. The selected reward is the
only measured loss given to the learner; complete rankings are evaluator-only.
Confirmation losses never enter a policy update or hyperparameter selection.

Caching preserves these fixed-model, fixed-text rewards exactly; it does not
approximate inference. Rank all 1,000 separately on screen and confirmation.
Compare policy-weighted BPB with uniform, the best fixed action and the exact
entropy-regularized optimum for the existing entropy bonus. Track mass on the
true top 1/10/50, discovery rank, unique coverage and pretokenizer probabilities.
The entropy bonus means optimal behavior need not put all mass on one action.
Do not change the reward, entropy weight or policy features after seeing results.

Both phases are visible in the dashboard: `frozen-policy-cache` and
`frozen-policy-search`. Local logs/status use matching underscore directories
under `modal_results/`. Policy snapshots and trace metadata persist to cloud;
corpora, tokenizer files and LM/hypernetwork checkpoints remain in cloud storage.
Material changes to this design require explicit user approval.
