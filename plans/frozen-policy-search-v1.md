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

## Completed 2026-11-13 UTC

All 1,000 tokenizers were freshly scored on all 576 original passages.
The source preparation required two rounds of full-pool validation; no
passages or tokenizers were dropped. Cut changes: 128 screen and 118
confirmation. Additional joint prefix trims: four and five. All 330
already-valid original contexts remained unchanged. Their fresh losses
match the original sweep to within 0.000013 nats per passage.

Scoring app: `ap-YXtUzJ4TaAJm5Ix9LiDWP5`; continuation app:
`ap-YArps896qtRGdWipTtkHPK`, on aurora. Each initial bounded worker
reached its 2,400-second limit, saving 376 and 382 completed results.
Continuation calls read those exact results and completed the remaining
124 and 118; the plan, weights, text and tokenizers were unchanged. An
earlier malformed status read required reconnecting the observer; no
model inference was restarted for that observer recovery. All scoring
apps were stopped after collection; zero GPU containers remain.

The local policy completed 20,000 updates in approximately 78 seconds,
including plotting and cloud checkpoint/report uploads. Final probability
on the reward-text hardest ten: 8.88%, versus 1% uniformly; on the single
worst #0635: 0.877%, versus 0.1%. Its most probable choice is #0213,
reward-text rank 10 and held-out rank 23. It sampled #0635 first at update
503; the single uniform baseline first did so at 869, which does not
establish a general speed advantage.

Expected transfer gap under the learned policy versus exact full-pool
uniform: reward text +0.08323 versus +0.06690 BPB; held-out check text
+0.10892 versus +0.08883. #0635 is the highest measured BPB across the
complete pool on both sets: gaps +0.10396 and +0.13784. The policy favors
a hard group rather than putting most mass on the exact worst.

Reproducible report: `reports/frozen-policy-search-20k/`. The same folder
and ten resumable policy checkpoints are persisted in aurora /
anytok-frozen-policy-search / frozen-policy-search-v1/. Cached-file
reproduction is fully local and does not require Modal credentials.
