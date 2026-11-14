# Frozen RL20k paired-gap reward, entropy 0.01

Only the policy reward changes: transferred BPB minus trained canonical BPB.
Fresh 20,000 policy updates, seed 43, same uniform initialization and shared
settings network, Adam learning rate .001, EMA .99, entropy 0.01.
Matched baseline: reports/frozen-policy-search-20k.

Preserve all 1,000 pool tokenizers and all original 288 reward + 288 check
passages, corrected boundaries, BOS + prefix + 32 scored canonical targets,
128-position total cap, byte-weighting within each domain and equal weighting
across all nine domains. Confirmation remains evaluator-only. Reuse exact
complete frozen losses without changing or excluding any data. No model updates.

The canonical reward-set BPB is action-independent and constant. The EMA
baseline should cancel this offset, leaving the same policy advantages up to
floating-point rounding. Execute the full rerun and record actual sampled
rewards, action matches, final probabilities and complete learning curves.
Do not infer outcomes from that expectation.

Run locally; persist policy checkpoints every 2,000 steps plus the final report
to aurora / anytok-frozen-policy-search / frozen-policy-regret-v1/. No GPU launch needed.
All outputs use separate paths and are visible in plans/run-dashboard.json.

## Completed

Both the actual selected reward and all 20,000 actions are recorded.
Raw-versus-regret action matches: 7,581/20,000.
Held-out final expected gap: +0.10861783 BPB.
Complete results and plots: `reports/frozen-policy-regret-20k`.
