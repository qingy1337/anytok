# Entropy-zero ablation of the frozen-model policy search

The user explicitly approved removing the entropy bonus after reviewing the
original 0.01 result. Change only that coefficient, to zero. Start fresh from
the same uniform policy initialization and seed 43; keep Adam lr 0.001,
EMA baseline decay 0.99, the original shared 18-feature Tanh MLP and 20,000
policy updates. Do not resume the already-trained 0.01 policy.

Use the byte-identical complete reward table in
`reports/frozen-policy-search-20k/reward_table.json.gz`: all 1,000 pool
tokenizers and all 288 original reward + 288 original confirmation passages,
corrected cuts, fixed trained RL20k LM and hypernetwork. The JSON plan pins
the parent plan hash and entire loss-table hash. Only the selected action's
exact full-reward-text BPB enters a policy update. Confirmation stays
evaluator-only. No new model inference or GPU allocation is needed.

Run locally:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/zli/.venv/bin/python scripts/frozen_policy_search.py --cache reports/frozen-policy-search-20k/reward_table.json.gz --run-plan plans/frozen-policy-entropy0-v1.json --persist
```

The baseline folder is preserved. New report: `reports/frozen-policy-entropy0-20k/`.
New status/logs: `modal_results/frozen_policy_entropy0/`; dashboard run:
`frozen-policy-entropy0`. Live plots compare both coefficients directly.
Persist checkpoints every 2,000 updates and the complete report separately
under aurora / anytok-frozen-policy-search / frozen-policy-entropy0-v1/.
Omit `--persist` to reproduce entirely offline from the same table and plan.

This is one matched policy seed under stationary exact rewards. It tests
the coefficient's effect in that setting, without establishing a cause for
the earlier noisy joint-training experiment. Report the actual outcome even
if entropy zero concentrates on a suboptimal tokenizer. Do not change the
seed, learning rate or run length after seeing results.

## Completed 2026-11-13 UTC

All 20,000 local updates completed in approximately 79 seconds, including
checkpoint/report persistence and plotting. Verified byte-identical complete
reward-table contents and bit-identical initial network weights/probabilities.
The only differing policy setting is entropy_weight: 0.01 to 0.

Final action #0106 has probability 0.9999999522: reward-text rank 2 and
held-out rank 8. Hardest-ten mass is 0.9999999579 on reward text, compared
with 0.0887552723 for the original coefficient. Mean held-out transfer gap
increases from +0.10892 to +0.13229 BPB: the adversary selects harder
tokenizers while the LM/hypernetwork stay frozen. It tried 225 distinct
actions, versus 917 with entropy 0.01, and settles on the runner-up rather
than exact worst #0635. The latter was sampled first at update84.

This coefficient-only test demonstrates much stronger exploitation in the
frozen, exact-reward setting. It does not establish the effect on model
generalization in the earlier noisy, co-adapting joint-training run.
The original report and cloud checkpoints remain unchanged.
