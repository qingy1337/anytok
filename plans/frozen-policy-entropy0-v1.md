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
