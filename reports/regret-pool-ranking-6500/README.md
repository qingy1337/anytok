# Frozen regret policy versus complete pool hardness

For checkpoint comparisons on held-out text, use the [4,992-token full-pool eval and H200 launch instructions](../heldout-pool-sweep/README.md). This report documents the earlier **32-batch training-reward diagnostic**.

Frozen checkpoint **6,500**, all **1,000** native pool tokenizers, **32 × 128** training-style examples per tokenizer. Training continued independently.

The saved policy favorite is **#771**, with **90.17%** probability and estimated hardness rank **323/1,000**. Its mean regret is **+0.073411 BPB**.

Uniform expected regret: **+0.066152 BPB**. Policy expected regret: **+0.073636 BPB**. Estimated maximum: **+0.093447 BPB**, tokenizer **#941**. Estimated oracle-minus-policy headroom: **0.019811 BPB**.

Policy probability on estimated hardest 10 / 50: **0.00% / 0.00%**.

Discovery winner **#941** minus policy favorite on the independent 16 confirmation batches: **+0.018600 BPB**, paired-batch bootstrap 95% interval **[+0.015373, +0.021691]**. Discovery winner has higher regret on confirmation batches.

## What was scored

The original corpus/Python state was replayed from full step 6,000 to 6,500 and all 500 actual paired-batch hashes matched. The next 32 recipe-sampled batches were frozen, covering all 23 corpus buckets. All actions used the same primary text draws and ordered replacement banks. Original rejected-example behavior and limits were preserved.

Each action used its native prefix, a variable canonical continuation, matched canonical control, batch 128, microbatch 32 and the original 128-total-position cap. Canonical targets/bytes match each transferred pair; cuts can differ between actions exactly as in training. Canonical controls were recomputed for each action. Hypernetwork dropout stayed active with fixed independent per-batch seeds; a new embedding table was generated for every batch. No model or policy updates occurred.

Regret is byte-weighted within each update, then averaged equally across updates, matching the policy objective. Lexical loss is excluded from the policy reward. No fixed-32-token or deterministic-dropout substitute was used.

## Interpretation limits

These are finite estimates of stochastic training-distribution rewards, not exact population ranks or a held-out generalization evaluation. The all-batch maximum is selected from 1,000 noisy estimates. The discovery winner is therefore also compared on independent confirmation batches. Bootstrap intervals describe variation across these fixed training-reward batches; sequential corpus windows can be correlated. Close ranks should not be treated as decisive.

![Policy versus measured hardness](regret_pool_ranking_6500.png)

[PDF](regret_pool_ranking_6500.pdf) · [Every tokenizer](tokenizer-ranking.csv) · [Analysis](analysis.json)

| Rank | Tokenizer | Algorithm / pretokenizer | Vocab | Policy probability | Mean regret |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1 | #941 | unigram / no-regex | 125,000 | 1.019e-07 | +0.093447 |
| 2 | #779 | unigram / no-regex | 118,000 | 2.5873e-09 | +0.091427 |
| 3 | #260 | unigram / no-regex | 121,000 | 2.8446e-09 | +0.091390 |
| 4 | #170 | unigram / no-regex | 125,000 | 8.292e-08 | +0.091166 |
| 5 | #409 | unigram / no-regex | 105,000 | 2.6186e-09 | +0.091060 |
| 6 | #757 | unigram / no-regex | 117,000 | 3.067e-09 | +0.091011 |
| 7 | #902 | unigram / no-regex | 124,000 | 5.7958e-09 | +0.091003 |
| 8 | #117 | unigram / no-regex | 115,000 | 2.1336e-09 | +0.090700 |
| 9 | #435 | unigram / no-regex | 113,000 | 3.1543e-09 | +0.090572 |
| 10 | #65 | unigram / no-regex | 115,000 | 1.3352e-08 | +0.090178 |
