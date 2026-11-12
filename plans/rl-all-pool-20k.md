# RL20k: evaluate the complete training-tokenizer pool

The user requested evaluation of the final RL-trained hypernetwork and LM on
all 1,000 existing pool tokenizers, and identification of the worst tokenizer.
There is no new model training or tokenizer generation.

Use H200 workers on `blackpantherx86` (estimated $16.20 remaining at planning).
Do not reserve CPU or memory. Store everything in its v2 Modal Volume
`anytok-rl-all-pool-20k`. Copy only necessary original model files, RL inference
weights, native tokenizer files and held-out corpora directly from gildmember;
the local caller receives metadata and prediction losses only.

## What is scored

Use the final 20k shared-settings RL checkpoint, with its trained LM, trained
input hypernetwork and original fixed canonical output head. Verify every
transferred file against the original checkpoint/input/tokenizer manifest.
Generate embeddings from **the complete native vocabulary**: inter-token
attention means generating only the prefix-used IDs would change the method.

Before seeing losses, choose 64 independent held-out documents from each of
the nine established domains by a deterministic ID hash. The first 32 per
domain form the 288-passage screen. The other 32 per domain are reserved for
confirmation. Score exactly 32 canonical continuation tokens after each cut.

Start with the usual maximum 64-token canonical prefix. Jointly shorten the
raw prefix until all 1,000 native tokenizers round-trip it without special or
unknown tokens and fit BOS + prefix + 32-token continuation within 128
positions. Iterate and revalidate the entire pool; do not assume suffix token
counts are monotonic. All tokenizers receive **identical raw prefix text,
document IDs, continuation IDs and scored UTF-8 bytes**. Prefix shortening
and rejection use encoding validity only, never prediction loss. This new
shared panel differs from the earlier tokenizer-specific panels, so its
absolute BPBs should not be compared directly with those earlier scores.

## Ranking and confirmation

Primary: compute BPB separately in each domain, then average the nine domain
BPBs with equal weights. The largest value is worst. Also report canonical
BPB and excess BPB, pooled-byte BPB, and individual-domain rankings. Because
all tokenizers share the canonical baseline, the primary BPB and gap rankings
agree. A tokenizer can be worst on one domain without being worst overall.

Screen all 1,000; confirm the top 10 overall plus the winner of every domain
on the separate confirmation passages. Report ranks and paired uncertainty
without claiming a unique worst tokenizer if close candidates cannot be
distinguished. This is a finite held-out text sample, not a proof of worst-case
performance on arbitrary text. Include actual training-selection counts and
settings in the resulting table.

## Execution and recovery

```bash
MODAL_PROFILE=blackpantherx86 modal run --detach scripts/modal_eval_rl_pool.py --action prepare
MODAL_PROFILE=blackpantherx86 modal run --detach scripts/modal_eval_rl_pool.py --action pilot
MODAL_PROFILE=blackpantherx86 modal run --detach scripts/modal_eval_rl_pool.py --action run
```

Run a three-tokenizer pilot before the complete sweep and use its measured
runtime to check the credit estimate. Results persist per tokenizer, so a
recovered launcher reuses verified completed scores. Record app URLs, logs,
progress and ETA in `plans/run-dashboard.json`. No dashboard job controls.

Local metadata: `modal_results/rl_pool_20k/`. Cloud raw-text contexts:
`panel.json`; per-tokenizer features: `features/NNNN/`; per-passage scores:
`screen/NNNN.json` and `confirm/NNNN.json`; combined rankings: `screen.json`
and `confirmation.json`. Scoring source is committed before launch.
