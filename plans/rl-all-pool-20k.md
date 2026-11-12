# RL20k: evaluate the complete training-tokenizer pool

**GPU update, November12:** the user approved B300s after the H200 pilot
remained queued for over an hour. The H200 app and its waiting controller
were stopped. Resume the same prepared cloud inputs on B300, first measuring
a new three-tokenizer pilot. The controller uses the B300 rate of
[$0.001972/second](https://modal.com/pricing), checks verified remaining
credits before the full run, and records the actual CUDA device in each
result. No H200 scores were produced or mixed with B300 scores. Historical
H200 queue details below describe the superseded launch.

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

## Prepared and queued, November 12

All 1,018 required files were copied and verified cloud-to-cloud. All 1,000
native vocabularies were prepared and validated against the final shared
contexts. Exactly 35 of the 576 total prefixes required trimming. The screen
has 288 passages, 9,216 continuation tokens and 26,690 continuation bytes;
confirmation has 288 passages, 9,216 tokens and 26,285 bytes. Their source IDs
are disjoint. All sequences fit 128 positions, and feature matrices/priors
exactly match all 22 previously evaluated in-pool references.

The H200 pilot app is `ap-hztoC1aLLGkZRobjbTZhbJ`; it is queued with zero
allocated GPU workers at this update. Short H200 availability probes in both
blackpantherx86 and gildmember also queued without workers and were stopped.
No model scores have been produced yet.

The finite launcher `scripts/continue_rl_pool_eval.py` is running in tmux
`anytok-rl-pool-controller`. It waits for the existing pilot result, checks a
conservative cost estimate against verified remaining credits, launches the
full sweep and independent confirmation, verifies the scores and generates
`reports/rl-pool-sweep-20k/`. It saves the completed report in the same cloud
Volume. Failures and insufficient-credit stops are visible in the dashboard;
it does not silently switch GPUs or launch training. Logs are in
`modal_results/rl_pool_20k/controller.log`. The pilot caller remains in tmux
`anytok-rl-pool-pilot`. No service manager is used.

## Completed on B300, November12

Full evaluation app `ap-tAYc3Dl2bjsALpj9NhOR2b` completed all1,000 screen
tokenizers and18 independently confirmed candidates. All workers used
`NVIDIA B300 SXM6 AC`, CUDA13.2, the frozen RL20k checkpoint and identical
canonical per-passage baselines. Scoring finished at10:38:42UTC; the app
stopped and has zero GPU workers. A report-upload workspace mismatch was
corrected after inference; no model forward passes needed rerunning.

Largest screen mean-domain BPB: pool0635, BPE/no-regex,120k vocabulary,
1.3114% merge noise,33 RL selections. BPB1.07968049 versus canonical
1.01331280, gap+0.06636769. Fresh confirmation gap+0.10072905,95% interval
[+0.07309398,+0.13823962]. Pool0003 has the largest confirmation score,
gap+0.10085389. Their paired difference interval crosses zero
[-0.00422083,+0.00434434]; these texts do not identify a unique worst one.

The report is `reports/rl-pool-sweep-20k/`, with all1,000 ranks, full
per-passage losses,18 confirmation rows, training exposures, Markdown setup
and interpretation, PNG/PDF plots and checksums. It is also saved under
`blackpantherx86 / anytok-rl-all-pool-20k / report/`; cloud manifest readback
matched. The dashboard is marked complete. Credits after inference were
estimated at$7.61, down from$16.14 before the B300 pilot (billing may lag).
