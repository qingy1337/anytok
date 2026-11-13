# Natural-text transfer gaps: a setting for adversarial tokenizer selection

The question is whether ordinary natural text exposes repeatable transfer
weaknesses that a tokenizer-selection policy can exploit. This is a fixed
evaluation of the final shared-settings RL20k LM/hypernetwork, not new training.
Do not select examples for a desired loss or substitute synthetic copying tasks.

Freeze 36 representative pool settings: three algorithms × three pretokenizers
× four requested vocabulary targets (32k,64k,96k,128k). Choose the nearest actual
vocabulary, with a deterministic hash tie-break, without consulting losses.
Keep previously weak0003/0635 as separately labeled diagnostic controls.
This balanced subset does not estimate uniform performance over all1000.

Use the existing persistent, hash-held-out English web, code, math, medical and
legal corpora. Exclude all source IDs from the previous all-pool evaluation.
Choose128 documents/domain/split, except medical28 (84 fresh documents exist; only58 pass two-window
shared encoding/length validation before any model scoring). Discovery and confirmation source-document IDs and complete
document-text hashes must be disjoint. Extract two disjoint windows/document,
at deterministic locations in the two halves of the full source document.
Reject only for insufficient length or shared encoding validity, recording
all rejected IDs/reasons. The panel is frozen before any model scores.

Use the previous scoring protocol: up to64 canonical prefix tokens,
exactly32 canonical continuation tokens, at most128 total positions including
BOS. Jointly shorten the raw prefix when needed for all38 native encoders;
keep at least16 canonical prefix tokens. Every setting gets the identical
raw prefix, continuation token IDs and scored UTF-8 byte count. Generate
input embeddings from the complete native vocabulary and use the frozen
canonical output head and canonical embeddings for continuation inputs.

Save every continuation token's NLL. Independently check the token losses
against Hugging Face's causal LM loss on every batch. Verify original model,
checkpoint, tokenizer and cached surface-feature checksums. Canonical input
and output tables must remain equal to their original values. Require exact
agreement of canonical per-token losses across GPU workers.

Primary aggregate: BPB within each domain, then equally average five domains.
Also show pooled-byte scores, per-passage90th percentile, mean worst10%, and
fractions above+.1/+.25BPB. Tails use passage-weighted per-passage BPB and
are explicitly distinguished from pooled-byte BPB. Keep negative gaps.

Measure canonical context benefit by comparing the identical continuation
with its actual prefix versus BOS alone. Freeze a per-domain75th-percentile
threshold on discovery and apply it to confirmation. This canonical-only
stratum tests natural dependence on prefix information without choosing on
transfer losses. Report first8/later8-token excess NLL to reveal dilution.
Report fixed, overlapping text tags: image-link markup, structured rows,
numeric density>=8%, a repeated>=3-digit numerical string, and plain prose.
Publish both with and without image-link markup.

Determine whether hard tokenizers remain hard: discovery-vs-confirmation
rank correlation, discovery-selected worst tokenizer's confirmation advantage
over uniform across representatives, and a per-domain selected setting's
confirmation advantage. Do not select and evaluate on the same texts.
Show worst-decile results separately from selection generalization. Bootstrap
paired source-document clusters within domains (two windows move together),
2000 replicates. Report noise and uncertainty honestly. Any promising subgroup
must be supported on confirmation; it is not population prevalence or proof
that an RL policy improves training.

Use at most two B300 workers on blackpantherx86; no explicit CPU/memory.
Measure two initial tokenizer evaluations before the remainder. Check credits
and preserve a$1 margin using a conservative measured runtime estimate.
Cloud folder: `anytok-rl-all-pool-20k/natural-transfer-v1/`. Local metadata:
`modal_results/natural_transfer/`. Dashboard run: `natural-transfer`.
Persist the raw corpus/panel, tokenizers and weights in the cloud; transfer
only score metadata and a few inspectable examples locally. Publish a clean
report with reproducible plots/statistics and checksums on the existing
GitHub branch after verification.
