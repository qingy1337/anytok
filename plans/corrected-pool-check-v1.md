# Corrected broad-text check: fixed random 50 and the old worst

User chose the inexpensive check before considering another full-pool sweep.
The original frozen random sample already contains tokenizer 0635, the old
screen winner. Therefore this run has **50 unique tokenizers**, not 51. Keep
all 50, including that overlap, without rerolling or loss-based exclusions.

Use only the final RL20k checkpoint. Score the original broad-text screen
and confirmation samples: 288 passages each, 32 per each of nine domains.
No context-benefit filtering, new text selection or stress-subset selection.
This is distinct from the recent 229-passage diagnostic.

An encoding-only audit found mismatched canonical tokenization across the
cut on 128/288 original screen passages and 118/288 confirmation passages.
Keep every already-valid context exactly unchanged. For invalid cuts, choose
the nearest reversible canonical token end within 64 Unicode characters.
Retain the old prefix start; trim only as necessary to keep a 16–64-token
canonical prefix and all native prefixes at most 95 tokens. Score 32 canonical
targets with a maximum of 128 positions including BOS. A shifted cut changes
the target bytes; record that explicitly rather than claiming identical
continuations between old and corrected protocols. All tokenizers within
each protocol share exactly the same passages, prefixes and targets.

Compare the old and corrected random-50 means, the fixed 0635 score, and
regex/line/no-regex group means. Use BPB within each domain, then equal weights
across all nine domains. Report screen and confirmation separately. Group
comparisons are observational rather than matched pretokenizer ablations.
This sample cannot establish the new worst across the complete pool.

Reuse full-vocabulary features, tokenizer files and RL checkpoint already in
blackpantherx86 / anytok-rl-all-pool-20k. Save the new panel and raw per-token
scores under `corrected-pool-check-v1/`. Only metadata passes locally. One
B300, no CPU/memory reservations, one GPU call capped at 600 seconds; require
$2.20 verified remaining credit before preparing and launching. Latest credit
estimate at planning: $3.57. Dashboard run `corrected-pool-check`.

The frozen JSON plan is committed before inference. Encoding logic tests use
a real BPE merge to exercise the split-token bug and verify valid contexts
are preserved. The scorer reuses the previously checked random-50 scorer,
with an explicit output prefix to keep old files intact.

Completed on 2026-11-13 UTC in app `ap-wTRzJNmOz6kF1n6jFIrHgj`.
All 50 tokenizers were scored on all 576 passages. Exactly 128 screen cuts
and 118 confirmation cuts changed; maximum moves were 12 and 20 characters.
All 160 + 170 already-valid contexts were preserved exactly. Each split
scores 9,216 canonical targets; corrected bytes are 27,039 and 26,668.

Corrected random-50 mean / #0635 gap: screen +0.06829 / +0.10398 BPB;
confirmation +0.09011 / +0.13782. The old screen estimate for #0635 was
+0.06637. It remains the worst among the sampled 50 on both splits.
The corrected full-pool worst is unknown. Report: `reports/corrected-pool-check/`.

An initial CPU container startup missed the shared scoring module. It was
stopped before GPU inference; the corrected images include that module.
The successful run log is `modal_results/corrected_pool_check/retry.log`.
