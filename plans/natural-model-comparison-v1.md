# Compare final trained models on the fixed context-dependent regime

The user asks whether the natural-text stress setting is extensive enough,
how much headroom it exposes, and how the different trained models perform.
Compare the final20k ZeTT-sampled-tokenizer run, uniform1000-pool run and
shared-settings RL run. Reuse the corrected natural-transfer-aligned-v2 panel
and all38 native tokenizer settings. Existing RL results are reused exactly;
score the other two model/hypernetwork pairs with their own canonical controls.
No training is launched.

Freeze the existing RL-canonical context-benefit thresholds and selected
passage IDs. Do not recompute the membership separately for each model. The
primary confirmation stress set has229 passages from198 documents (7,328
continuation tokens;23,074 bytes), spanning web54, code61, math57, medical17
and legal40 passages. Discovery has251 passages from203 documents. Report
this selection as conditioned on RL's canonical model, not a model-neutral
unbiased benchmark. Whole-panel results are an accompanying control.

Score the full2,032-passage parent panel for all38 settings, with identical
raw prefixes, native IDs, canonical targets, document IDs and scored bytes.
Keep128 total positions,32 canonical continuation targets and valid canonical
token boundaries. Generate embeddings from the complete native vocabularies.
Use the existing checksum-verified surface-form features and priors. Both
canonical tables must remain frozen at their original values. The shared
scoring helper checks every batch against the model's causal loss.

Primary summaries: absolute transferred BPB, model-specific canonical BPB,
and their gap, equally weighting five source domains. Keep the fixed0771
tokenizer, the24 weak-pretokenization representatives and12 regex controls,
and all36-representative summaries separate. Report no-regex/line/regex
groups, and the large-vocabulary subset, so one hard setting does not stand
in for a broad basin. Include paired RL-minus-uniform and RL-minus-ZeTT
effects on the same passages, with source-document-cluster bootstrap within
domains. Show both absolute-loss changes and transfer-gap changes; their
signs need not agree when trained LM canonical quality differs.

Checkpoints: uniform20k from gildmember / anytok-pool-20k / run/step-020000;
ZeTT20k from grading / anytok-zett-20k / run/step-020000. Both original full
manifests are read as metadata and pinned in the plan. Only three inference
files/model (8,625,161,062 bytes) are copied directly between cloud volumes;
optimizer and restart state are excluded. Local caller receives metadata only.

Use blackpantherx86 / anytok-rl-all-pool-20k / natural-model-comparison-v1/,
at most two B300 workers, no explicit CPU or memory reservations. Verified
remaining credits at planning:$6.76. BoundedGPU calls each1200seconds;
expected measured native-setting cost is about5seconds/setting. Keep the
dashboard entry natural-model-comparison visible. Local metadata/logs:
modal_results/natural_model_comparison/. Source is committed before launch.

## Completed

Both additional final checkpoints were copied and verified cloud-to-cloud.
All76 model/tokenizer pairs completed on B300 in
`ap-AdFdZwMtnx55XlwOVtv4sI`. The38 original RL results are reused unchanged.
No model training was run. Final remaining credits were estimated at$4.57.

On the same229 confirmation stress passages, canonical BPB is0.68044 ZeTT,
0.67984 uniform and0.68029 RL. For fixed WordPiece/no-regex/127k tokenizer0771:

| Model | Transferred BPB | Transfer gap |
|---|---:|---:|
| ZeTT20k |1.16970|+0.48926|
| Uniform20k |0.86476|+0.18491|
| RL20k |0.85804|+0.17775|

Across the24 weak-pretokenization representatives, median gaps are+0.41835,
+0.14240 and+0.13437 respectively. All24 RL point estimates lie between
+0.09114 and+0.17775. The large-vocabulary weak subset contains12 settings;
its medians are+0.45151,+0.15638,+0.14736. The corresponding regex-control
medians are+0.06378,+0.04662,+0.04858.

Paired mean RL-minus-uniform transfer-gap change over the24 weak settings:
−0.00809BPB [−0.01236,−0.00320]. Absolute transferred-loss change:
−0.00764 [−0.01185,−0.00295]. For tokenizer0771 alone, the RL-minus-uniform
interval crosses zero; do not claim that single-setting gain is established.

The settings region contains648 members of the1000 pool, including206 with
vocabulary>=96k. Those counts describe generator configurations, not proof of
a large corrected gap for every member. The corrected paired evaluation
samples24 weak representatives and12 regex controls. The confirmation text
set is229 passages /198 documents /7,328 target tokens /23,074 scored bytes,
22.5% of the parent confirmation passages. It is a diagnostic panel, not a
large population benchmark. All models use the same RL-defined membership.

Report: `reports/natural-model-comparison-20k/`, including reproducible
per-passage loss totals, model hashes, paired confidence intervals, PNG/PDF
plots, readable Markdown and checksums. Full per-token scores remain in the
cloud model folders. The report is mirrored under
`natural-model-comparison-v1/report/` in the same persistent volume.
