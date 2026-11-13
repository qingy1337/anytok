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
