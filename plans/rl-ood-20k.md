# RL20k versus uniform20k on the unchanged weakness panel

Run on gildmember with H200; no explicit CPU or memory reservation. Reuse its
anytok-pool-20k input/model data and anytok-uniform-trajectories cohorts,
tokenizers, feature arrays, and baseline scores read-only. Copy only LM and
hypernetwork inference weights directly from neural-net-1411 / anytok-pool-rl-20k /
settings-run/step-020000 to gildmember / anytok-rl-ood-20k. All files are checked
against the final committed RL manifest. The historical checkpoint collection
error is separate from validation of the final checkpoint.

Use all12 previously frozen cases, with no new selection or text/cut changes.
Primary historical cases: Chinese35/Chinese, Japanese35/Japanese, Base64k/medical.
The five OOD weak cases, two in-pool weak cases and five controls are reported
separately, determined from the saved uniform20k weakness/utility gates.
Every continuation has32 canonical tokens, with BOS + prefix + suffix capped
at128 positions. Reuse the same native IDs, priors, hypernetwork surface forms,
frozen canonical embedding tables and output head. Each model receives its own
canonical control. Shared grid_inputs validates source articles, document IDs,
context digests, suffix bytes, tokenizer hashes and feature-array hashes.

Store new weights and results under a spec-derived immutable prefix on the new
evaluation Volume; never overwrite uniform results. Local artifacts are score
and provenance metadata under modal_results/rl_ood_20k. Exact prepare/evaluate
call IDs are saved and reused after observer timeouts. Default action is plan.

Compute per-case canonical BPB, transferred BPB and transfer gaps, then paired
RL-minus-uniform effects with5000 source-document-cluster bootstrap draws using
the same helper as the uniform trajectory analysis. Preserve negative controls.
Intervals describe held-out text variation for one training seed per method,
not population or multiple-comparison inference. Do not average across cases.

```
MODAL_PROFILE=gildmember modal run --detach scripts/modal_eval_rl_ood.py --action run
OPENBLAS_NUM_THREADS=2 python scripts/analyze_rl_ood.py
```

The run registry includes the evaluation, logs and12-case progress. Export the
paired-effect plot to plots/rl_ood_20k.png and .pdf using established Manrope styling.

## Verified results (2026-11-10 UTC)

All12 fixed cases completed on H200. Final inference weights (8,625,161,062 bytes) matched the committed RL20k manifest. Source cohorts, native tokenizer files, surface-form features, scored IDs, counts and contexts passed the shared validator; frozen canonical tables matched the original weights.

| Case | Uniform gap | RL gap | RL − uniform [paired95% CI] |
|---|---:|---:|---|
| ood48-chinese35--web-zh (ood_weak) | 0.208577 | 0.213038 | +0.004461 [-0.012476, +0.022671] |
| ood48-japanese35--web-ja (ood_weak) | 0.132078 | 0.138520 | +0.006442 [-0.009146, +0.021614] |
| base64--medical-en (ood_weak) | 0.197768 | 0.152688 | -0.045080 [-0.066297, -0.028911] |
| pool-0416--medical-en (control) | 0.031742 | 0.023284 | -0.008457 [-0.012550, -0.004610] |
| pool-0822--web-ja (in_pool_weak) | 0.140445 | 0.139562 | -0.000883 [-0.008302, +0.006464] |
| pool-0616--web-zh (in_pool_weak) | 0.152158 | 0.163640 | +0.011483 [+0.003088, +0.019230] |
| starcoder2--web-ja (control) | 0.076119 | 0.072031 | -0.004089 [-0.010859, +0.003001] |
| wordpiece-2gb--medical-en (control) | -0.036882 | -0.036338 | +0.000544 [-0.002538, +0.003583] |
| unigram-2gb--web-ja (ood_weak) | 0.154132 | 0.153277 | -0.000855 [-0.007451, +0.005803] |
| qwen25coder--web-zh (ood_weak) | 0.162577 | 0.169534 | +0.006957 [-0.001546, +0.015558] |
| pool-0635--medical-en (control) | 0.023757 | 0.015368 | -0.008390 [-0.013021, -0.003955] |
| ood48-base--web-en (control) | -0.023694 | -0.024418 | -0.000724 [-0.004084, +0.002628] |

Base64k/medical improves in both absolute transferred BPB (0.900581 to0.856170) and transfer gap (0.197768 to0.152688). The paired gap-change interval excludes zero on these96 documents. Medical pool416 and pool635 controls also improve; they were not retained uniform weaknesses on these later-passage cohorts. Pool616/Chinese worsens. The other weak cases have intervals crossing zero. This is selected-panel evidence from one training seed per method, not broad OOD generalization or training-seed uncertainty.

The monitor serves /plots/rl_ood_20k.png and .pdf. Full scores and input audits remain in gildmember / anytok-rl-ood-20k / shared-settings-20k-84704db04ba6. Local metadata: modal_results/rl_ood_20k/{evaluation-spec.json,evaluate-result.json,comparison.json}.
