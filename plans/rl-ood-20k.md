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
