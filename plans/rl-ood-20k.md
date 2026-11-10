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


## Coverage and reward evidence audit (2026-11-10 UTC)

The evidence supports broadening generator coverage, but does not establish that
continuous generation is necessary or sufficient. Unlimited sampling of the
current generator cannot express the most relevant missing language mixtures.
The fixed pool is also not the sole explanation: substantial gaps remain on
in-pool tokenizers and on contexts whose individual features already occur in
training vocabularies.

The audit re-evaluated all48 existing utility-selected tokenizers across nine
domains (432 paired cases,48 saved screening documents per case). Every
continuation has32 canonical tokens and at most128 positions including BOS.
Both models are20k checkpoints; each has its own trained canonical control.
Uniform reproduction error was exactly0.0 BPB. Model, tokenizer, surface-form,
document and continuation checks passed. The original12-case confirmation
panels were retained for feature and component-swap checks. Screening effects
are exploratory; intervals cover source-article variation, not training seeds.

### What favors a coverage explanation

* The pool's largest Chinese or Japanese tokenizer-training byte fraction is
  **8.07%**. The theoretical maximum of the existing nine-domain generator is
  **12.5%**, because languages have fixed proportions inside web/Wikipedia.
  Their fractions move together. Chinese35 and Japanese35 target35% of the
  focused language. The best possible current-generator mixture still requires
  reassigning27.39% of text-mixture mass; the nearest saved mixture requires31.80%.
  This is an exact mixture-support calculation, not a loss-generalization bound.
* Chinese35 and Japanese35 have identical nine-domain feature vectors. The
  shared policy cannot distinguish their language focus with this feature schema.
  Independent language/bucket controls and matching policy features are needed.
* In Base64k/medical,25 of96 passages contain one of two prefix-token forms absent
  from the22 evaluated training reference vocabularies. Those passages comprise
  only10.02% of scored bytes but **96.63% of uniform excess loss** and80.67% of
  RL's gap improvement. Their gap is1.90684 BPB under uniform and1.54398 under RL;
  the other71 passages have gaps0.00740 and−0.00228 respectively.
* Querying all1,000 SHA-verified native vocabularies found **`{#sp` in zero**.
  It occurs38 times in24 medical prefixes. The other form, `utro`, occurs once
  in one prefix and is present in15 training vocabularies. Native-string absence
  is not exhaustive absence of equivalent hypernetwork feature encodings, and
  the passage split is observational: it does not isolate token embeddings from
  source-text difficulty. The marker makes this a narrow coverage clue rather
  than evidence of a general medical-language failure.

### What prevents a simple coverage-only conclusion

* In just22 training vocabularies,95.40% of Chinese35 prefix-token occurrences
  and97.36% of Japanese35 occurrences already have the same surface-form tuple.
  Including the prior-range requirement gives93.61% and94.73%.
  Availability is a lower bound; it does not demonstrate gradient or joint-context
  exposure. On passages with every feature and prior supported, uniform gaps
  remain0.19869 Chinese (39 passages) and0.12100 Japanese (40 passages); RL gaps
  are0.21278 and0.12467. Individual feature novelty cannot explain all of this.
* In-pool616/Chinese is still weak despite RL sampling47 batches versus26 under
  uniform, including333 versus177 Chinese examples. Its original-panel gap
  worsens by0.01148 [0.00309,0.01923]. Pool822/Japanese receives23 versus16 batches
  and143 versus116 Japanese examples, yet its Japanese
  gap is effectively unchanged. Exposures are endogenous, so this is not a causal
  claim that extra training hurts, but unseen tokenizer identity is not required
  for a large gap. Learning-rate-weighted exposures were also audited.
* Pool vocabulary sizes are32k–128k; some controls are150k–152k. Specialty
  tokenizer-training corpora used2GB versus200MB per pool tokenizer. These
  additional shifts prevent attributing all OOD effects to domain mixtures.

### What the policy and reward checks say

Both runs' final Python RNG hashes match the replayed corpus-bucket draws exactly.
The LM does see Chinese and Japanese (about6.89 examples each per128-example
batch); medical contributes6.4. The missing coverage concerns tokenizer
configurations and their input representations, not absent LM language data.

Nonlinear five-fold held-out-batch prediction explains2.62% of policy-advantage
variance from time+settings, versus19.65% from time+batch-domain counts and20.35%
from both. Holding tokenizer IDs out gives2.47%,19.46%,20.12%. Linear prediction
has the same qualitative result. Batch composition is a substantial source of
reward variation; the unexplained remainder is not necessarily pure noise.
A context-conditioned baseline using pre-action batch information is a plausible
variance-reduction follow-up that preserves the raw-BPB objective; it is untested.

The policy is not random: effective count is about855,989 pool members were
sampled, and final probability correlates+0.506 with final raw difficulty on the
22 selected in-pool probes. Log RL/uniform exposure correlates−0.663 with mean
transfer-gap change, including a negative association after excluding each
algorithm. These are selected, endogenous associations, not causal estimates.

Across the432 fixed screen pairs, the mean medical gap change is−0.01406 BPB
[−0.02494,−0.00546] using shared-source-article bootstrap draws. Chinese is
−0.01353 [−0.02669,+0.00015], Japanese−0.00331 [−0.01662,+0.01020]. These screen
means do not replace the separately frozen confirmation-panel results.

### Component swaps

All four LM/hypernetwork combinations were scored on the unchanged12 panels;
UU and RR controls reproduce original scores within1e−5 BPB. For Base64k/medical:

| LM | Hypernetwork | Absolute transferred BPB |
|---|---|---:|
| Uniform | Uniform | 0.900581 |
| Uniform | RL | 0.860347 |
| RL | Uniform | 0.898392 |
| RL | RL | 0.856170 |

Swapping only the hypernetwork on the uniform LM changes BPB by−0.04023
[−0.06046,−0.02480], compared with−0.00219 [−0.00371,−0.00087] for the LM swap.
The interaction is−0.00199; total change−0.04441. Most observed medical improvement
is recovered by the hypernetwork swap with the LM fixed. Co-adaptation and the
interaction prevent treating this as a universal additive causal decomposition.

### Implication for the next experiment

Broaden the generator's reachable language/bucket mixtures and expose those
settings to the policy. Keep the agreed200MB tokenizer-training budget and the
same128-position LM setting. To test continuous generation specifically, compare
a broader fixed pool with fresh generation from the **same broader distribution**
at matched compute/tokenizer budgets; the original narrow pool is a baseline.
Hold model training, policy/reward, evaluation cohorts and new unseen tokenizer
seeds fixed across comparisons. Otherwise a gain cannot distinguish better
coverage from continuity. Use fresh confirmatory tokenizers/passages after this
exploratory diagnosis. A small prefix-token decomposition ablation would further
test the `{#sp` mechanism before investing in another long run. None of these
new training/generation experiments has been launched.

Reproduce analysis and all four styled figures:

```
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python scripts/analyze_coverage_evidence.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python plotting_scripts/plot_coverage_evidence.py
```

Metadata: modal_results/coverage_evidence/{analysis.json,paired-screen.csv,
feature-audit.json,medical-token-audit.json,component-swaps.json,nonlinear-reward.json}.
The monitor's coverage runs serve the overview, full432-cell heatmap, policy and
mechanism figures. Persistent source/result Volume:
gildmember / anytok-rl-ood-20k / shared-settings-20k-84704db04ba6 /
coverage-evidence-<spec digest>; the final analysis package has its own file-hash
manifest under analysis/ and contains source, report, metadata and PNG/PDF figures.
