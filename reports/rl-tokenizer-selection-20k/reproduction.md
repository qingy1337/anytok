# Reproduce the figures and inspect the data

## Regenerate on a CPU

From the repository root, using Python 3.12:

```bash
python -m pip install -r reports/rl-tokenizer-selection-20k/code/requirements.txt
python reports/rl-tokenizer-selection-20k/code/reproduce.py --check-only
python reports/rl-tokenizer-selection-20k/code/reproduce.py
```

The first command installs the three numerical/plotting dependencies. The
second verifies the report's file checksums and recomputes saved comparisons.
The third also regenerates all seven PNG/PDF figure sets into
`reports/rl-tokenizer-selection-20k/reproduced-plots/`. It does not overwrite the
published figures. Use `--output /path/to/figures` for another destination.

This workflow needs no GPU, Modal credentials, training logs, checkpoints or
network access after installing dependencies. It uses the score/configuration
files included here. It checks:

* File sizes and SHA-256 hashes against [manifest.json](manifest.json).
* All 12 final paired comparisons and their document-cluster confidence intervals.
* All 432 broad-screen loss aggregates, scored counts and nine joint-domain intervals.
* All 12 component-swap decompositions and their intervals.
* All 40 policy snapshots, exact agreement with training action counts, initial
  sanity-score agreement and the exploration summary.
* Generator language limits and mixture distances, plus both saved corpus RNG
  replay receipts.

This recomputes statistics from saved prediction losses. It does **not** rerun
the LM forward passes or independently validate raw corpus text against the
cloud checkpoints. Original inference code and validation receipts are included
so those operations can be inspected separately.

The plotting code uses the repository's Manrope font when available. A copied
report folder without that font uses Matplotlib's DejaVu Sans fallback. PDF
creation timestamps and plotting-library details can affect output hashes;
checksums identify the supplied files, rather than promising identical bytes
from every rendering environment.

## Where to find each result

| File | Contents |
|---|---|
| [training/rl.csv](data/training/rl.csv), [training/uniform.csv](data/training/uniform.csv) | One row per training update, 20,000 per method. Selected tokenizer, model losses, learning rates and RL metrics where available. |
| [training/sanity.csv](data/training/sanity.csv) | Every saved periodic sanity score, including initialization, for both methods and both fixed tokenizers. |
| [policy/history.json](data/policy/history.json) | All 1,000 probabilities and cumulative selection counts at each 500-step checkpoint. |
| [policy/distributions.json](data/policy/distributions.json) | Probability mass by algorithm, pretokenizer, vocabulary/noise bin and domain, plus top candidates. |
| [policy/exposures.csv](data/policy/exposures.csv) | All 1,000 tokenizers' actual RL/uniform update counts and final RL probabilities. |
| [analysis/exploration.json](data/analysis/exploration.json) | Distinct-tokenizer milestones, expected uniform coverage and low-exposure counts. |
| [evaluation/original-panel.csv](data/evaluation/original-panel.csv) | All 12 final cases: absolute, canonical and gap BPB, counts, paired differences and 95% intervals. |
| [evaluation/original-panel.json](data/evaluation/original-panel.json) | Same final comparison with candidate metadata and provenance. |
| [evaluation/uniform-per-document.json](data/evaluation/uniform-per-document.json), [evaluation/rl-per-document.json](data/evaluation/rl-per-document.json) | Per-passage canonical and transferred negative log likelihoods for the 12 cases. |
| [evaluation/panel-spec.json](data/evaluation/panel-spec.json) | Frozen case definitions, ordered document IDs, source-cluster IDs, scored byte counts and context/tokenizer hashes. |
| [evaluation/broad-screen.csv](data/evaluation/broad-screen.csv) | All 432 pairs in a flat table, including absolute/canonical/gap scores and difference intervals. |
| [evaluation/broad-screen.json](data/evaluation/broad-screen.json) | Per-example losses, document IDs, byte counts, 48 candidate definitions and original screen receipts. |
| [analysis/coverage.json](data/analysis/coverage.json) | Generator-support calculations, reward prediction, exposure associations, domain/stratum summaries, feature masks and swap effects. |
| [analysis/component-swaps.json](data/analysis/component-swaps.json) | All four LM/hypernetwork combinations on every fixed case, with per-passage losses. |
| [analysis/feature-audit.json](data/analysis/feature-audit.json) | Input-feature availability in 22 training reference vocabularies, with passage masks. |
| [analysis/medical-token-audit.json](data/analysis/medical-token-audit.json) | Both rare medical-prefix tokens and their native-string membership across all 1,000 training vocabularies. |
| [analysis/reward-prediction.json](data/analysis/reward-prediction.json) | Nonlinear cross-validation results used in the reward diagnostic. |
| [pool/plan.json](data/pool/plan.json), [pool/ood-48k-plan.json](data/pool/ood-48k-plan.json) | All 1,000 generator configurations and the specialty mixture quotas used for the coverage calculation. |
| [provenance/](data/provenance/) | Source hashes, experiment/validation specifications, initial checks, final checkpoint audit, RNG receipts and cloud output pointer. |

Counts are integer counts; probabilities are fractions from 0 to 1. Negative log
likelihoods are in **nats**, while BPB fields are in **bits per UTF-8 byte**:

```text
BPB = sum(negative_log_likelihood_nats) / (sum(scored_bytes) * ln(2))
gap = transferred_BPB - own_trained_canonical_BPB
delta = RL_value - uniform_value
```

For BPB/gap differences, negative values favor RL. Each method's canonical
baseline uses that method's trained LM. Within a case, byte counts and scored
continuations are paired; between cases, contexts need not match.

Training CSV fields that did not exist in a run's original logs are blank. In
particular, the uniform run did not log training BPB/scored bytes; do not fill
these with zeros or infer them from RL. Adaptive training losses use changing
tokenizers and can cover different continuation spans. Use the fixed evaluations
for the RL-versus-uniform performance comparison.

## Code layout

[code/reproduce.py](code/reproduce.py) is the portable report entry point. It
calls the original statistical/plotting helpers through a temporary metadata
layout, and adds the training/exploration figures.

[code/original/](code/original/) preserves the relevant source snapshots:

* The shared policy: `runtime/anytok_pool_settings_rl_20k/hf_hypernet/settings_policy.py`.
* RL trainer: `runtime/anytok_pool_settings_rl_20k/scripts/train_joint_smollm_input.py`.
* Uniform trainer and shared model/input/loss helpers: `runtime/anytok_pool_20k/`.
* Inference: `scripts/modal_eval_rl_ood.py`, `scripts/modal_coverage_evidence.py`
  and `scripts/pool_ood_eval.py`.
* Analysis: `scripts/rl_ood_eval.py`, `scripts/analyze_coverage_evidence.py`
  and `scripts/analyze_uniform_trajectory.py`.
* Original plotting scripts and relevant policy/evaluation tests.

These originals retain their historical repository/cloud paths. Use the
portable entry point above to reproduce this folder's figures; launching the
original training/inference programs requires the parent repository's full
environment and persistent assets. The pinned GPU packages are recorded in the
Modal image definitions. The compiled Rust binary is not duplicated here;
its build sources are included.

To refresh the export in the original workspace, with its saved local metadata:

```bash
python scripts/export_rl_report.py
```

This maintainer command copies structured metadata and source, recomputes
derived report tables/figures, and refreshes the manifest. It launches no jobs.

## Persistent assets for a full inference rerun

These are Modal Volume locations, requiring access to the named workspaces:

| Asset | Workspace / Volume / path |
|---|---|
| Uniform checkpoints and training inputs | `gildmember / anytok-pool-20k / run/` and `input/` |
| RL checkpoints, every 500 steps through 20k | `neural-net-1411 / anytok-pool-rl-20k / settings-run/` |
| Frozen evaluation contexts and features | `gildmember / anytok-uniform-trajectories /` |
| RL inference weights and final scores | `gildmember / anytok-rl-ood-20k / shared-settings-20k-84704db04ba6/` |
| Published diagnostic package | Prefix recorded in [persistent-analysis.json](data/provenance/persistent-analysis.json), on the same evaluation Volume |

The report includes hashes and scores, not model weights, complete tokenizer
vocabularies or raw text corpora. It is sufficient for reproducing the reported
statistics and figures. A new LM evaluation additionally needs those cloud assets.
Large artifacts transfer directly between cloud services; they are not staged
through the local caller or uploaded to GitHub.
