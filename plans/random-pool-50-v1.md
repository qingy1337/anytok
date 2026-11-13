# Measured random50 reference for the selected24

Replace the36-setting proxy completely. Draw50 IDs uniformly without
replacement from the full1000-member training pool using Python Random seed
20261117. The exact draw is committed in the JSON plan before scoring. Do
not stratify, exclude previously tested settings, filter on losses or reroll.
The draw overlaps the selected24 at IDs0011 and0912, leaving72 unique settings.

Compare final ZeTT20k, uniform20k and RL20k on the same229 existing stress
passages. Jointly validate all72 encoders under128 positions with32 canonical
targets and canonical-boundary validity. If a raw prefix must be shortened,
use that identical shortened prefix for every setting/model and rescore the
selected24 as well. Preserve all document IDs, targets and bytes. Never
discard a randomly sampled tokenizer for an inconvenient encoding result.

Reuse existing cloud checkpoints, native tokenizers and complete-vocabulary
surface features on blackpantherx86 / anytok-rl-all-pool-20k. Save new raw
contexts and scores under random-pool-50-v1/; do not replace historical
model-comparison results. At most two B300 workers, no CPU/memory reservation.
Three calls are bounded to600seconds each; verify sufficient credits before
launch. No model training or large artifact transfers are required.

Report arithmetic mean BPB gaps for the50 uniformly sampled settings and
the24 selected settings, their difference, medians and settings distributions.
Average BPB within source domains, then weight the five domains equally.
Use paired source-document-cluster bootstrap for text uncertainty and
resample the random50 IDs to quantify tokenizer-sampling uncertainty. Include
their overlap in the pairing; apply the finite-population correction to
tokenizer-sampling deviations. Keep all three trained models paired.
Intervals do not cover different model-training seeds. The stress subset
remains defined from the prior RL-canonical diagnostic, fixed for all models.

Dashboard run: pool-regime-comparison. Local metadata/logs:
modal_results/random_pool_50/. The36-setting proxy is removed from the live
dashboard and current report folder. Keep original experiments as provenance.

Completed on 2026-11-13 UTC in Modal app `ap-SDXaZnfMU2dSPNiIYQm4n6`.
All 216 model/tokenizer pairs finished. Encoding validation retained all 229
passages without shortening any prefix: 198 source documents, 7,328 scored
canonical tokens and 23,074 continuation bytes. Independent domain-balanced
arithmetic reproduced the report aggregates.

Mean transfer gaps (random 50 / selected 24): ZeTT +0.30896 / +0.40008 BPB;
uniform +0.11835 / +0.14238; RL +0.11322 / +0.13429. The random sample also
has substantial loss on this fixed stress subset. The selected group adds
about +0.021 BPB for RL. This is not a prevalence estimate for arbitrary text.

Reproducible data, analysis, explanation and plots: `reports/random-pool-50/`.
The frozen JSON draw remains unchanged. Full per-token scores and all assets
remain in the persistent cloud volume under `random-pool-50-v1/`.
