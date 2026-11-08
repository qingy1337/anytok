# Fixed tokenizer edge cases: 12k versus 18k

The user requested a rerun on the newest saved checkpoint. At selection time, training was past18k and step018000 was the newest committed checkpoint. The three LM/hypernetwork weight files were transferred directly from gildmember/anytok-pool-20k into trenton/anytok-pool-generalization/weakness-v1/reruns/step-018000/checkpoint/. No optimizer or state files were transferred.

The exact original96-document confirmation cohorts, raw prefixes, tokenizer files and32 canonical suffix IDs per document were reused. Frozen input/output embedding tables, checkpoint and tokenizer checksums, prefix token counts, suffix bytes, document IDs/order and maximum128 positions were checked. The original12k search files remain unchanged.

| Case | 12k canonical BPB | 18k canonical BPB | 12k transferred BPB | 18k transferred BPB | 12k gap | 18k gap | Gap change [paired95% CI] |
|---|---:|---:|---:|---:|---:|---:|---|
| ood48-chinese35 / web-zh | 1.784536 | 1.769471 | 2.053168 | 1.979928 | 0.268632 | 0.210457 | -0.058175 [-0.081661, -0.036228] |
| ood48-japanese35 / web-ja | 1.460069 | 1.451023 | 1.632955 | 1.586710 | 0.172886 | 0.135686 | -0.037200 [-0.055003, -0.019177] |
| base64 / medical-en | 0.713855 | 0.703343 | 0.929539 | 0.905494 | 0.215684 | 0.202151 | -0.013532 [-0.026027, -0.001408] |

All three gaps improved beyond the LM’s own canonical-score improvement. Each paired95% interval for the change lies below zero. Each18k gap remains above+0.1, including the lower bound of its per-case95% interval. These are per-case paired-document intervals, with no population or multiple-comparison claim.

All cases score3072 continuation tokens. Chinese scores4943 UTF-8 suffix bytes; Japanese5082; medical11634, matching12k exactly. Input compression remains unchanged because the tokenizer files and excerpts did not change.

This comparison establishes behavior at18k, not the eventual20k checkpoint. It does not demonstrate that adaptive sampling or RL would fix these weaknesses.

Reproduce: `MODAL_PROFILE=trenton modal run --detach scripts/modal_find_uniform_weakness.py --rerun-latest` (selects latest at invocation); `--collect-only` reattaches to the saved call. Plot this immutable result with `python plotting_scripts/plot_uniform_weakness.py --rerun-step 18000`.

Persistent report: trenton / anytok-pool-generalization / weakness-v1/reruns/step-018000/report.json. Local score metadata: modal_results/uniform_weakness/reruns/step-018000/report.json.
