# Uniform tokenizer transfer across all 40 checkpoints

All 40 saved checkpoints (500–20,000, every 500 steps) were evaluated on 12 fixed tokenizer/domain cases: 480 evaluations. The three historical panels reproduce their separate 12k/18k/20k evaluations. Every checkpoint uses identical document IDs, prefix text, native prefix token IDs, canonical continuation IDs and continuation bytes within each case. Each continuation has 32 canonical tokens; BOS + prefix + continuation stays within 128 positions. Output and continuation input embeddings are the frozen canonical tables.

Seven cases have a final transfer gap ≥0.1 BPB and ≥10% token savings. Four are additional cases on fresh scored passages, including two actual training-pool members. None establishes a practical transfer-gap plateau under the prespecified criterion: the entire 95% slope interval must fit inside ±0.002 BPB per 1k steps over 15k–20k (±0.01 BPB across that window). All seven weak cases still show improving absolute transferred BPB over that window.

| Tokenizer / domain | 20k canonical BPB | 20k transferred BPB | Gap [95% CI] | Input tokens saved | Gap slope / 1k [95% CI], 15k–20k |
|---|---:|---:|---|---:|---|
| ood48-chinese35--web-zh | 1.7664 | 1.9750 | +0.2086 [+0.1648, +0.2545] | 63.0% | -0.00565 [-0.00925, -0.00236] |
| ood48-japanese35--web-ja | 1.4456 | 1.5777 | +0.1321 [+0.1029, +0.1634] | 71.6% | -0.00247 [-0.00475, -0.00022] |
| base64--medical-en | 0.7028 | 0.9006 | +0.1978 [+0.1206, +0.2946] | 22.8% | -0.00116 [-0.00293, +0.00068] |
| pool-0416--medical-en | 0.8491 | 0.8809 | +0.0317 [+0.0153, +0.0478] | 42.1% | -0.00312 [-0.00402, -0.00223] |
| pool-0822--web-ja | 1.3977 | 1.5381 | +0.1404 [+0.1233, +0.1587] | 71.0% | -0.00286 [-0.00427, -0.00147] |
| pool-0616--web-zh | 1.7417 | 1.8939 | +0.1522 [+0.1343, +0.1715] | 57.3% | -0.00729 [-0.00916, -0.00538] |
| starcoder2--web-ja | 1.3977 | 1.4738 | +0.0761 [+0.0638, +0.0883] | 45.0% | -0.00177 [-0.00296, -0.00057] |
| wordpiece-2gb--medical-en | 0.8491 | 0.8123 | -0.0369 [-0.0509, -0.0245] | 15.6% | -0.00137 [-0.00206, -0.00069] |
| unigram-2gb--web-ja | 1.3977 | 1.5518 | +0.1541 [+0.1357, +0.1730] | 72.6% | -0.00292 [-0.00422, -0.00164] |
| qwen25coder--web-zh | 1.7417 | 1.9043 | +0.1626 [+0.1429, +0.1832] | 60.6% | -0.00545 [-0.00718, -0.00375] |
| pool-0635--medical-en | 0.8491 | 0.8729 | +0.0238 [+0.0085, +0.0386] | 41.5% | -0.00324 [-0.00426, -0.00227] |
| ood48-base--web-en | 0.9211 | 0.8974 | -0.0237 [-0.0439, -0.0072] | 11.8% | -0.00128 [-0.00191, -0.00066] |

The in-pool weak cases are Unigram 123k/regex (pool 822, seen 16 updates) on Japanese and BPE 115k/line256 (pool 616, seen 26 updates) on Chinese. Their final gaps are about +0.140 and +0.152 BPB. Pool 616 has only 4.1836% merge noise, so this example does not require extreme noise. Both retain useful lossless compression. The other new weak cases are Unigram 150k/Japanese and Qwen2.5-Coder 151,665/Chinese; both vocabularies are outside the training pool’s 32k–128k range, so they are not evidence of an in-range sampling failure.

The two roughly +0.46 BPB medical screening gaps (pool 416 WordPiece and pool 635 BPE) did not replicate on the later scored passages: final gaps are only about +0.032 and +0.024. The 150k WordPiece medical case also fails to confirm a weakness. These negative results are retained. The original Base64k medical panel is a different early-passage cohort and remains weak.

Japanese35, Base64k medical, pool822 and 150k Unigram have slope intervals inside the looser ±0.005 BPB/1k band. This supports slow improvement at that tolerance, not the stricter plateau claim. Shorter 17.5k–20k and 18k–20k fits are exploratory and can be noisy; they do not replace the primary window. Base64k medical even has a positive 17.5k–20k gap slope, illustrating sensitivity to the chosen endpoint.

The original three cases use 96 documents each and canonical prefix budgets up to 95 tokens. New cases use 256 passages and canonical prefix budgets up to 64. Chinese/Japanese/English fresh cohorts use distinct unseen documents. Medical fresh cohorts use later, nonoverlapping passages from 102 source articles; article IDs overlap screening, so they are not an independent-article confirmation. Article-cluster bootstrap preserves this dependence. Different cases are not averaged into an overall metric.

Uncertainty uses 5,000 paired bootstrap draws of original source-document clusters, shared across all checkpoints and canonical/transferred scores. Raw worker quick confidence intervals are discarded. These intervals describe text variation for a single training seed; checkpoint fits may oscillate, intervals are per case, and no population or multiple-comparison claim is made. Endpoint-selected cases can have selection bias. Exact pool configurations, tokenizer hashes, domain mixtures, exposure counts and utility measurements are saved in trajectory-spec.json.

Removing every passage whose prefix contains a truncated or aliased hypernetwork feature leaves the main residual gaps largely unchanged. Pool 616 and Qwen have no such prefix tokens in this panel at all. This is a sensitivity analysis, not a new primary cohort; it does not remove all possible model architecture limitations or establish that sampling caused the residual gap.

| Case | Clean passages / articles | Clean 20k gap | Clean late gap slope [95% CI] |
|---|---:|---:|---|
| ood48-chinese35--web-zh | 92 / 92 | +0.2085 | -0.00589 [-0.00944, -0.00235] |
| ood48-japanese35--web-ja | 81 / 81 | +0.1284 | -0.00214 [-0.00445, +0.00016] |
| base64--medical-en | 93 / 93 | +0.1994 | -0.00094 [-0.00278, +0.00094] |
| pool-0416--medical-en | 219 / 98 | +0.0291 | -0.00270 [-0.00363, -0.00179] |
| pool-0822--web-ja | 203 / 203 | +0.1431 | -0.00292 [-0.00449, -0.00130] |
| pool-0616--web-zh | 256 / 256 | +0.1522 | -0.00729 [-0.00916, -0.00538] |
| starcoder2--web-ja | 255 / 255 | +0.0766 | -0.00178 [-0.00297, -0.00061] |
| wordpiece-2gb--medical-en | 255 / 102 | -0.0369 | -0.00139 [-0.00207, -0.00071] |
| unigram-2gb--web-ja | 184 / 184 | +0.1570 | -0.00316 [-0.00465, -0.00161] |
| qwen25coder--web-zh | 256 / 256 | +0.1626 | -0.00545 [-0.00718, -0.00375] |
| pool-0635--medical-en | 215 / 99 | +0.0225 | -0.00275 [-0.00364, -0.00185] |
| ood48-base--web-en | 256 / 256 | -0.0237 | -0.00128 [-0.00191, -0.00066] |

The data establish useful residual transfer weaknesses, including members of the sampled pool. They do not establish that uniform sampling has reached a strict wall by 20k. Most weak cases continue improving, and a comparison against a different sampling policy is needed to attribute gains to RL rather than more training.

Three WordPiece/no-regex candidates were excluded before full-trajectory selection after a measured encoding-latency problem; all remaining 48 candidates completed the 9-domain screen without errors. Excluded runtime failures are not counted as high-loss successes.

Figures: [absolute BPB](http://100.84.104.59:1337/plots/uniform_trajectory_bpb.png), [full transfer gaps](http://100.84.104.59:1337/plots/uniform_trajectory_gap.png), [late-window zoom](http://100.84.104.59:1337/plots/uniform_trajectory_late.png), [slope summary](http://100.84.104.59:1337/plots/uniform_trajectory_plateau.png). Each also has a PDF at the same path with a .pdf suffix.

Reproduce: `OPENBLAS_NUM_THREADS=2 python scripts/analyze_uniform_trajectory.py`, then `python plotting_scripts/plot_uniform_trajectory.py`. Scores and cohorts persist in Modal profile gildmember, Volume anytok-uniform-trajectories; original weights remain in anytok-pool-20k. Only score metadata was collected locally. Publish derived reports/figures with `MODAL_PROFILE=gildmember modal run scripts/modal_uniform_trajectory.py --action publish`.
