# Fixed tokenizer edge cases: 12k, 18k and final20k

The same three useful tokenizer cases were evaluated using the completed20k uniform-pool checkpoint. Original confirmation documents, raw prefixes, native tokenizer files and32 canonical suffix tokens per document were reused. Each checkpoint uses its own trained canonical score as the baseline, with the fixed original canonical input/output embedding tables checked unchanged. Every sequence stays within128 total positions.

| Case | 12k gap BPB | 18k gap BPB | 20k gap BPB | 18k→20k gap change [paired95% CI] |
|---|---:|---:|---:|---|
| Chinese35 (48k) / Chinese | 0.268632 | 0.210457 | 0.208577 | -0.001880 [-0.017157, +0.013243] |
| Japanese35 (48k) / Japanese | 0.172886 | 0.135686 | 0.132078 | -0.003608 [-0.018397, +0.011027] |
| Base (64k) / medical English | 0.215684 | 0.202151 | 0.197768 | -0.004384 [-0.014529, +0.005190] |

All three point gaps improved slightly from18k. Each paired95% interval for the18k→20k change includes zero, so this panel does not resolve those small latest changes. This is not evidence that learning has stopped. All three final20k point gaps remain >=+0.1 BPB, and their individual95% lower bounds also remain above+0.1. These are per-case selected-panel results, not a population or multiple-comparison claim.

| Case | 20k canonical BPB | 20k transferred BPB | 20k gap95% CI | Fewer input tokens |
|---|---:|---:|---|---:|
| Chinese35 (48k) / Chinese | 1.766402 | 1.974979 | [0.164831, 0.254549] | 63.03% |
| Japanese35 (48k) / Japanese | 1.445638 | 1.577716 | [0.102930, 0.163391] | 71.55% |
| Base (64k) / medical English | 0.702813 | 0.900581 | [0.120592, 0.294587] | 22.84% |

Each case uses96 identical confirmation documents and3072 suffix tokens. Chinese scores4943 suffix bytes, Japanese5082, and medical11634. The18k and20k context digests, tokenizer hashes, document IDs/order and byte counts match exactly. Per-document loss gaps and confidence intervals were recomputed locally from saved scores; the20k worker verified model/tokenizer checksums, source text alignment and frozen canonical tables before scoring. Only LM/hypernetwork weights were copied from gildmember to trenton, directly between Modal services.

Persistent evidence: trenton / anytok-pool-generalization / weakness-v1/reruns/step-020000/{spec.json,checkpoint/,report.json,comparison-vs-018000.json}. Prior12k and18k results remain separate.

Reproduce the figure: `python plotting_scripts/plot_uniform_weakness.py --rerun-step 20000`. This includes the available12k/18k/20k points and per-case95% gap intervals.

Report SHA256: 7511e75360e0b23e9ebeec1b77e4d2f11d033ad204f31a25c73ace89cfdf4ee3
18k→20k comparison SHA256: b65e22443201e015a92925790d7af7fbd22db0a547bfc1a0b8f768f9ba0a9ac3
