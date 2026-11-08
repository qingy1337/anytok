# Useful tokenizer weaknesses at uniform-pool step 12k

All three examples below were confirmed on 96 documents independent of the 48-document screening set. Their paired 95% bootstrap intervals lie entirely above the requested +0.1 BPB gap. Gap is against the same checkpoint’s canonical score on identical prefix text and scored suffix bytes.

| Tokenizer / held-out text | Canonical BPB | Transferred BPB | Gap BPB | 95% CI | Fewer input tokens |
|---|---:|---:|---:|---|---:|
| Chinese35 (48k) / Chinese | 1.784536 | 2.053168 | +0.268632 | [0.217328, 0.323444] | 63.03% |
| Japanese35 (48k) / Japanese | 1.460069 | 1.632955 | +0.172886 | [0.136767, 0.212245] | 71.55% |
| Base (64k) / medical English | 0.713855 | 0.929539 | +0.215684 | [0.137282, 0.314592] | 22.84% |

The selected tokenizer files are absent from the 1000-model training pool. All use byte-level BPE/line256 and vocabulary sizes within the training range. Their tokenizer-training corpus budgets and mixture settings differ from those of pool members; this is not a claim that the complete configurations are in distribution. Compression counts cover matched held-out document excerpts (up to 4096 characters), not an end-to-end latency measurement.

Each evaluation sequence is BOS + the same raw prefix encoded by the target or canonical tokenizer + exactly32 canonical suffix tokens. Both frozen canonical embedding tables were checked against the pinned original weights. Total positions are capped at128 for every target and canonical control. The three confirmation sets score3072 suffix tokens each, with4943,5082 and11634 suffix bytes, respectively. This differs from the earlier short English-only FineWeb-Edu panel; scores are comparable within each fixed cohort, not absolute scores across these datasets.

26 native tokenizers were attempted across nine held-out text domains.25 produced scores (225 screening pairs);88 pairs exceeded the screening cutoff and received independent confirmation.79 pairs met both the point gap>=0.1 and input-token reduction>=10% gates;39 also had a 95% CI lower bound>=0.1. This is an exploratory selected panel, not population coverage. DeepSeek-Coder failed surface-form construction for an unmarked FIM control string; it is an interface limitation and was not counted as model weakness.

Audit passed: native prefix round-trips, source text alignment, document separation, suffix IDs, byte counts, per-document canonical-score pairing and position caps were rechecked independently of the scorer.

These results establish weaknesses of the12k checkpoint. They do not establish the performance of the eventual20k checkpoint, or that RL will improve the result.

Persistent evidence: trenton / anytok-pool-generalization / weakness-v1/{plan.json,panels.json,canonical-nats.json,screen.json,confirmation.json,report.json,audit.json}. Model weights remain in persistent Modal storage; only metadata returned locally.

Reproduce plots with `python plotting_scripts/plot_uniform_weakness.py`. The immutable design is in `plans/uniform-weakness-v1.json`; run instructions are in `plans/uniform-weakness-v1.md`.

Report SHA256: 83644c85f83a0d8a83b48d3517876489c1d0cc5dc8aaee6e7713d78a2fca1ffb
Audit SHA256: 37736cbf7e8d89a6937990468414ea20c1aa500c3bf1508a00f9d45b759937c6
Plan SHA256: 4d89f7581c6ca62dd957c83b09a5c091f9f0872a06e93f150950aea79482f038


Follow-up: the fixed cases remain above+0.1 BPB at the completed20k checkpoint; see [the final paired comparison](uniform-weakness-rerun-020000.md).
