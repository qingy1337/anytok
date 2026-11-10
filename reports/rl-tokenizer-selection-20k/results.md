# Results

## What happened during training

The LM and hypernetwork improved substantially, especially early on. RL's two
fixed sanity scores stayed close to uniform; it did not establish a consistent
advantage. At 20k, the transfer gaps were:

| Fixed sanity tokenizer | Uniform gap BPB | RL gap BPB |
|---|---:|---:|
| T1 | 0.04678 | 0.04871 |
| T2 | 0.01617 | 0.01524 |

Lower is better. These two small sanity panels do not establish broad
performance. Their continuation lengths vary; they are not the 32-token final
panels. [Complete sanity scores](data/training/sanity.csv) include both absolute
BPB and each checkpoint's own canonical baseline.

The policy changed its preferences. At 1k, about 82% of its probability mass was
on WordPiece and less than 1% on Unigram. At 20k it assigned about 45% to Unigram,
40% to WordPiece and 16% to BPE (rounded). It sampled 989 different tokenizers
throughout training. Its largest final individual probability was 0.208%, versus
0.100% under uniform. These final statistics conceal early concentration and
uneven cumulative exposure, discussed below.

[Training plot](plots/training_sanity.png) · [PDF](plots/training_sanity.pdf)

## Final evaluation on the original 12 cases

The table reports **transferred BPB minus that run's own trained canonical BPB**.
Lower gaps are better. The difference is RL minus uniform; negative favors RL.
All cases use their original saved text and the next 32 canonical continuation
tokens, within 128 total positions. Each row is a separate fixed panel.

| Tokenizer / text | Role | Uniform gap | RL gap | Difference [95% interval] | Result on this panel |
|---|---|---:|---:|---|---|
| Chinese35 / Chinese | OOD weakness | 0.20858 | 0.21304 | +0.00446 [-0.01248, +0.02267] | No clear difference |
| Japanese35 / Japanese | OOD weakness | 0.13208 | 0.13852 | +0.00644 [-0.00915, +0.02161] | No clear difference |
| Base BPE 64k / medical | OOD weakness | 0.19777 | 0.15269 | -0.04508 [-0.06630, -0.02891] | RL improves |
| Pool 416 / medical | Control | 0.03174 | 0.02328 | -0.00846 [-0.01255, -0.00461] | RL improves |
| Pool 822 / Japanese | In-pool weakness | 0.14045 | 0.13956 | -0.00088 [-0.00830, +0.00646] | No clear difference |
| Pool 616 / Chinese | In-pool weakness | 0.15216 | 0.16364 | +0.01148 [+0.00309, +0.01923] | RL worsens |
| StarCoder2 / Japanese | Control | 0.07612 | 0.07203 | -0.00409 [-0.01086, +0.00300] | No clear difference |
| WordPiece 150k / medical | Control | -0.03688 | -0.03634 | +0.00054 [-0.00254, +0.00358] | No clear difference |
| Unigram 150k / Japanese | OOD weakness | 0.15413 | 0.15328 | -0.00085 [-0.00745, +0.00580] | No clear difference |
| Qwen2.5-Coder 152k / Chinese | OOD weakness | 0.16258 | 0.16953 | +0.00696 [-0.00155, +0.01556] | No clear difference |
| Pool 635 / medical | Control | 0.02376 | 0.01537 | -0.00839 [-0.01302, -0.00395] | RL improves |
| Base BPE 48k / English | Control | -0.02369 | -0.02442 | -0.00072 [-0.00408, +0.00263] | No clear difference |

“OOD” means the tokenizer was absent from the training pool. An in-pool weakness
uses a training tokenizer on held-out text. The controls are retained comparisons;
control does not imply a zero gap. The weakness/utility grouping comes from the
saved uniform 20k panel, not from selecting favorable RL results. Five tokenizers
are outside the pool across the five OOD-weak rows; some also differ in vocabulary
size or tokenizer-training data budget.

The clearest OOD improvement is **Base BPE 64k/medical**: absolute transferred BPB
falls from 0.90058 to 0.85617, and its transfer gap falls from 0.19777 to 0.15269.
Two in-pool medical controls also improve. Pool 616/Chinese worsens. The other
weaknesses have intervals spanning zero; that means the direction is inconclusive
on these panels, rather than proving equal performance.

[Final comparison plot](plots/rl_ood_20k.png) · [PDF](plots/rl_ood_20k.pdf) ·
[Full scores/counts/intervals](data/evaluation/original-panel.csv)

## The broader 432-case screen

We also compared 48 existing, compression-selected tokenizers across nine text
domains, 48 examples per pair. Both methods were rescored on the exact saved
screening contexts. The maximum uniform reproduction error was **0.0 BPB**.

Across the 48 fixed tokenizers, the mean medical transfer-gap change was
**−0.01406 BPB [−0.02494, −0.00546]**. Chinese was−0.01353
[−0.02669, +0.00015]; Japanese was−0.00331 [−0.01662, +0.01020].
This supports a modest medical benefit on the selected screen, with inconclusive
Chinese/Japanese effects. These screen averages concern different passages
from the 12-case panels and should not be substituted for those results.

[All 432 cells](plots/coverage_evidence_heatmap.png) ·
[PDF](plots/coverage_evidence_heatmap.pdf) ·
[Full table](data/evaluation/broad-screen.csv)

## Exploration: the strongest new caution

| Training update | Different tokenizers tried by RL | Actually tried by uniform | Expected under uniform |
|---|---:|---:|---:|
| 1,000 | 525 | 636 | 632 |
| 2,000 | 620 | 860 | 865 |
| 5,000 | 769 | 988 | 993 |
| 10,000 | 943 | 1000 | 1000 |
| 20,000 | 989 | 1000 | 1000 |

At 20k, **157 tokenizers received fewer than five training updates** under RL;
under the actual uniform run, none did. All 157 use regex pretokenization. Eleven
pool tokenizers were never selected by RL. One selection trains on 128 examples,
so these counts measure complete model updates, not individual examples.

The policy temporarily became much more concentrated than its final distribution
suggests: its effective count reached about 264, compared with 1,000 for uniform.
The final effective count recovered to about 855. “Effective count” is
exp(entropy), a measure of probability concentration; it is not the number of
unique tokenizers visited.

**Insufficient exploration is a plausible contributor, not an established cause.**
Some difficult cases received more updates under RL and still did poorly. A
forced-uniform exploration fraction would test this explanation, but no such
comparison was run. We cannot conclude that extra exploration would fix the
results, or that the remaining explanation must be continuous generation.

[Exploration plot](plots/exploration.png) · [PDF](plots/exploration.pdf) ·
[All 1,000 exposure counts](data/policy/exposures.csv) ·
[Policy history plot](plots/coverage_evidence_policy.png)

## Additional diagnostics, separated from conclusions

**Generator coverage:** the largest Chinese/Japanese tokenizer-training text
fraction in the pool is 8.07%. The current generator cannot exceed 12.5% or vary
the two languages independently, while the skewed test tokenizers target 35%.
This is a real limitation of the available settings. It does not prove that
broadening them improves transfer. Continuous generation using the same narrow
knobs cannot reach the missing mixtures either.

**The medical result is concentrated:** 25 of 96 Base64k/medical passages contain
one of two prefix-feature forms absent from the 22 reference pool vocabularies.
They comprise 10.02% of scored bytes but 96.63% of uniform excess loss and 80.67%
of the RL gap improvement. The token `{#sp` appears 38 times in 24 prefixes and
in none of the 1,000 native training vocabularies. The other token, `utro`, appears
once and exists in 15 training vocabularies. This is an association with a small
set of passages, not a token-removal intervention or proof of broad medical
language weakness. Native-string absence does not exhaustively test equivalent
hypernetwork encodings.

**Component swaps:** on the medical case, replacing only the uniform hypernetwork
with the RL hypernetwork changes absolute BPB by−0.04023; replacing only the LM
changes it by−0.00219. The full improvement is−0.04441, including a−0.00199
interaction. Most of this observed improvement can be recovered with the
hypernetwork swap while the LM is held fixed. Hybrid components can be
co-adapted, so this is not a universal additive explanation.

**Known features do not eliminate the Chinese/Japanese gaps:** on passages where
every input feature has a matching form and an in-range prior in the 22 reference
vocabularies, uniform gaps remain 0.19869 for Chinese and 0.12100 for Japanese.
Feature availability alone does not demonstrate useful gradient or joint-context
exposure. Missing individual token forms cannot explain every failure.

**Reward variation:** held-out nonlinear prediction explains about 2.6% of policy
advantage variation from time plus tokenizer settings, versus 19.7% from time plus
batch-domain counts. This is descriptive prediction, not proof that the
unexplained remainder is noise or that batch composition caused poor learning.
Both runs' final corpus RNG states match the replay used for this audit.

[Diagnostic overview](plots/coverage_evidence_overview.png) ·
[Overview PDF](plots/coverage_evidence_overview.pdf) ·
[Feature and component-swap plot](plots/coverage_evidence_mechanism.png) ·
[Mechanism PDF](plots/coverage_evidence_mechanism.pdf)

## What this experiment establishes

The shared policy changed selection probabilities, the model learned, and some
medical cases improved. This run does not establish a broad advantage over
uniform sampling or a specific cause for the remaining weaknesses. Early
under-exploration and restricted generator coverage are both supported concerns.
Exploration, generator changes and continuous generation have not yet been
compared in controlled follow-up training runs.

All intervals concern variation across held-out source documents for one training
seed per method. The panels are selected and the many comparisons are not
multiplicity-adjusted. Read the observations as an experiment record and
hypothesis-generating evidence.
