# Setup, in plain language

## The question

We want a language model to work with many different input tokenizers. A
tokenizer turns text into token IDs. Different tokenizers split the same text
into different pieces, so their IDs cannot simply reuse the model's original
embedding table.

A **hypernetwork** generates an input embedding for each token in the new
tokenizer. The language model then reads a prefix using those generated
embeddings and predicts the continuation using its original vocabulary.

The experiment asks whether choosing difficult training tokenizers with an RL
policy produces better transfer than choosing them uniformly at random.

## The two runs

| Setting | Uniform baseline | Shared-settings RL |
|---|---|---|
| Training length |20,000 updates |20,000 updates |
| Available tokenizers | Same fixed 1,000 | Same fixed 1,000 |
| Choice each update | Uniform random | Sample from learned probabilities |
| Starting model and hypernetwork | Same initialization | Same initialization |
| Examples per update |128 |128 |
| Maximum positions per example |128, including BOS |128, including BOS |
| What learns | LM backbone and hypernetwork | LM backbone, hypernetwork and policy |

The LM is **SmolLM2-1.7B**. The hypernetwork starts from the same 10k-step identity
warmup used by the uniform run. The RL run starts fresh from that initialization;
it does not continue the finished uniform model. The earlier ID-only policy
experiment, stopped at 335 updates, is separate and is not the RL run reported here.

The model's original input embedding table and output head remain fixed. The LM
backbone and input hypernetwork learn. Training uses B300 GPUs; final inference
evaluations use H200 GPUs.

## What happens in one training update

1. Draw 128 text snippets from the pinned, mixed-domain training corpus.
2. Select **one tokenizer for the whole 128-example update**.
3. Split each snippet into prefix and continuation. Encode the prefix with the
   selected tokenizer and the continuation with the model's original tokenizer.
4. Generate the selected tokenizer's input embeddings with the hypernetwork.
5. Measure how well the LM predicts the continuation. Update the LM and
   hypernetwork to reduce their training objective.
6. In the RL run, update the policy using the continuation loss measured before
   the model update. Higher loss encourages selecting that tokenizer more often.

There are 20,000 tokenizer selections, with repetitions counted, and 2,560,000
training examples. This is not 20,000 different tokenizers.

## Exactly where the context goes

Every example has this shape:

```text
[BOS] [prefix tokens: read as context] [continuation tokens: predict and score]
  └──────────────────── at most 128 total positions ────────────────────────┘
```

**BOS** is the one start position. Prefix tokens provide context and are not
scored. Only continuation tokens contribute to the prediction loss.

The tokenization method changes the prefix. Continuation IDs and the output
vocabulary always use the original, canonical tokenizer. During prediction,
earlier continuation tokens also provide context for later continuation tokens.

There are three different measurement settings in this report:

| Setting | Prefix and continuation rule | Position limit |
|---|---|---|
| Training | Start from up to 512 characters; split text approximately halfway by character count. Shorten and retry until it fits. Both token counts vary. | BOS + prefix + continuation ≤128 |
| Periodic sanity evaluation | Two fixed tokenizers; fixed raw text splits, 92 examples each from 23 held-out text buckets. Continuation length varies. | Both canonical and transferred versions fit within 128 |
| Final 12-case panel and 432-case screen | Same saved prefix text and **exactly the next 32 canonical tokens** after the cut, for both methods and both models. | Both versions fit within 128 |

For the final evaluations, 32 scored tokens plus one BOS leave at most 95 prefix
positions. The same prefix **text** can have different token counts under two
tokenizers. Some frozen cohorts use a smaller prefix budget. Per-case budgets,
counts and hashes are in [the panel specification](data/evaluation/panel-spec.json).
The 128-position cap does not mean 128 scored continuation tokens.

## What the policy sees and learns

For every candidate tokenizer, the policy sees its generator settings:

* Algorithm: BPE, WordPiece or Unigram.
* Pretokenizer: regex, line256 or no-regex.
* Requested vocabulary size: 32k–128k.
* BPE merge noise: 0–75%, plus whether noise applies.
* Nine text-domain fractions: web, Wikipedia, code, papers, math, QA, books,
  medical and legal.

It does not receive tokenizer IDs, seeds, current model state or the current
text batch as features. One shared network maps 18 features through two 64-unit
Tanh layers to one score per candidate. A softmax turns the 1,000 scores into
sampling probabilities. There are 5,441 shared parameters and no per-tokenizer
trainable parameters. Initial probabilities are exactly uniform.

The reward is **raw continuation BPB**. The policy favors tokenizers on which
the current model has higher prediction loss. REINFORCE subtracts a moving
average reward baseline (decay 0.99); Adam uses learning rate 0.001. An entropy
bonus of 0.01 encourages a spread of probabilities. There is no forced-uniform
sampling fraction or guaranteed probability floor. That matters for interpreting
the exploration audit.

The LM and hypernetwork minimize token prediction loss plus an input-embedding
alignment penalty weighted 0.5. That penalty is excluded from the policy reward.
Their peak learning rates are 1e-6 and 6e-5 respectively. Both runs use the same
10k-update ramp and 190k-update schedule, stopping at 20k; the schedule was not
compressed to 20k. Model/corpus seed is 42; policy seed is 43. Source snapshots are
under [code/original/](code/original/).

## What the pool covers

Each saved tokenizer was built using 200 MB of text. The full configurations and
mixture recipe are in [data/pool/plan.json](data/pool/plan.json). LM training also
uses a mixed-domain corpus; the policy changes tokenizer selection, not the
corpus sampling proportions.

Languages inside the web/Wikipedia domains have fixed proportions. They cannot
be adjusted independently with the existing nine-domain knobs. The largest
Chinese/Japanese fraction in a pool tokenizer is 8.07%; the theoretical maximum
of the current generator is 12.5%. Some difficult test tokenizers use 35% of their
focused language. These percentages refer to text used to **build tokenizers**.
They do not mean the LM never sees Chinese or Japanese.

## How to read the scores

**BPB means bits per UTF-8 byte. Lower is better.** Dividing prediction loss by
the number of scored text bytes makes the unit independent of how many tokens
the prefix tokenizer uses.

For each checkpoint we measure:

* **Canonical BPB:** that trained LM reads the prefix with its original input
  embeddings.
* **Transferred BPB:** the same LM reads the same prefix text using the new
  tokenizer and hypernetwork-generated embeddings.
* **Transfer gap = transferred BPB − canonical BPB.** Positive means an extra
  cost from transfer; negative means transfer performs better on that panel.

Each run uses its **own trained canonical baseline**. An improvement in the LM
itself can lower absolute transferred BPB without reducing the transfer gap.
The report therefore supplies all three values. The final comparison uses the
same saved documents, prefix text and scored continuation bytes. It verifies
tokenizer/feature/model hashes, context hashes and scored counts.

The 12-case intervals use 5,000 paired bootstrap draws of source documents;
passages from the same article stay together. The 432-case screen is a separate
exploratory panel of 48 selected tokenizers across nine domains, 48 examples per
pair. Its domain summaries resample source articles jointly across tokenizers.
These are selected panels and one training seed per method; a small interval
does not establish robustness across training seeds.
