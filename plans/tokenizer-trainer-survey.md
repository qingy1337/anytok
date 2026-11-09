# Existing tokenizer trainers

Checked 2026-11-08. Scope: faster **training**, retaining the full 200 MB input.
None of the alternatives below has been verified to reproduce our pinned
Hugging Face trainer's vocabulary, IDs, scores and settings exactly. No new
training or package installation was launched for this survey.

| Algorithm | Existing implementation | Evidence and fit |
| --- | --- | --- |
| Unigram | [SentencePiece](https://github.com/google/sentencepiece) | Established C++ implementation with multithreaded EM training. First external implementation to investigate. Its published README throughput table measures encoding, not training; it establishes no training speedup on our workload. |
| BPE | [rustbpe](https://github.com/karpathy/rustbpe) | Focused Rust trainer with Rayon, Python iterator input, configurable regex and tiktoken export. The README provides no matched training benchmark against our pinned trainer. |
| BPE | [bpeasy](https://github.com/gautierdag/bpeasy) | Compact Rust byte-level trainer with configurable regex, token length limit and export to HF/tiktoken. Its own training benchmark reports a modest advantage, not 5–10×. |
| BPE | [fastBPE](https://github.com/glample/fastBPE) | Existing C++ trainer (`learnbpe`). The conventional word-oriented NMT pipeline is less directly compatible with our byte-level and no-regex settings. |
| BPE | [YouTokenToMe](https://github.com/VKCOM/YouTokenToMe) | Multithreaded C++ trainer with substantial historical training-speed gains over HF. The repository has been archived since April 2024; its exposed settings do not cover our custom merge noise or establish HF output equivalence. |
| WordPiece | [Hugging Face tokenizers](https://github.com/huggingface/tokenizers/blob/v0.22.2/tokenizers/src/models/wordpiece/trainer.rs) | Already our native Rust baseline. This pinned version trains a BPE model internally and transfers its vocabulary into WordPiece. No separately verified faster exact-output replacement found. |
| WordPiece | [TensorFlow Text / Apache Beam vocabulary learner](https://www.tensorflow.org/text/guide/subwords_tokenizer) | Existing scalable alternative. The official guide explicitly recommends Beam for scalability, but describes a top-down vocabulary-learning procedure, so it is not an equivalent replacement for HF's trainer. |

## What the available benchmarks actually show

[YouTokenToMe's historical benchmark](https://github.com/VKCOM/YouTokenToMe/blob/master/benchmark.md)
uses Wikipedia text, 30k vocabulary and four threads for YouTokenToMe. At 100 MB,
its reported training times are 4.2 seconds versus HF's 10.3 seconds for English,
and 11.1 versus 61.6 for Japanese (about 2.4–5.5× across the four languages).
At 1 GB it reports 25.4 versus 97.7 seconds for English and 106.7 versus 1981.3
for Japanese. These are the project's older measurements against older library
versions, not verified gains on our full generator or evidence of identical
tokenizers. They are a stronger training-speed lead than encoding-only claims.

[bpeasy's published C4 benchmark](https://github.com/gautierdag/bpeasy/blob/main/benchmarks/README.md)
reports HF training at 0.8165 seconds and bpeasy at 0.68815 seconds (about 1.19×),
with large reported standard deviations. It also reports different compression,
so those numbers are not proof of identical learned tokenizers. They cannot
predict speed on our 200 MB, 32k–128k, sometimes unsegmented workload.

[rustbpe's training API](https://github.com/karpathy/rustbpe#api-reference)
exposes vocabulary size, buffering and regex. It does not document our custom
all-candidate merge-noise semantics. The similarly named tiktoken primarily
provides inference; rustbpe supplies a trainer, rather than speeding up encoding
alone.

[SentencePiece's training options](https://github.com/google/sentencepiece/blob/master/doc/options.md)
include training threads, seed vocabulary size, shrink factor, sub-iterations,
maximum piece length and pretokenization controls. A valid comparison would need
the exact original byte-mapped chunks, counts and boundaries, with normalization
and default script/number/whitespace splitting matched and no data sampling or
long-sentence skipping. Even matched options do not establish identical seeds,
EM numerics, pruning, ties or final token IDs. Its default raw-text pipeline is
not an equivalent replacement for our ByteLevel setup.

The TensorFlow tutorial's small example timings are not a comparison with HF
and do not establish a speedup for our generator. Likewise, SentencePiece Lite,
fast WordPiece encoding papers and most high-throughput tokenizer libraries
accelerate use of an already trained vocabulary; they do not solve vocabulary
training.

## Decision

Other leads checked: [U32k](https://huggingface.co/xTimeCrystal/U32k) claims a GPU
Unigram EM trainer, but its public file listing contains model/config files and
a README, without trainer source or training-speed measurements. It is not yet
a usable trainer from this evidence. [ztok](https://github.com/sirus20x6/ztok)
exposes training for all three families; its headline parity and speed claims
concern encoding existing models, not reproducing HF vocabulary training.

Keep the original pool-v2 trainer active. SentencePiece (Unigram) and rustbpe /
bpeasy (BPE) are external candidates worth investigating before writing another
trainer. Replacing an implementation would first require exact learned-output
checks on matched data and settings. Until then, there is no substantiated
5–10× exact-output replacement for this experiment.
