# Matched seen/fresh tokenizer generalization check

The immutable design is `plans/pool-generalization-v1.json`, selected with
seed 20261107 before evaluating losses. It contains 18 random training-pool
members, two in each of nine algorithm/pretokenizer strata. Each gets a fresh
matched tokenizer with identical vocabulary size, merge-noise rate, domain
shares, within-domain recipe and 200 MB raw-byte training quota. Fresh BPE
merge seeds change. Fresh tokenizers use disjoint source-document ranges:
for each bucket, skip whole records past the maximum prefix quota used by
any of the original 1,000 tokenizers. This also produces distinct WordPiece
models, whose vocabulary training would otherwise be deterministic.

This tests unseen tokenizer instances conditional on matched configurations;
it does not test novel mixture/vocabulary/noise settings. No model is selected
or replaced based on speed or eval score. Three selected pool members have
not yet been sampled at 2k; all 18 have been sampled by 6k. Exposure counts
are retained. Primary comparison is the fresh-minus-seen BPB penalty at 12k
and its growth from 6k; 2k is an early reference point.

Evaluate frozen joint checkpoints 2k (csproject), 6k (data-pipelines), and 12k
(deployment), without interrupting ongoing training. All members score the
same 256 saved FineWeb-Edu documents, 8,192 canonical continuation tokens and
38,534 continuation bytes. Jointly trim each raw prefix so every seen/fresh
and canonical encoding fits: canonical BOS + target prefix + 32 canonical
continuation tokens <= 128 total positions. Prefix and continuation bytes
must be identical across all 36 tokenizers and all checkpoints. No documents
are silently dropped; max prefix/total lengths and hashes are recorded.

Report the trained canonical BPB, seen and fresh BPB, both transfer gaps, and
the fresh-minus-seen difference. Aggregate strata according to their original
pool counts, matching uniform sampling from the 1,000 pool members. Use a
paired bootstrap over tokenizer pairs within strata and shared document
indices (5,000 draws), including a paired CI on growth from 6k to 12k.
A positive growing penalty supports a generalization problem; lack of a
significant penalty does not establish equivalence. Only two pairs per stratum
and one checkpoint trajectory limit the strength of conclusions.

Run `python scripts/run_pool_generalization.py` in tmux. Source preparation
runs on group-cron's read-only original corpus; CPU generation and H200 eval
use trenton (remote-worker fallback). No CPU/memory reservations; at most 100
containers. Each fresh model has the same patched tokenizer runtime as its
seen partner. Worker crashes are isolated in subprocesses and failed models
are preserved, never replaced. Fresh-model forecast from matched cases is
$3.84 including margin; the slowest matched cases took about two hours.
Fresh data can change runtime substantially.

Volumes: `anytok-pool-generalization` holds corpus bundles, seen/fresh models,
frozen checkpoint weights, shared panels and results. The fresh training text
is installed into `anytok-pool-generalization-corpus`; original corpora and
training outputs are not modified. Large files transfer directly between
Modal workers and Volumes; only metadata passes through the local caller.
The fixed saved panel is uploaded from its existing local copy. CPU input and
checkpoint copying overlaps fresh-model generation; the H200 starts only when
inputs and all 18 fresh models are ready.

Progress: `modal_results/pool_generalization/status.json`, controller.log,
per-stage logs and fresh-model metadata. Tests:
`python -m unittest discover -s tests -p test_pool_generalization.py -v`.
