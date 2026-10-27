# AnyTok

AnyTok trains byte-level BPE tokenizers from a fixed corpus while varying the
training mixture, pretokenization rule, and merge selection. The current
trainers use a 150,000-token vocabulary. Training runs on Modal with the
`group-cron` profile; models and manifests are stored on the `anytok-corpus`
volume.

## Repository layout

- `scripts/modal_prepare_corpus.py` downloads source pools and writes the corpus recipe.
- `scripts/generate_mixture_examples.py` and `scripts/generate_heavy_mixtures.py`
  calculate exact byte quotas for each source bucket.
- `scripts/modal_train_tokenizers.py` trains standard BPE models with regex,
  unrestricted, or 256-character line-chunk pretokenization.
- `scripts/modal_train_all_candidate_noise.py` trains BPE models that sometimes
  choose a random eligible merge. `scripts/modal_train_merge_noise.py` retains
  the earlier top-32 experiment.
- `patches/` contains the changes to Hugging Face `tokenizers` 0.22.2 used by
  the two noise trainers.
- `data/balanced-full-v3.json` records the 23 corpus buckets and pinned source
  revisions. `results/mixture_examples.json` and
  `results/pretoken_pilot_mixtures.json` are input plans, despite their paths.

The repository does not contain corpus documents, trained tokenizer files,
dashboard excerpts, or evaluation outputs. The recipe names paths on the
Modal volume; it does not download those files by itself.

## Prepare a run

Use Python 3.11 or newer with the Modal CLI. The corpus preparation image
installs its own `datasets` and `requests` dependencies. Standard training
uses `tokenizers==0.22.2` in its Modal image.

```bash
export MODAL_PROFILE=group-cron
modal run scripts/modal_prepare_corpus.py --size specialty --verify
python scripts/generate_mixture_examples.py \
  --recipe data/balanced-full-v3.json \
  --output results/mixture_examples.json \
  --code-focus-percent 50
```

The verify command checks the specialty pools on `anytok-corpus`. On a new
volume, submit `--size full`, `--size extra`, and `--size specialty` in that
order. Run `modal run scripts/modal_prepare_corpus.py --publish`, then download
`/recipes/balanced-full-v3.json` if you need to replace the checked-in recipe.
Regenerate plans if you replace it.

Train the standard balanced model, or all five saved mixtures:

```bash
modal run scripts/modal_train_tokenizers.py --smoke --only Base
modal run scripts/modal_train_tokenizers.py
```

The trainer saves models and manifests on the Modal volume. It writes local
segmentations only if `dashboard/sample_passages.json` is present; without
that optional file, the segmentation map is empty.

## Random merge training

The current random rule samples uniformly from pair types that remain
eligible under the normal minimum-frequency and token-length limits. The
noise rate is an integer in millionths: `500000` means that half of merge
steps choose a random eligible pair. A seed makes a run reproducible. Rate
zero follows ordinary frequency-ranked BPE.

Build the patched wheel on Linux x86-64 before running the all-candidate
trainer:

```bash
git clone --depth 1 --branch v0.22.2 \
  https://github.com/huggingface/tokenizers.git /tmp/anytok-tokenizers-all
git -C /tmp/anytok-tokenizers-all apply \
  "$PWD/patches/tokenizers-0.22.2-all-candidate-noise.patch"
mkdir -p build/all-candidate
uvx maturin build --release \
  --manifest-path /tmp/anytok-tokenizers-all/bindings/python/Cargo.toml \
  -o build/all-candidate
```

For the matched 2 GB balanced experiment with original regex
pretokenization:

```bash
MODAL_PROFILE=group-cron modal run scripts/modal_train_all_candidate_noise.py \
  --rates 0,500000,750000 --seeds 1,2,3 \
  --total-bytes 2000000000 --pretokenizer regex \
  --output results/merge_noise_all_regex_2gb.json
```

The 200 MB pilot uses the checked-in `results/pretoken_pilot_mixtures.json`.
The trainer accepts `--pretokenizer line256` for the line-chunk rule. A 2 GB
line-chunk noise run exceeded the memory used by the regex runs, so start
with a smaller pilot for that setting.

## Current parameters

| Parameter | Current support |
| --- | --- |
| Mixture | Exact per-bucket byte quotas in a plan; 9 domains and 23 buckets |
| Vocabulary | Fixed at 150,000 in the training scripts |
| Pretokenization | `regex`, `no-regex`, `line-chunks` for standard training; `regex` or `line256` for all-candidate noise |
| Merge noise | 0–1,000,000 millionths; all eligible pairs or earlier top-32 rule |
| Seed | Nonnegative integer for noisy runs |

All models use byte-level BPE, minimum pair frequency 2, maximum token length
64, and `<|endoftext|>` as the special token. The training scripts hash the
recipe, quotas, and settings into a job ID and reuse a completed matching
model on the volume.
