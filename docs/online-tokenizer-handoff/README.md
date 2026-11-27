# On-the-fly adversarial tokenizer training

This is the handoff for one 20,000-update run on a Tillicum H200.

The run starts from the LM, hypernetwork and AdamW state at the end of the
Uniform 20k run. The tokenizer policy starts fresh. At every update, the policy
proposes tokenizer settings directly, a new byte-level Unigram tokenizer is
built from those settings, and the LM and hypernetwork train on that tokenizer.
There is no fixed tokenizer pool and no menu of candidate settings.

## What happens in one update

1. The policy chooses a vocabulary size, a text-splitting rule, domain shares,
   and separate web and Wikipedia language shares.
2. The generator builds a new lossless byte-level Unigram tokenizer from a
   200 MB sample with exactly those shares.
3. We make 128 training examples. The generated tokenizer encodes the prefix.
   The continuation remains canonically tokenized, and the fixed canonical LM
   output head scores it. BOS, prefix and continuation together use at most 128
   positions.
4. The LM and hypernetwork run once with dropout enabled. Continuation BPB from
   this training pass is the policy reward. There is no separate scoring pass.
5. REINFORCE updates the tokenizer policy to favor higher BPB. The LM and
   hypernetwork update in the opposite direction, lowering continuation loss.

The student also uses the existing lexical alignment loss with weight 0.5.
That loss is not part of BPB and is not given to the policy. Canonical input
embeddings and the canonical output head stay frozen.

After the policy receives its reward, CPU preparation for the next tokenizer
overlaps the remaining backward and optimizer work. This overlap is always on.

## What the policy can choose

| Setting | Support |
|---|---|
| Vocabulary size | Every integer from 32,000 through 128,000 |
| Splitting | regex, 256-character line chunks, or no regex |
| Domains | Any nonempty subset and shares over web, Wikipedia, code, papers, math, QA, books, medical and legal |
| Web languages | Any nonempty subset and shares over Arabic, English, Spanish, French, Hindi, Japanese, Russian and Chinese |
| Wikipedia languages | The same eight-language support, chosen separately |

The algorithm is Unigram. The three splitting choices are not three tokenizer
algorithms. A fresh generator seed is drawn independently for every accepted
update.

The policy is direct REINFORCE with Adam learning rate 0.001, an EMA reward
baseline with decay 0.99, and entropy weight 0. It learns a shared distribution
over these settings rather than one probability per saved tokenizer.

## Clean reporting

Training rewards use dropout because they come from the training pass. At
updates 100, 250, 500, and every 1,000 updates afterward, the run evaluates the
highest training reward seen so far with dropout disabled. This clean check uses
the same fixed 156-passage panel as the frozen direct-policy experiment: 4,992
canonical continuation tokens and 14,429 continuation bytes across nine
domains. Clean evaluation never trains the policy or student.

## Preliminary results

These are sanity checks, not results from the planned joint 20k run.

### Frozen direct-policy search

We froze the Uniform 20k LM and hypernetwork and let the same direct policy
propose 2,000 tokenizers. It found useful settings early and stronger settings
later.

| Search prefix | Step where the best setting appeared | Clean held-out transfer gap |
|---:|---:|---:|
| 100 | 42 | +0.1545 BPB |
| 1,000 | 934 | +0.2110 BPB |
| 2,000 | 1,912 | +0.2553 BPB |

The step-42 setting used a 34,346 vocabulary, no regex, and a broad mixture led
by QA, legal, books and code. The step-934 setting used an 89,331 vocabulary,
no regex, and 77.9% math. The step-1,912 setting used a 44,522 vocabulary,
256-character line chunks, and 100% math.

The proposal distribution changed during search. Settings placing more than
80% of their data on one domain increased from 4.8% in the first 500 proposals
to 29.2% in the last 500. This shows learning, although the policy was still
broad at step 2,000.

### One-pass reward check

On 32 policy-proposed tokenizers, we compared one dropout-off score with eight
independent dropout-on scores on identical text and model weights.

| Measurement | Result |
|---|---:|
| Mean dropout-on shift | +0.00235 BPB |
| Median dropout standard deviation for one tokenizer | 0.00087 BPB |
| Tokenizer-to-tokenizer standard deviation | 0.04085 BPB |
| Mean rank correlation | 0.9985 |
| Hardest-tokenizer agreement | 8/8 draws |
| Top-eight overlap | 8/8 in every draw |

The extra dropout-off pass was therefore removed from training. It remains only
in the clean milestone evaluation.

### H200 timing

The exact one-pass update was measured on a Tillicum H200.

| Vocabulary size | Settled seconds per update | Peak reserved GPU memory |
|---:|---:|---:|
| 32k | 2.554 | 61.12 GiB |
| 80k | 4.219 | 99.21 GiB |
| 128k | 6.382 | 136.20 GiB |

For an initially uniform 32k–128k size distribution, the fitted mean is 4.302
seconds per update. The 20k update loop is 23.9 hours. Allow roughly 24–26
hours after startup, checkpoints, resumes and clean evaluations. At the
Tillicum rate of $0.90 per H200-hour, that is about $22–$24. A policy that
strongly favors 128k vocabularies will take longer.

The full joint 20k experiment has not been run yet. The preliminary checks show
that the generator works, the direct policy can discover hard regions, the
one-pass reward is stable, and the largest vocabulary fits. They do not show
that this curriculum beats Uniform training.

The same numbers are saved in `preliminary-results.json`.

## Files and environment

Clone the public branch:

```bash
git clone --branch online-tokenizer-training-20k https://github.com/qingy1337/anytok.git anytok-online-training
cd anytok-online-training
```

Use Linux x86-64 and Python 3.12. The verified H200 environment used PyTorch
2.15.0+cu132 and CUDA 13.2. Install a Rust toolchain and a C compiler before
building the tokenizer generator.

```bash
python -m pip install -r docs/online-tokenizer-handoff/requirements.txt
(cd runtime/online_unigram && python -m maturin build --release --locked \
  --interpreter "$(command -v python)" --target-dir ../../build/online-unigram-target \
  --out ../../build/online-unigram-wheels)
python -m pip install --no-deps build/online-unigram-wheels/anytok_online_unigram-*.whl
```

Keep data and checkpoints on persistent storage. A complete checkpoint grid is
about 0.5 TB. Allow at least 600 GB for extracted inputs, initialization,
outputs and temporary checkpoint writes.

## Download the exact inputs

The public input archive is pinned to Hugging Face revision
`c2421b0c046967a7eb05fa628546c5058f69d035`. It is 9,314,144,730 bytes with
SHA-256 `51f24daf94303f63f00b2d5458c7d34e346163fffa3cefac5d55f36edde3bf5f`.

```bash
export ANYTOK_DATA=/gpfs/path/to/persistent/anytok-inputs
mkdir -p "$ANYTOK_DATA"
python - <<'PY'
import os
from huggingface_hub import hf_hub_download
for name in ('anytok-training-inputs-v1.tar.gz',
             'anytok-training-inputs-v1.tar.gz.sha256'):
    hf_hub_download('cminst/anytok-training-inputs', repo_type='dataset',
                    revision='c2421b0c046967a7eb05fa628546c5058f69d035',
                    filename=name, local_dir=os.environ['ANYTOK_DATA'], token=False)
PY
cd "$ANYTOK_DATA"
sha256sum -c anytok-training-inputs-v1.tar.gz.sha256
tar -xzf anytok-training-inputs-v1.tar.gz
cd -
python scripts/verify_training_inputs.py --root "$ANYTOK_DATA"
export ANYTOK_INPUT="$ANYTOK_DATA/input"
```

The verification should report 1,062 files, 1,000 saved historical tokenizers,
and 23 corpus buckets. The saved pool is present for provenance and other
experiments. This run does not select from it.

## Download the Uniform 20k initialization

The public checkpoint revision is
`6043d83785fd40293bef0a94d588f525f153ec2f`.

```bash
export ANYTOK_WORK=/gpfs/path/to/persistent/online-20k
mkdir -p "$ANYTOK_WORK/logs"
python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
snapshot_download('qingy2024/anytok-pool-smollm2-20k',
                  revision='6043d83785fd40293bef0a94d588f525f153ec2f',
                  allow_patterns='checkpoints/step-020000/*',
                  local_dir=Path(os.environ['ANYTOK_WORK']) / 'uniform-init',
                  token=False)
PY
export ANYTOK_CHECKPOINT="$ANYTOK_WORK/uniform-init/checkpoints/step-020000"
```

This directory contains the LM, hypernetwork and AdamW state. The launcher
checks their hashes before doing any work.

## Verify, smoke test and launch

The preflight performs no GPU work:

```bash
python scripts/run_online_tokenizer_training.py \
  --inputs "$ANYTOK_INPUT" \
  --checkpoint "$ANYTOK_CHECKPOINT" \
  --output "$ANYTOK_WORK/run" \
  --prepare-only
```

Run one isolated 128k update before the real job:

```bash
export SLURM_ACCOUNT=your_account
sbatch --account="$SLURM_ACCOUNT" --time=00:15:00 \
  --output="$ANYTOK_WORK/logs/smoke-%j.log" \
  scripts/online_tokenizer_training.sbatch \
  --inputs "$ANYTOK_INPUT" \
  --checkpoint "$ANYTOK_CHECKPOINT" \
  --output "$ANYTOK_WORK/smoke" \
  --smoke
```

Use a new output directory for the real run:

```bash
sbatch --account="$SLURM_ACCOUNT" \
  --output="$ANYTOK_WORK/logs/train-%j.log" \
  scripts/online_tokenizer_training.sbatch \
  --inputs "$ANYTOK_INPUT" \
  --checkpoint "$ANYTOK_CHECKPOINT" \
  --output "$ANYTOK_WORK/run"
```

The Slurm file requests one H200 and does not request CPU or host memory. Its
wall limit is six hours. Five minutes before the limit, the launcher finishes
the current update and writes a full restart checkpoint.

To continue the same run, submit the same command with `--resume`:

```bash
sbatch --account="$SLURM_ACCOUNT" \
  --output="$ANYTOK_WORK/logs/resume-%j.log" \
  scripts/online_tokenizer_training.sbatch \
  --inputs "$ANYTOK_INPUT" \
  --checkpoint "$ANYTOK_CHECKPOINT" \
  --output "$ANYTOK_WORK/run" \
  --resume
```

Repeat until `status.json` reports step 20,000 and stage `complete`. Do not
change code, inputs or flags between segments. `--stop-after N` is available
for an intentional bounded run, where `N` is the absolute update within this
20k continuation.

## Checkpoints and logs

- `status.json` is the atomic dashboard status.
- `events.jsonl` contains every training reward, tokenizer setting and timing.
- `milestones.json` contains the clean fixed-panel evaluations.
- `step-XXXXXX/` contains a checksum manifest and saved weights.
- Every 500 steps saves LM, hypernetwork and policy weights.
- Every 2,000 steps also saves optimizer, RNG and corpus state for exact resume.
- A graceful segment stop always saves full resume state, even between 2k marks.

The expected size is about 8.63 GB for a weight-only checkpoint and 23.46 GB
for a full checkpoint. The run records source hashes, input hashes, seeds and
the complete configuration in `run-config.json`.

For the repository dashboard, add the run with `status.json` as a
`progress_json` source and `step` as the completed field. The total is 20,000.
The dashboard is read-only; launch and resume from Slurm.
