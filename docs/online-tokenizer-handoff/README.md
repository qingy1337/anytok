# On-the-fly tokenizer training

Start a fresh 20,000-step run from the same SmolLM2-1.7B and identity-warmed hypernetwork as the pool runs. The LM, hypernetwork, and REINFORCE policy all update during training. Each step draws a fresh menu of 64 settings, the policy selects one, and the generator builds that tokenizer. The policy gets the transferred model’s raw continuation BPB before the student update, with entropy weight zero. We also log canonical BPB and the paired gap.

Use `--selector reinforce` to match Allen’s raw-BPB pool baseline. The new generator uses byte-level Unigram, so the comparison changes both the search space and the tokenizer family. The launcher also supports `--selector random` and `--selector predictor`. Use a separate output directory for each.

## Inputs and hardware

On the compute host, check out the launcher branch.

```bash
git clone --branch online-tokenizer-training-20k https://github.com/qingy1337/anytok.git anytok-online-training
cd anytok-online-training
```

Reuse the verified `input/` directory from the [raw-BPB handoff](../raw-bpb-handoff/README.md). It includes the original corpus, starting weights, and held-out sanity set. If it needs to be downloaded again, use the pinned Hugging Face revision and checksum in [bundle-reference.json](../raw-bpb-handoff/bundle-reference.json). Download and verify it on the compute host. Keep at least 600 GB free on persistent storage for the checkpoint grid and temporary writes.

Use one H200 with an exclusive allocation, or a GPU with at least as much usable memory. A prior 128k joint step using this memory path reserved 136.19 GiB on an H200. The frozen Rift search needed less memory because it only ran forward passes. Count on a long job. At 6–8 seconds per step, 20k takes about 33–44 GPU-hours before evaluation and checkpoint writes. Measure throughput on the actual machine.

Use the Python 3.12 environment from the raw-BPB run with PyTorch `2.15.0+cu132`, Transformers `5.15.1`, and Tokenizers `0.22.2`. The launcher uses the Linux `patch` command and the same Linux x86-64 ZeTT helper as the pool run. Build the generator from the repo with Rust, a C compiler, and Maturin.

```bash
python -m pip install 'maturin>=1.7,<2'
(cd runtime/online_unigram && python -m maturin build --release --locked \
  --interpreter "$(command -v python)" --target-dir ../../build/online-unigram-target \
  --out ../../build/online-unigram-wheels)
python -m pip install --no-deps build/online-unigram-wheels/anytok_online_unigram-*.whl
```

## Smoke test and launch

Set these paths on the training machine. Use an empty run directory and record the stdout log outside it. Run the smoke commands under the host’s normal GPU allocation.

```bash
export ANYTOK_INPUT=/cloud/anytok-inputs/input
export ANYTOK_RUNS=/cloud/anytok-runs
export ANYTOK_RUN_PLATFORM='Allen · H200'
export ANYTOK_RUN_PROFILE=allen
mkdir -p "$ANYTOK_RUNS"

python scripts/run_online_tokenizer_training.py --inputs "$ANYTOK_INPUT" \
  --output "$ANYTOK_RUNS/online-smoke" --prepare-only
python -u scripts/run_online_tokenizer_training.py --inputs "$ANYTOK_INPUT" \
  --output "$ANYTOK_RUNS/online-smoke" --stop-at-update 1
python -u scripts/run_online_tokenizer_training.py --inputs "$ANYTOK_INPUT" \
  --output "$ANYTOK_RUNS/online-smoke" --stop-at-update 2 --resume
```

Check `online-smoke/events.jsonl` for two training rows with 128 valid paired examples each, at most 128 positions, and `policy_reward_bpb` equal to `train_bpb`. The second command should restore step 1 and write a full step 2 checkpoint. Check both `peak_gib` and `reserved_gib`. Vocabulary size varies by step, so this checks startup and resume while the earlier 128k test supplies the max-size memory reference.

Launch the real run from a fresh output directory.

```bash
python -u scripts/run_online_tokenizer_training.py --inputs "$ANYTOK_INPUT" \
  --output "$ANYTOK_RUNS/online-reinforce-20k" \
  > "$ANYTOK_RUNS/online-reinforce-20k.log" 2>&1
```

Resume the same run by appending `--resume` before the log redirection and using `>>` to keep the existing stdout log. A graceful segment stop can use `--stop-at-update N`. Weights are saved every 500 steps, and full restart state every 2,000 plus a graceful segment endpoint. If the process dies after its last full restart, resume preserves later weights and logs under `superseded/` and replays those steps from the verified full checkpoint.

## What gets recorded

- The training update uses 128 examples, microbatches of 32, and at most 128 total positions per input. Native and canonical prefixes use corrected paired cuts with identical canonical continuation tokens and bytes. The original LR schedule, lexical loss, fixed output head, and both existing sanity tokenizers stay in place.
- The generator targets 200 MB per draw. Settings vary over every integer vocab size from 32k to 128k, regex, line-256 and no-regex splitting, continuous domain fractions, and independent web and Wikipedia language fractions. The menu respects available bytes in each bucket. Requests without enough distinct pieces are recorded and redrawn without a reward. The `events.jsonl` row stores the selected settings, seed, actual bucket quotas, and tokenizer hash so it can be regenerated.
- The full checkpoint stores the student’s AdamW state, the policy state, the corpus cursor, and all random streams. `run_config.json` pins the trainer, generator and policy hashes. Checkpoints have SHA-256 manifests.
- The run writes atomic `status.json` and timestamped `events.jsonl`. Register the run using [dashboard-entry.json](dashboard-entry.json). Fill in the actual platform, profile and paths. For the shared dashboard, mirror the status and logs to its host and keep model checkpoints and corpus files in cloud storage.
