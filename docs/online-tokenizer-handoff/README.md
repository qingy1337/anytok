# Direct tokenizer policy on Tillicum

This run continues the Uniform LM and hypernetwork from step 20,000 for **2,000 more updates**. It restores the student’s AdamW state and continues its original learning-rate schedule. The REINFORCE policy starts fresh, draws tokenizer settings directly, and learns from raw BPB.

Each scored update builds one byte-level Unigram from a 200 MB mixture. The policy chooses any integer vocabulary size from 32,000 through 128,000, the splitting rule, domain shares, and independent web and Wikipedia language shares. It redraws infeasible settings and logs the rejection. The data and policy start from the same seeds as the frozen Eagle2 run.

Each update performs one student pass. The LM and hypernetwork run with dropout enabled, exactly as they do for training. The continuation BPB already computed by that pass is the REINFORCE reward. The same pass then backpropagates the existing LM and lexical losses. There is no separate per-step evaluation pass and no per-step canonical score. The canonical embeddings and output head stay fixed.

At 100, 250, 500, 1,000, and 2,000 updates, evaluation runs separately with dropout disabled. It rescores the highest observed training reward on the original 156 passages and reports both transferred and canonical BPB. Those milestone scores are stable reporting measurements. They never train the policy or student.

## Check out the code and stage inputs

```bash
git clone --branch online-tokenizer-training-20k https://github.com/qingy1337/anytok.git anytok-online-training
cd anytok-online-training
```

Reuse the Python 3.12 environment from the [raw-BPB run](../raw-bpb-handoff/README.md) with PyTorch `2.15.0+cu132`, Transformers `5.15.1`, and Tokenizers `0.22.2`. Build the generator with Rust, a C compiler, and Maturin.

```bash
python -m pip install 'maturin>=1.7,<2'
(cd runtime/online_unigram && python -m maturin build --release --locked \
  --interpreter "$(command -v python)" --target-dir ../../build/online-unigram-target \
  --out ../../build/online-unigram-wheels)
python -m pip install --no-deps build/online-unigram-wheels/anytok_online_unigram-*.whl
```

Point at the existing verified `input/` directory from the [public input bundle](../raw-bpb-handoff/README.md). The launcher rebuilds the fixed passage panel from that bundle and checks the exact panel hash. The initialization files are in the [public Uniform checkpoint repo](https://huggingface.co/qingy2024/anytok-pool-smollm2-20k/tree/6043d83785fd40293bef0a94d588f525f153ec2f/checkpoints/step-020000). Download the 23.46 GB of weights and optimizer directly onto Tillicum. Allow about 100 GB of free space for the initial download and run outputs.

```bash
export ANYTOK_INPUT=/gpfs/path/to/verified-bundle/input
export ANYTOK_WORK=/gpfs/path/to/persistent/run-storage
export ANYTOK_PYTHON="$(command -v python)"
mkdir -p "$ANYTOK_WORK/logs"
python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
snapshot_download('qingy2024/anytok-pool-smollm2-20k',
                  revision='6043d83785fd40293bef0a94d588f525f153ec2f',
                  allow_patterns='checkpoints/step-020000/*',
                  local_dir=Path(os.environ['ANYTOK_WORK']) / 'uniform-init', token=False)
PY
export ANYTOK_CHECKPOINT="$ANYTOK_WORK/uniform-init/checkpoints/step-020000"
python scripts/run_online_tokenizer_training.py --inputs "$ANYTOK_INPUT" \
  --checkpoint "$ANYTOK_CHECKPOINT" --output "$ANYTOK_WORK/direct-joint-2k" --prepare-only
```

`--prepare-only` verifies the corpus, model, optimizer and panel without an allocation. The checkpoint manifest records the original Uniform restart. This run uses its LM, hypernetwork and AdamW files, with a fresh policy and data stream.

## Smoke test and launch

Run one full 128k update on an H200 first. This checks the exact startup weights, the loaded optimizer, the full training batch, and the held-out scorer. Use your normal Tillicum account in place of `stf` if needed.

```bash
sbatch --account=stf --time=00:15:00 --output="$ANYTOK_WORK/logs/smoke-%j.log" \
  scripts/online_tokenizer_training.sbatch --inputs "$ANYTOK_INPUT" \
  --checkpoint "$ANYTOK_CHECKPOINT" --output "$ANYTOK_WORK/direct-joint-smoke" --smoke
```

A successful smoke ends at step 1 with 128 paired examples, zero nonfinite losses, and a held-out result on 156 passages. Check `train_allocated_gib` and `train_reserved_gib` in `direct-joint-smoke/events.jsonl`. Then submit the actual run.

For a short throughput curve, use separate empty output directories and add `--smoke-vocab-size 32000`, `80000`, or `128000`, plus `--smoke-repeats 3 --overlap-preparation`. Each isolated run starts from the same Uniform checkpoint. Only its last update performs the clean 156-passage confirmation.

```bash
sbatch --account=stf --output="$ANYTOK_WORK/logs/train-%j.log" \
  scripts/online_tokenizer_training.sbatch --inputs "$ANYTOK_INPUT" \
  --checkpoint "$ANYTOK_CHECKPOINT" --output "$ANYTOK_WORK/direct-joint-2k" \
  --overlap-preparation
```

The job requests one H200 and uses scheduler defaults for CPU and host memory. The default wall limit is six hours. The runner commits weights every 500 updates and full restart state at 2,000 or at a graceful segment stop. For a new segment, repeat the launch command with `--resume`. Add `--stop-after N` for an earlier absolute step within this 2k run. On resume, the launcher restores the last full checkpoint and archives scores and weights after that checkpoint before replaying those steps. If it stopped before its first full save, it replays from Uniform 20k with the original seeds.

The `events.jsonl` log records the dropout-on training BPB rewards and the `milestones.json` file records the five clean dropout-off confirmation scores. The launcher writes atomic `status.json` for the dashboard. Register it with [dashboard-entry.json](dashboard-entry.json) and sync the status and logs to the dashboard host.

## Timing tokenizer preparation

Add `--overlap-preparation` for the actual run. Once the final training microbatch produces the complete BPB reward, the direct policy updates. It then draws the next settings and builds the next tokenizer and paired batch on CPU while the GPU finishes the current backward pass and optimizer step. Checkpoint and confirmation steps finish before starting another CPU preparation. A graceful signal finishes any update whose preparation has already begun.

Compare `preparation_wait_seconds`, `policy_seconds`, `training_seconds`, and `step_seconds` in the `train` events. Keep the same flag and code revision on resume. The GPU smoke runs one update at maximum vocabulary size. It measures memory fit and does not measure steady-state overlap.
