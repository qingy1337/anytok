# Continue the entropy-zero regret run from 8,000 to 20,000

Qingyun's joint regret RL20k run with entropy weight 0
(`plans/anytok-pool-regret-rl-20k.md`) trained updates 1-5,288 on Modal
(B300) and 5,289-8,000 on Tillicum (H200) with `scripts/tillicum_regret_job.py`
(fdbd3ea), then paused at its step-8,000 full restart (HF
`cminst/anytok-regret-rl-20k`). The writeup needs this run at 20,000 updates,
and his Tillicum credits were low, so he asked a collaborator to finish it.
It was finished on Tillicum under the Slurm account `stf`, starting
2026-11-20.

Updates 8,001-20,000 continue his committed trajectory from that restart.
This is not a new run, and the experiment does not change.

## What must not change

- **Trainer.** The staged code tree `.raw-bpb-code-d875965262f96b29` that the
  raw-BPB run trained with. Its `scripts/train_joint_smollm_input.py`
  (staged from `runtime/anytok_pool_raw_bpb_rl_20k/scripts/`) has SHA-256
  `c411e683c2251047ad497512aea1ca948ffb0bece398ce5fd21992782e18ce8b`,
  the source run's trainer. Before the check and every training job the
  driver checks, without writing, that the tree matches the cluster checkout
  and this hash. The trainer command comes from the raw-BPB launcher's own
  builder (`scripts/run_raw_bpb_training.py`) with `--reward-objective
  regret`, `--memory-execution optimized` and `--resume`.
- **Inputs.** The bundle from HF `cminst/anytok-training-inputs`. Its
  `manifest.json` has SHA-256
  `53d2968d9a1e1a7917b7f05716bca77acfcf9b61a06a08d35fa2bb3b4bbefb8c`, as in
  `docs/raw-bpb-handoff/BUNDLE_MANIFEST.json`. The driver checks the five
  model and warmup files against it and records the hash in `SOURCE.json`.
- **Settings.** Every training setting is the source run's. The corpus
  position, the optimizer and policy state and every random state come from
  the restart.

Only the input locations (the model and warmup paths) change, and so do the
Slurm account and the upload destination. The guard
`scripts/resume_regret_continuation.py` enforces this. It is adapted from
`scripts/resume_regret_tillicum.py`, which guarded the Modal-to-Tillicum
handoff at 5,288. It wraps the trainer's resume-settings
check: identical settings go to the trainer's own check, and the guard
decides any difference.
A difference is allowed only when resuming the step-8,000 restart, only for
the entropy-zero regret run with this trainer and the optimized memory
execution, and only in `model` and `warmup_checkpoint`, whose file names must
stay the same. A check run may also protect a subset of the saved milestones
(`protected_steps`). The settings record the corpus, recipe, pool and
evaluation data by SHA-256 rather than by path, so those entries must match
too. Any other difference refuses the resume. The guard records the handoff
as a redacted `execution_migration` event at step 8,000 in `events.jsonl`.

## Starting point

The source is the step-8,000 full restart at revision
`89bbf8d4cb9f5fb40399d4c4ea5ed51083499b51`, prefix `checkpoints/step-008000`,
manifest SHA-256
`3829bc297ffdeab836c97de036e78a5ea81ead0e60aa2bf444a0005c9a02bda3`.
`scripts/fetch_regret_continuation_restart.py` ran on a Tillicum login node,
so the 23.5 GB restart moved from Hugging Face straight to Tillicum and only
metadata passed through the caller. It checks every file against the
manifest, the manifest against the repository's `latest_restart.json`, and the
saved run config against the entropy-zero regret run (seeds 42 and 43).

The source run logged update 8,001 after that checkpoint and then stopped.
The fetched `events.jsonl` is therefore cut back to steps up to 8,000; the
later rows are kept only in a copy of the source log,
`events-source-through-8001.jsonl`, as in his own Modal-to-Tillicum handoff.
The starting checkpoint is made read-only.

## Resume check

`scripts/continue_regret_rl.py --check` resumes a hard-linked copy of the run
folder (`check-run/`) to update 8,002, with no uploads and no resubmission.
Its ten checks: the trainer exits 0; exactly one `execution_migration` event,
at step 8,000, changing nothing but the model and warmup locations and
`protected_steps`; train rows exactly 8,001 and 8,002; policy counters 8,001
and 8,002; 128 valid paired examples and at most 128 positions in every row;
the regret reward (`policy_reward_bpb == paired_gap_bpb`); the source run's
logged tokenizer and `paired_batch_sha256` at update 8,001; and a saved
step-8,002 restart. Training refuses to start without a passed
`check-receipt.json`.

Job 505608 (debug QoS, 1 h limit, 0:01:59 elapsed) passed all 10 checks.
Update 8,001 matched the source run's logged update exactly except for the
time per update: tokenizer #771, `paired_batch_sha256` `add5dba1...`, loss
2.0617054746295866, peak 115.18 GiB. The receipt compares these fields for the
check; the training job's update-8,001 row equals the source row in all 32
fields other than `seconds`. The receipt is in `SOURCE.json` and
`records/receipts/` on Hugging Face.

## Allocations and approvals

The approved scope (2026-11-20): uploads to the Hugging Face repository
`Dekiofsa/RLtokenizers`, folder `regret-rl-20k/`; a working folder on scrubbed
storage; the 23.5 GB step-8,000 copy; a resume check of at most 1 h in the
debug queue; and up to two 24 h normal-QoS training allocations on `stf`, the
first queueing the second.

Each job asks for one H200 on gpu-h200 and no explicit CPU or memory; the
scheduler gave 4 CPUs and 120G. The driver records each training allocation
in `allocations.json` and refuses a third (`needs_approval`). Slurm sends TERM
15 minutes before the 24 h limit (`--signal=B:TERM@900`). The trainer then
saves a full restart and exits, and the driver submits the second allocation
(`--dependency=afterany`) before its final uploads, so a walltime stop cannot
break the chain. A failed trainer is never resubmitted. A `pause-request.json`
in the state folder stops any job before it trains or uploads.

One allocation was enough. Job 505613 (normal QoS, 24 h limit) ran from
2026-11-21 10:45 to 2026-11-22 09:35 PST (22:50:25) and trained updates
8,001-20,000 at 6.85 s per update on average. The second allocation was
never queued. With the check, the continuation used 22.87 H200 GPU-hours
(training 22.84, check 0.03).

## Checkpoints and uploads

The source run's grid is unchanged: weights every 500 updates, and full
restarts (adding `state.pt` and `optimizer.pt`) every 2,000 and at segment
ends. Updates 8,001-20,000 gave 24 checkpoint folders, `step-008500` to
`step-020000`: 18 weights-only, and full restarts at 10k, 12k, 14k, 16k, 18k
and 20k.

While the trainer ran, an upload thread in the driver handled each new
folder. It re-checked the folder against its `manifest.json` and uploaded it
straight from Tillicum to `regret-rl-20k/checkpoints/step-XXXXXX/`. It then
read the commit's file list back, compared every size and LFS SHA-256 with
the manifest, committed the redacted metadata and wrote a receipt in
`backups/`. Published metadata replaces the cluster user name with `<user>`,
and `events.jsonl` is published only up to its last complete line. The upload
token stays out of the trainer's environment. `continue_regret_rl.py
--backup-only` publishes and verifies anything still pending, without
training.

`latest_restart.json` is published only after a full restart is verified,
and it never moves back to an older step. It names step 20,000: revision
`62b24d741eeb870e91a15a5008c33cba7c21874e`, manifest
`42a196afcbcd2b18402f5b4e8b29d77840f7239c8694fd5f3deafe0ee7e5e6c5`.

On 2026-11-22 every file was re-checked from outside the cluster: 140 files,
296.08 GB (before `records/` was added; 184 files at `40ae95d6`); all 108
payload files match size and SHA-256; all 5 metadata files
match the cluster's redacted copies; `events.jsonl` is 23,326,526 B, with
20,090 rows and train steps 1-20,000 contiguous.

## Where everything is

Hugging Face `Dekiofsa/RLtokenizers`, folder `regret-rl-20k/`. When reading,
pin revision `40ae95d65a36e6649c51c6ec1c9a89928d57f5c4` (`main` on
2026-11-22).

- `README.md`, `SOURCE.json` (provenance plus the check receipt) and
  `latest_restart.json`.
- `metadata/`: `run_config.json`, `run_config.modal.json`, `initial_eval.json`,
  `heldout_tokenizers.json` and `events.jsonl` (updates 1-20,000).
- `checkpoints/step-008500` to `checkpoints/step-020000`.
- `records/`, added at revision `4f68d62e935be836f14e5c0440e3d0db87c93036`:
  `receipts/` (check, restart, fetch, complete, allocations, submissions,
  driver and backup status, and the 25 upload receipts in `backups/`), `logs/`
  (`slurm-505613.log.gz`, `slurm-check-505608.log.gz`, `fetch.log`) and
  `code/` (the scripts as run). `records/check-run/`, added at revision
  `60670ac8cea5b53695ec43db8160c86856c2b70b`, holds the resume check's own
  log rows after the handoff and its status. The cluster user name is masked
  as `<user>`.

His checkpoints stay in `cminst/anytok-regret-rl-20k`, which holds steps
5,288 to 8,000 only.

The jobs ran in `/gpfs/scrubbed/<user>/anytok-runs/regret-rl-20k` (`run/`,
`check-run/`, `backups/`, `code/`). Scrubbed storage is purged after 60 days
without access, so the Hugging Face folder is the copy of record. The cluster
folder was deleted on 2026-11-22, after everything in it was saved on Hugging
Face and in this repository.

Local metadata (gitignored): `modal_results/regret_rl_20k_continuation/`
(`status.json`, `events.jsonl`, `controller.log`). The monitor-only observer
`scripts/run_regret_continuation.py` writes it by streaming
`scripts/regret_continuation_status.py` to the cluster over the existing SSH
connection; it never submits or cancels anything. Dashboard entry:
`regret-rl-20k-continuation`, category Training.

## Commands

On a Tillicum login node, from a checkout with these scripts. `E` is the
training environment built from `docs/raw-bpb-handoff/requirements.txt`; the
run used `/gpfs/projects/stf/<user>/anytok-data/env/train`, and the job
scripts name the run's own paths and account.

    R=/gpfs/scrubbed/<user>/anytok-runs/regret-rl-20k
    mkdir -p $R/code && cp scripts/fetch_regret_continuation_restart.py $R/code/
    (export HF_HOME=$R/.hf-home HF_XET_CACHE=$R/.hf-home/xet HF_XET_CHUNK_CACHE_SIZE_BYTES=0
     unset HF_TOKEN HUGGING_FACE_HUB_TOKEN PYTHONPATH   # caches stay off the 10 GB home
     nice -n 10 $E/bin/python -I $R/code/fetch_regret_continuation_restart.py --state $R)
    cp scripts/continue_regret_rl.py scripts/resume_regret_continuation.py \
      scripts/continue_regret_rl_check.sbatch scripts/continue_regret_rl.sbatch $R/code/
    sbatch $R/code/continue_regret_rl_check.sbatch    # 505608: writes check-receipt.json
    sbatch $R/code/continue_regret_rl.sbatch          # 505613: queues at most one more allocation

On the local machine, the monitor-only observer (standard library only), then
the report (numpy and matplotlib; the committed figure used numpy 2.0.2 and
matplotlib 3.9.4):

    python3 scripts/run_regret_continuation.py --host <user>@tillicum.hyak.uw.edu \
      --remote-state $R --python $E/bin/python --until complete
    python scripts/extract_regret_continuation.py --sacct <sacct -X -P rows of 505608 and 505613>
    python scripts/report_regret_continuation.py      # from reports/regret-rl-20k-continuation/data/

## Results

The results of the continued run are in `reports/regret-rl-20k-continuation/`.
