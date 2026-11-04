# ZeTT 20k on Modal

Objective: train the joint SmolLM2/ZeTT model from the pinned warmup to update
20,000 and retain LM plus hypernetwork weights every 500 updates for later
evaluation curves. Keep full restart checkpoints every 2,000 updates.

## Canonical inputs and code

- Launcher: `scripts/modal_zett_20k.py`; profile transfer: `scripts/modal_zett_20k_artifacts.py`.
- Trainer snapshot: `runtime/zett_20k/`. This is a pinned copy of the working
  ZeTT source for this run; the separate `zett/` checkout remains the research
  workspace. Do not mix revisions during a resume.
- Input hashes and byte sizes: `plans/zett-modal-20k-inputs.json`.
- Source corpus: `zett/.joint_corpus/`; heldout evaluation: `modal_results/eval/`.
- Base model: SmolLM2-1.7B revision `effd688a12921b4cc83e3312b6feb579f70f9c71`.
- Warmup: `zett/output_smollm2_input_warmup_2048_retry/step-010000`.

## Run contract

The three Modal profiles are `projects`, `grading`, and `test-joining`, in that
order. Each owns a distinct v2 volume named `anytok-zett-20k`. The GPU function
requests `B300` only; it does not set CPU or memory. Each invocation has a
10,800-second stop timer, below the 4-hour Modal timeout, to keep one profile's
GPU charge comfortably within its $30 credit. Modal credits are not a hard
spending cap, so inspect `modal billing summary --json` before each segment.

Every 500 updates the trainer saves model, hypernetwork, optimizer, RNG and
corpus state in `run/step-NNNNNN/`, adds a hash manifest, and commits the
volume. Once a newer full checkpoint is committed, it deletes optimizer and
state files from the preceding 500-step checkpoint unless that step is a
multiple of 2,000. An interrupted segment also saves an off-grid restart
checkpoint. The next segment resumes the latest full checkpoint. The first
`projects` app used the original all-full saver; after it stopped, step 1,500
was audited and older grid saves were compacted.
After a handoff is audited in the next profile, the old source's terminal
checkpoint is compacted or removed if off-grid. The final mixed grid belongs
in `test-joining`'s volume. The trainer's 23-bucket heldout
check is a run sanity metric; the FineWeb-Edu tokenizer-transfer evaluation is
a separate paired sweep over these saved checkpoints.

## Commands

Run from `/home/zli/anytok` with the local Modal CLI environment:

```bash
MODAL_PROFILE=projects modal run scripts/modal_zett_20k.py --action plan
python scripts/modal_zett_20k_artifacts.py upload projects
MODAL_PROFILE=projects modal run scripts/modal_zett_20k.py --action audit
MODAL_PROFILE=projects modal run scripts/modal_zett_20k.py --action smoke
MODAL_PROFILE=projects modal run --detach scripts/modal_zett_20k.py --action train --max-seconds 1800
MODAL_PROFILE=projects modal run --detach scripts/modal_zett_20k.py --action train
python scripts/modal_zett_20k_artifacts.py status projects
```

The 1,800-second first segment measures updates per dollar and leaves a
resumable checkpoint. Continue in `projects` with the normal 10,800-second
limit only if the measured pace leaves a credible path to 20,000 updates
within the three credits. After that segment stops, upload and audit inputs
in `grading`, then run
`python scripts/modal_zett_20k_artifacts.py handoff projects grading`. Repeat
for `grading` to `test-joining`. Run the train command in each destination
profile. Use `modal run --detach` for each long segment so a local disconnect
does not stop the remote worker. Once update 20,000 is committed in
`test-joining`, run
`python scripts/modal_zett_20k_artifacts.py collect`. `collect` transfers one
checkpoint at a time and needs about 25 GiB local free space; it verifies each
download against its checkpoint manifest. It collects the 500-step grid and
converts any remaining non-2,000-step full saves to weights only. A full save is
about 23.46 GB, and a weights-only save about 8.63 GB. Ten full and thirty
weights-only checkpoints should occupy about 494 GB, or about 503 GB with
the 8.87 GB inputs, in the final volume.

The unattended controller handles the remaining capped segments, audits each
handoff, collects the grid, and writes `run/final_audit.json` in the final
volume. It stops on a failed stage or if a planned segment would put metered
compute above $29 in its profile. Start it in a named tmux owner session;
its log and compact status are under `modal_results/zett_20k_controller/`.

```bash
python scripts/modal_zett_20k_controller.py --resume-grading --plan
tmux new-session -d -s zett-modal-20k 'bash scripts/run_modal_zett_20k_controller.sh --resume-grading'
tmux attach -t zett-modal-20k
cat modal_results/zett_20k_controller/status.json
tail -n 40 modal_results/zett_20k_controller/controller.log
```

## Current handoff

Status: inputs in all three profiles passed the same audit (56 files,
8,868,520,307 bytes; manifest SHA-256
`53d2968d9a1e1a7917b7f05716bca77acfcf9b61a06a08d35fa2bb3b4bbefb8c`).
`projects` stopped cleanly at step 6,120. Its latest full checkpoint passed an
independent hash audit; all 500-step grid saves through 6,000 are present, with
full restarts at 2,000, 4,000, and 6,000. Metered compute was $28.47, below
the $29 guard. The first directory download for the handoff to `grading`
stalled with a wrong LM checksum; no checkpoint was uploaded. Four files are
verified in `/tmp/zett-step-006120-salvage`, and a bounded SDK download is
repairing the LM file. Once the entire checkpoint verifies, upload it to
`grading`, audit there, and restart the controller with `--resume-grading`.
The artifact helper now downloads each file to a temporary path and checks its
SHA-256 before publishing it locally. The one-off `zett-handoff-finish` tmux
session waits for the LM repair and optimizer upload, verifies the local full
checkpoint, uploads the remaining files, audits the remote copy, and restarts
the controller. Its log is
`modal_results/zett_20k_controller/handoff_finish.log`.
