# ZeTT 20k on Modal

Objective: train the joint SmolLM2/ZeTT model from the pinned warmup to update
20,000 and retain LM plus hypernetwork weights every 500 updates for later
evaluation curves. Keep full restart checkpoints every 2,000 updates.

## Canonical inputs and code

- Launcher: `scripts/modal_zett_20k.py`; direct Modal-to-Hugging-Face transfer:
  `scripts/modal_zett_20k_hf.py`.
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
checkpoint. The next segment resumes the latest full checkpoint. After a
handoff is audited in the next profile, the old source's terminal checkpoint
is compacted or removed if off-grid. The trainer's 23-bucket heldout
check is a run sanity metric; the FineWeb-Edu tokenizer-transfer evaluation is
a separate paired sweep over these saved checkpoints.

The handoff repository is the private model
`qingy2024/anytok-zett-smollm2-20k`. Its `checkpoints/step-NNNNNN/` directories
hold checkpoint files and hash manifests; `run/` holds the resume metadata.
`latest.json` is published last, after all files, and tells the next Modal
profile which full checkpoint to download. Modal workers upload and download
directly. Earlier 500-step grid checkpoints remain in their source Volumes for
later archival. Filling the whole grid is outside the current training path;
ten full and thirty weights-only checkpoints would be about 494 GB.

## Commands

Run from `/home/zli/anytok` with the local Modal CLI environment. The upload
and download Functions use the existing `huggingface-secret` in each profile;
both were checked to authenticate as `qingy2024`. Neither Function requests a
GPU, CPU, or memory allocation explicitly.

```bash
MODAL_PROFILE=projects modal run --detach scripts/modal_zett_20k_hf.py --action upload
MODAL_PROFILE=grading modal run --detach scripts/modal_zett_20k_hf.py --action download
MODAL_PROFILE=grading modal run scripts/modal_zett_20k.py --action audit-checkpoint --step 6120
python scripts/modal_zett_20k_controller.py --resume-grading --plan
tmux new-session -d -s zett-modal-20k 'bash scripts/run_modal_zett_20k_controller.sh --resume-grading'
cat modal_results/zett_20k_controller/status.json
tail -n 40 modal_results/zett_20k_controller/controller.log
```

The controller checks the $29 metered-compute ceiling in each profile before
starting another B300 segment. It performs the later `grading` to
`test-joining` handoff through the same Hugging Face repository, then publishes
the final step 20,000 restart checkpoint there. The 500-step grid collection
remains pending after training.

## Current handoff

Status: inputs in all three profiles passed the same audit (56 files,
8,868,520,307 bytes; manifest SHA-256
`53d2968d9a1e1a7917b7f05716bca77acfcf9b61a06a08d35fa2bb3b4bbefb8c`).
`projects` stopped cleanly at step 6,120. Its latest full checkpoint passed an
independent hash audit; all 500-step grid saves through 6,000 are present, with
full restarts at 2,000, 4,000, and 6,000. Metered compute was $28.47, below
the $29 guard. The local relay failed and was stopped. Step 6,120 is now in
the private Hugging Face model repo, with `latest.json` published after all
checkpoint and metadata files. A CPU-only `grading` Modal Function is
downloading directly into its Volume. After its full hash audit, the
`zett-hf-handoff-finish` tmux session runs an independent `grading` audit and
starts the controller with `--resume-grading`. Logs are under
`modal_results/zett_20k_controller/hf_*.log`. No B300 is running during this
transfer.
