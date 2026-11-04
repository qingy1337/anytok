# ZeTT 20k on Modal

Objective: train the joint SmolLM2/ZeTT model from the pinned warmup to update
20,000 and retain a full restart checkpoint at every 500 updates for later
evaluation curves.

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
volume. An interrupted segment also saves an off-grid restart checkpoint. The
next segment resumes the latest full checkpoint. All grid checkpoints remain
on their source profiles until collection. The final copy of all full
checkpoints belongs in `test-joining`'s volume.

## Commands

Run from `/home/zli/anytok` with the local Modal CLI environment:

```bash
MODAL_PROFILE=projects modal run scripts/modal_zett_20k.py --action plan
python scripts/modal_zett_20k_artifacts.py upload projects
MODAL_PROFILE=projects modal run scripts/modal_zett_20k.py --action audit
MODAL_PROFILE=projects modal run scripts/modal_zett_20k.py --action smoke
MODAL_PROFILE=projects modal run scripts/modal_zett_20k.py --action train
python scripts/modal_zett_20k_artifacts.py status projects
```

After the first segment stops, upload and audit inputs in `grading`, then run
`python scripts/modal_zett_20k_artifacts.py handoff projects grading`. Repeat
for `grading` to `test-joining`. Run the train command in each destination
profile. Once update 20,000 is committed in `test-joining`, run
`python scripts/modal_zett_20k_artifacts.py collect`. `collect` transfers one
checkpoint at a time and needs about 25 GiB local free space; it verifies each
download against its checkpoint manifest.

## Current handoff

Status: inputs uploading to `projects`; no training job is running.

Next: audit the uploaded inputs, check billing, run the B300 smoke from the
committed snapshot, and benchmark the first segment before using the rest of
the credit. Inspect only the compact status, the latest checkpoint manifest,
and bounded training log lines. The existing local 15k run and its checkpoints
are separate and should not be overwritten.
