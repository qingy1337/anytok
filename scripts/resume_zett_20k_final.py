"""Resume the final segment within each profile's existing compute guard."""
import fcntl
import traceback

import modal_zett_20k_controller as c

with (c.STATUS_DIR / "controller.lock").open("a+") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        c.audit_checkpoint("test-joining")
        c.run_segment("test-joining", 3000)
        profile = "test-joining"
        if c.latest(profile) < c.FINAL_STEP:
            c.handoff_and_audit(profile, "grading")
            profile = "grading"
            c.run_segment(profile, 650)
        if c.latest(profile) < c.FINAL_STEP:
            raise RuntimeError("Remaining credit-safe segments ended before step 20000")
        c.audit_checkpoint(profile)
        c.emit("final_publish_start", final_step=c.FINAL_STEP, repo=c.HF_REPO)
        c.cli(profile, "run", "--detach", c.HF_LAUNCHER, "--action", "upload")
        c.emit("training_complete", final_step=c.FINAL_STEP, final_profile=profile,
               volume=c.VOLUME, repo=c.HF_REPO, full_grid_collection_pending=True)
    except Exception as exc:
        c.emit("failed", error=str(exc), traceback=traceback.format_exc()[-4000:])
        raise
