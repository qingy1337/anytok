"""Finish from the audited grading checkpoint within its remaining credit."""
import fcntl
import traceback

import modal_zett_20k_controller as c

c.SPEND_CEILING = 29.95
with (c.STATUS_DIR / "controller.lock").open("a+") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        c.audit_checkpoint("grading")
        c.run_segment("grading", 460)
        if c.latest("grading") < c.FINAL_STEP:
            raise RuntimeError("Final budget-safe segment ended before step 20000")
        c.audit_checkpoint("grading")
        c.emit("final_publish_start", final_step=c.FINAL_STEP, repo=c.HF_REPO)
        c.cli("grading", "run", "--detach", c.HF_LAUNCHER, "--action", "upload")
        c.emit("training_complete", final_step=c.FINAL_STEP, final_profile="grading",
               volume=c.VOLUME, repo=c.HF_REPO, full_grid_collection_pending=True)
    except Exception as exc:
        c.emit("failed", error=str(exc), traceback=traceback.format_exc()[-4000:])
        raise
