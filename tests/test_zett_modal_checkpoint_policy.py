"""Safety checks for converting the 500-step archive during a live run."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import modal_zett_20k as launcher
from scripts import modal_zett_20k_artifacts as artifacts
from scripts import modal_zett_20k_hf as hf_relay


def make_checkpoint(run: Path, step: int):
    checkpoint = run / f"step-{step:06d}"
    names = (*launcher.WEIGHT_FILES, "optimizer.pt", "state.pt")
    files = {}
    for name in names:
        path = checkpoint / name
        path.parent.mkdir(parents=True, exist_ok=True)
        value = f"{step}:{name}".encode()
        path.write_bytes(value)
        files[name] = {"bytes": len(value),
                       "sha256": hashlib.sha256(value).hexdigest()}
    (checkpoint / "manifest.json").write_text(json.dumps({
        "step": step, "kind": "full_restart_checkpoint", "files": files,
    }))
    return checkpoint


class CheckpointPolicyTest(unittest.TestCase):
    def test_keep_milestones_and_latest_then_clean_handoff_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            mount = Path(temporary)
            run = mount / "run"
            for step in (500, 1000, 1500, 2000, 2500, 2600):
                make_checkpoint(run, step)
            with patch.object(launcher, "MOUNT", mount):
                self.assertEqual(launcher.compact_grid_checkpoints(run),
                                 [500, 1000, 1500, 2500])
                for step in (500, 1000, 1500, 2500):
                    result = launcher.verify_checkpoint_files(
                        step, "weights_only_checkpoint")
                    self.assertEqual(result["files"], 3)
                    self.assertFalse((run / f"step-{step:06d}" / "optimizer.pt").exists())
                for step in (2000, 2600):
                    launcher.verify_checkpoint_files(step, "full_restart_checkpoint")
                self.assertEqual(launcher.compact_grid_checkpoints(run, False), [2600])
                self.assertFalse((run / "step-002600").exists())
                launcher.verify_checkpoint_files(2000, "full_restart_checkpoint")

    def test_bad_weights_never_delete_restart_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            old = make_checkpoint(run, 500)
            make_checkpoint(run, 2000)
            (old / "lm.pt").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                launcher.compact_grid_checkpoints(run)
            self.assertTrue((old / "optimizer.pt").exists())
            self.assertTrue((old / "state.pt").exists())

    def test_collection_conversion_preserves_weight_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            checkpoint = make_checkpoint(run, 500)
            before = artifacts.verify_checkpoint(checkpoint, "full_restart_checkpoint")
            artifacts.compact_local_checkpoint(checkpoint, before)
            after = artifacts.verify_checkpoint(checkpoint, "weights_only_checkpoint")
            for name in launcher.WEIGHT_FILES:
                self.assertEqual(before["files"][name], after["files"][name])
            self.assertFalse((checkpoint / "optimizer.pt").exists())

    def test_transfer_downloads_files_under_the_run_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = make_checkpoint(Path(temporary) / "run", 500)
            saved = json.loads((source / "manifest.json").read_text())
            uploaded = []

            def fake_get(profile, remote, destination):
                self.assertEqual(remote, "run/step-000500/manifest.json")
                Path(destination).write_text(json.dumps(saved))

            def fake_fetch(profile, remote, destination, expected):
                name = remote.removeprefix("run/step-000500/")
                self.assertEqual(expected, saved["files"][name])
                Path(destination).parent.mkdir(parents=True, exist_ok=True)
                Path(destination).write_bytes((source / name).read_bytes())

            def fake_put(profile, local, remote):
                self.assertEqual(remote, "run/step-000500")
                artifacts.verify_checkpoint(local, "weights_only_checkpoint")
                uploaded.append(remote)

            with (patch.object(artifacts, "get", fake_get),
                  patch.object(artifacts, "fetch_file", fake_fetch),
                  patch.object(artifacts, "put", fake_put)):
                artifacts.transfer_checkpoint("projects", "grading", 500,
                                              weights_only=True)
            self.assertEqual(uploaded, ["run/step-000500"])

    def test_hf_relay_requires_a_complete_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = make_checkpoint(Path(temporary) / "run", 2000)
            manifest, digest = hf_relay.verify_checkpoint(checkpoint, 2000)
            self.assertEqual(manifest["kind"], "full_restart_checkpoint")
            self.assertEqual(len(digest), 64)
            manifest["files"].pop("optimizer.pt")
            (checkpoint / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                hf_relay.verify_checkpoint(checkpoint, 2000)

    def test_resume_rebinds_only_profile_volume_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            config = {"seed": 42,
                      "warmup_checkpoint": "/__modal/volumes/vo-source/input/warmup"}
            (run / "run_config.json").write_text(json.dumps(config))
            (run / "events.jsonl").write_text("")
            target = "/__modal/volumes/vo-target/input/warmup"
            self.assertTrue(launcher.rebind_warmup_volume_path(run, target))
            self.assertEqual(json.loads((run / "run_config.json").read_text()),
                             {"seed": 42, "warmup_checkpoint": target})
            self.assertFalse(launcher.rebind_warmup_volume_path(run, target))
            self.assertEqual(len((run / "events.jsonl").read_text().splitlines()), 1)
            with self.assertRaises(ValueError):
                launcher.rebind_warmup_volume_path(run, "/wrong/input/warmup")


if __name__ == "__main__":
    unittest.main()
