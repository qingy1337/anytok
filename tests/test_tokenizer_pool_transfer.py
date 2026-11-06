"""Transfers must close listing streams and tolerate transient read failures."""
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
transfer = importlib.import_module("modal_tokenizer_pool_transfer")


class TransferTest(unittest.TestCase):
    def test_listing_closes_before_manifest_reads(self):
        config = {"index": 0, "algorithm": "bpe"}
        manifest = {"status": "complete", "config": config,
                    "tokenizer_bytes": 3, "tokenizer_sha256": "expected"}
        class Volume:
            listing_open = False
            def iterdir(self, *args, **kwargs):
                self.listing_open = True
                try:
                    yield SimpleNamespace(path="pool/tokenizers/0000/manifest.json")
                finally:
                    self.listing_open = False
            def read_file(self, path):
                if self.listing_open:
                    raise AssertionError("Listing stream still open during nested read")
                yield json.dumps(manifest).encode()
        items = transfer.gather_items(Volume(), {"configs": [config]}, "pool")
        self.assertEqual(len(items), 2)
        self.assertEqual(items[1]["sha256"], "expected")
        self.assertTrue(items[1]["source_path"].endswith("/tokenizer.json"))

    def test_transient_read_retries_but_invalid_data_does_not(self):
        with patch.object(transfer.time, "sleep"), patch.object(
                transfer, "sha256", side_effect=[transfer.StreamTerminatedError("lost"), "ok"]) as operation:
            self.assertEqual(transfer.retry_read(operation), "ok")
            self.assertEqual(operation.call_count, 2)
        with patch.object(transfer.time, "sleep"), patch.object(
                transfer, "sha256", side_effect=transfer.StreamTerminatedError("lost")) as operation:
            with self.assertRaises(transfer.StreamTerminatedError):
                transfer.retry_read(operation)
            self.assertEqual(operation.call_count, 4)
        with patch.object(transfer, "sha256", side_effect=ValueError("bad data")) as operation:
            with self.assertRaises(ValueError):
                transfer.retry_read(operation)
            self.assertEqual(operation.call_count, 1)


if __name__ == "__main__":
    unittest.main()
