"""Verify the byte-coverage and exact round-trip gates reject lossy models."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from tokenizer_speed import lossless_audit, unicode_fixtures


class LosslessAuditTest(unittest.TestCase):
    def test_all_unicode_planes_and_literal_special_strings_roundtrip(self):
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders
        alphabet = pre_tokenizers.ByteLevel.alphabet()
        tokenizer = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        tokenizer.decoder = decoders.ByteLevel()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'tokenizer.json'; tokenizer.save(str(path))
            report = lossless_audit(path)
            self.assertEqual(report['roundtrip_failures'], 0)
            self.assertGreater(report['fixtures'], 256)
            self.assertTrue(report['all_256_byte_symbols_present'])

    def test_normalization_and_missing_byte_coverage_are_rejected(self):
        from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders
        tokenizer = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(pre_tokenizers.ByteLevel.alphabet())}, merges=[]))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        tokenizer.decoder = decoders.ByteLevel()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'tokenizer.json'
            tokenizer.normalizer = normalizers.Lowercase(); tokenizer.save(str(path))
            with self.assertRaisesRegex(ValueError, 'normalize'): lossless_audit(path)
            tokenizer.normalizer = None; tokenizer.save(str(path))
            config = json.loads(path.read_text()); config['model']['vocab'].pop('\u0100')
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'byte alphabet'): lossless_audit(path)


if __name__ == '__main__': unittest.main()
