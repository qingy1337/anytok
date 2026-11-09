"""Verify the byte-coverage and exact round-trip gates reject lossy models."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from tokenizer_speed import lossless_audit, unicode_fixtures, run


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

    def test_wordpiece_audit_records_original_cutoff_without_rewriting(self):
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders
        vocab = {c: i for i, c in enumerate(pre_tokenizers.ByteLevel.alphabet())}
        vocab['<|unk|>'] = len(vocab)
        tokenizer = Tokenizer(models.WordPiece(vocab=vocab, unk_token='<|unk|>', continuing_subword_prefix='', max_input_chars_per_word=10))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        tokenizer.decoder = decoders.ByteLevel()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'tokenizer.json'; tokenizer.save(str(path))
            with self.assertRaisesRegex(ValueError, 'round-trip'): lossless_audit(path)
            tokenizer.model.max_input_chars_per_word = 1_000_000
            tokenizer.save(str(path))
            original = path.read_bytes()
            report = lossless_audit(path)
            self.assertEqual(report['roundtrip_failures'], 0)
            self.assertEqual(report['wordpiece_max_input_chars_per_word'], 1_000_000)
            self.assertEqual(path.read_bytes(), original)

    def test_training_wrapper_preserves_reference_file_and_manifest_exactly(self):
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders
        vocab = {c: i for i, c in enumerate(pre_tokenizers.ByteLevel.alphabet())}
        vocab['<|unk|>'] = len(vocab)
        tokenizer = Tokenizer(models.WordPiece(vocab=vocab, unk_token='<|unk|>',
            continuing_subword_prefix='', max_input_chars_per_word=1_000_000))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        tokenizer.decoder = decoders.ByteLevel()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            config, recipe = folder/'config.json', folder/'recipe.json'
            config.write_text(json.dumps({'algorithm': 'wordpiece'})); recipe.write_text('{}')
            model, manifest = folder/'tokenizer.json', folder/'manifest.json'
            tokenizer.save(str(model)); manifest.write_text('{"original": true}\n')
            original_model, original_manifest = model.read_bytes(), manifest.read_bytes()
            with patch('tokenizer_pool.train', return_value={'original': True}):
                result = run(config, recipe, folder)
            self.assertEqual(model.read_bytes(), original_model)
            self.assertEqual(manifest.read_bytes(), original_manifest)
            self.assertEqual(result['manifest'], {'original': True})

    def test_whitespace_stripping_specials_and_truncation_are_rejected(self):
        from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers, decoders
        vocab = {c: i for i, c in enumerate(pre_tokenizers.ByteLevel.alphabet())}
        tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[]))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        tokenizer.decoder = decoders.ByteLevel()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'tokenizer.json'
            tokenizer.enable_truncation(10); tokenizer.save(str(path))
            with self.assertRaisesRegex(ValueError, 'truncate'): lossless_audit(path)
            tokenizer.no_truncation()
            tokenizer.add_special_tokens([AddedToken('<|endoftext|>', lstrip=True, special=True)])
            tokenizer.save(str(path))
            with self.assertRaisesRegex(ValueError, 'strip'): lossless_audit(path)


if __name__ == '__main__': unittest.main()
