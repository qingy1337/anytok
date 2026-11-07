"""Saved-tokenizer integrity, native token IDs, priors and resumable sampling."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from tokenizers import Tokenizer, models

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/anytok_pool_20k'))
from hf_hypernet.pool_sampling import TokenizerPool


class PoolTest(unittest.TestCase):
    def fixture(self, directory, unigram=False):
        config = [{'index': i, 'vocab_size': 3} for i in range(1000)]
        model = (models.Unigram([('<|unk|>', 0.), ('<|endoftext|>', -1.), ('a', -2.)], unk_id=0)
                 if unigram else models.WordPiece({'<|unk|>': 0, '<|endoftext|>': 1, 'a': 2}, unk_token='<|unk|>'))
        raw = Tokenizer(model).to_str().encode()
        entries = [{'status':'complete','config':c,'vocab_size_actual':3,
                    'tokenizer_sha256':hashlib.sha256(raw).hexdigest()} for c in config]
        (directory/'plan.json').write_text(json.dumps({'plan_sha256':'p','configs':config}))
        (directory/'full-results.json').write_text(json.dumps({'plan_sha256':'p','completed':1000,'tokenizers':entries}))
        model_dir = directory/'tokenizers/0000'; model_dir.mkdir(parents=True)
        (model_dir/'tokenizer.json').write_bytes(raw)

    def test_wordpiece_native_ids_are_preserved_and_priors_are_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.fixture(root)
            pool=TokenizerPool(root); tokenizer,vocab=pool.load(0)
            self.assertEqual(tokenizer.token_to_id('<|endoftext|>'),1)
            self.assertEqual(vocab,[('<|unk|>',0.),('<|endoftext|>',0.),('a',0.)])

    def test_unigram_priors_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.fixture(root,unigram=True)
            _,vocab=TokenizerPool(root).load(0)
            self.assertEqual([score for _,score in vocab],[0.,-1.,-2.])

    def test_corrupted_or_incomplete_pool_fails_before_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.fixture(root)
            pool=TokenizerPool(root)
            (root/'tokenizers/0000/tokenizer.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'checksum'):
                pool.load(0)
            path=root/'full-results.json'; value=json.loads(path.read_text())
            value['tokenizers'].pop();path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'1000'):
                TokenizerPool(root)

    def test_uniform_draws_cover_all_ids_and_continue_exactly_after_resume(self):
        rng=np.random.default_rng(42)
        draws=rng.integers(1000,size=20000)
        self.assertEqual(len(set(draws)),1000)
        state=rng.bit_generator.state
        expected=rng.integers(1000,size=100)
        resumed=np.random.default_rng();resumed.bit_generator.state=state
        np.testing.assert_array_equal(expected,resumed.integers(1000,size=100))


if __name__=='__main__':
    unittest.main()
