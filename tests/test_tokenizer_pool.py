"""Check the saved sampling distribution and patched trainer integrations."""
import contextlib
import gzip
import importlib.util
import io
import json
from pathlib import Path
import random
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pool", ROOT / "scripts/tokenizer_pool.py")
pool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pool)


class PoolPlanTest(unittest.TestCase):
    def test_saved_plan_matches_sampler_and_all_quotas_are_feasible(self):
        recipe = json.loads((ROOT / "data/balanced-full-v3.json").read_text())
        saved = json.loads((ROOT / "plans/tokenizer-pool-1000-v1.json").read_text())
        plan = pool.make_plan(recipe)
        self.assertEqual(saved["configs"], plan["configs"])
        self.assertEqual(len({c["config_sha256"] for c in plan["configs"]}), 1000)
        sources = {s["name"]: s for s in recipe["buckets"]}
        for config in plan["configs"]:
            self.assertEqual(sum(config["bucket_bytes"].values()), 200_000_000)
            self.assertEqual(sum(config["domain_bytes"].values()), 200_000_000)
            self.assertIn(config["vocab_size"], range(32000, 128001, 1000))
            if config["algorithm"] == "bpe":
                self.assertTrue(0 <= config["noise_per_million"] <= 750000)
            else:
                self.assertIsNone(config["noise_per_million"])
            for name, amount in config["bucket_bytes"].items():
                self.assertLessEqual(amount, sources[name]["train_available_bytes"])
            for domain, names in pool.DOMAINS.items():
                self.assertEqual(sum(config["bucket_bytes"][n] for n in names),
                                 config["domain_bytes"][domain])
        pilot = [plan["configs"][i] for i in plan["pilot_indices"]]
        self.assertEqual({(c["algorithm"], c["pretokenizer"]) for c in pilot},
                         {(a, p) for a in pool.ALGORITHMS for p in pool.PRETOKENIZERS})

    def test_remainder_allocation_preserves_total(self):
        self.assertEqual(pool.allocate(10, [1, 1, 1]), [4, 3, 3])
        self.assertEqual(pool.allocate(0, [1, 2, 3]), [0, 0, 0])


class TrainerTest(unittest.TestCase):
    def test_all_nine_algorithm_pretokenizer_combinations(self):
        from tokenizers import Tokenizer, trainers
        patched = json.loads(trainers.BpeTrainer().__getstate__())["BpeTrainer"]
        if "noise_per_million" not in patched:
            self.skipTest("Requires AnyTok's patched wheel (use isolated PYTHONPATH)")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rng = random.Random(4)
            words = ["".join(rng.choices("abcdefghijklmno", k=6)) for _ in range(500)]
            docs = [" ".join(rng.choices(words, k=100)) + " 中文 العربية 🙂\n" for _ in range(60)]
            docs[0] += " Literal <UNK> occurs once."
            total = sum(len(text.encode("utf-8")) for text in docs)
            with gzip.open(root / "train.jsonl.gz", "wt", encoding="utf-8") as stream:
                for text in docs:
                    stream.write(json.dumps({"text": text}) + "\n")
            recipe = {"buckets": [{"name": "toy", "source_path": str(root)}]}
            for algorithm in pool.ALGORITHMS:
                for pretokenizer in pool.PRETOKENIZERS:
                    with self.subTest(algorithm=algorithm, pretokenizer=pretokenizer):
                        config = {"index": 0, "algorithm": algorithm, "pretokenizer": pretokenizer,
                                  "vocab_size": 512, "noise_per_million": 750000,
                                  "seed": 42, "bucket_bytes": {"toy": total}}
                        destination = root / algorithm / pretokenizer
                        with contextlib.redirect_stdout(io.StringIO()):
                            manifest = pool.train(config, recipe, destination)
                        self.assertEqual(manifest["total_used_bytes"], total)
                        self.assertEqual(manifest["vocab_size_actual"], 512)
                        tokenizer = Tokenizer.from_file(str(destination / "tokenizer.json"))
                        self.assertEqual(tokenizer.decode(tokenizer.encode(docs[0]).ids), docs[0])


if __name__ == "__main__":
    unittest.main()
