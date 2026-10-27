"""Train matched 200 MB cross-word BPE tokenizers with stochastic merge choices.

Build the patched wheel described in dashboard/README.md, then run with the
group-cron Modal profile. The zero-noise model checks the patched trainer.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import time

import modal


PROJECT = Path(__file__).resolve().parent.parent
ROOT = Path("/corpus")
WHEEL_NAME = "tokenizers-0.22.2-cp39-abi3-manylinux_2_34_x86_64.whl"
WHEEL_PATH = PROJECT / "build" / WHEEL_NAME
VOCAB_SIZE = 150_000
NOISE_CANDIDATES = 32

image = (modal.Image.debian_slim(python_version="3.11")
         .add_local_file(str(WHEEL_PATH), f"/tmp/{WHEEL_NAME}", copy=True)
         .run_commands(f"pip install --no-deps /tmp/{WHEEL_NAME}"))
app = modal.App("anytok-merge-noise")
volume = modal.Volume.from_name("anytok-corpus")


def iter_batches(recipe: dict, quota: dict[str, int], stats: dict):
    for source in recipe["buckets"]:
        name = source["name"]
        remaining = quota[name]
        used = documents = 0
        batch = []
        batch_bytes = 0
        with gzip.open(Path(source["source_path"]) / "train.jsonl.gz", "rt",
                       encoding="utf-8") as handle:
            for line in handle:
                if remaining <= 0:
                    break
                content = json.loads(line)["text"]
                raw = content.encode("utf-8")
                if len(raw) > remaining:
                    content = raw[:remaining].decode("utf-8", errors="ignore")
                    raw = content.encode("utf-8")
                if not raw:
                    break
                batch.append(content)
                batch_bytes += len(raw)
                used += len(raw)
                remaining -= len(raw)
                documents += 1
                if batch_bytes >= 2_000_000 or len(batch) >= 128:
                    yield batch
                    batch = []
                    batch_bytes = 0
        if batch:
            yield batch
        if remaining > 3:
            raise RuntimeError(f"{name} ended {remaining} bytes short")
        stats[name] = {"quota_bytes": quota[name], "used_bytes": used, "documents": documents}


@app.function(image=image, volumes={str(ROOT): volume}, cpu=8, memory=32768,
              timeout=86_400, max_containers=3)
def train_one(plan: dict, recipe: dict, rate: int, seed: int) -> dict:
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

    quota = plan["baseline"]["bucket_bytes"]
    settings = {"vocab_size": VOCAB_SIZE, "min_frequency": 2, "max_token_length": 64,
                "pretokenizer": "line256", "noise_per_million": rate,
                "noise_seed": seed, "noise_candidates": NOISE_CANDIDATES,
                "tokenizers_version": "0.22.2+anytok-noise-v1"}
    payload = {"recipe": plan["source_recipe"], "quota": quota, "settings": settings}
    job_id = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    destination = ROOT / "tokenizers/merge-noise-v1" / job_id / "base"
    tokenizer_path = destination / "tokenizer.json"
    manifest_path = destination / "manifest.json"
    if tokenizer_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["job_id"] != job_id:
            raise RuntimeError("Existing model has a different job ID")
        return manifest

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(r"[^\n]{1,256}\n?|\n"), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE, min_frequency=2, max_token_length=64,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|endoftext|>"], show_progress=False,
        noise_per_million=rate, noise_seed=seed,
    )
    stats = {}
    start = time.monotonic()
    tokenizer.train_from_iterator(iter_batches(recipe, quota, stats), trainer=trainer)
    elapsed = time.monotonic() - start
    if tokenizer.get_vocab_size() != VOCAB_SIZE:
        raise RuntimeError("Tokenizer did not reach the target vocabulary size")
    destination.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_path))
    manifest = {"job_id": job_id, "settings": settings, "source_stats": stats,
                "total_used_bytes": sum(item["used_bytes"] for item in stats.values()),
                "vocab_size_actual": tokenizer.get_vocab_size(),
                "train_seconds": round(elapsed, 2), "model_path": str(tokenizer_path)}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    volume.commit()
    print(f"Rate {rate}/1M seed {seed}: trained in {elapsed:.1f}s", flush=True)
    return manifest


@app.local_entrypoint()
def main(rates: str = "0,1000,10000,50000", seeds: str = "1,2",
         output: str = "results/merge_noise_runs_200m.json") -> None:
    if __import__("os").environ.get("MODAL_PROFILE") != "group-cron":
        raise RuntimeError("Set MODAL_PROFILE=group-cron")
    plan = json.loads((PROJECT / "results/pretoken_pilot_mixtures.json").read_text())
    recipe = json.loads((PROJECT / "data/balanced-full-v3.json").read_text())
    if plan["source_recipe"] != recipe["name"] or plan["total_train_bytes_per_mixture"] != 200_000_000:
        raise ValueError("Expected the matched 200 MB corpus plan")
    rate_values = [int(value) for value in rates.split(",")]
    seed_values = [int(value) for value in seeds.split(",")]
    if not rate_values or any(not 0 <= rate <= 1_000_000 for rate in rate_values):
        raise ValueError("Noise rates must be integers from 0 to 1000000")
    if not seed_values or any(seed < 0 for seed in seed_values):
        raise ValueError("Seeds must be nonnegative integers")
    jobs = [(plan, recipe, 0, 1)] if 0 in rate_values else []
    jobs += [(plan, recipe, rate, seed) for rate in rate_values if rate > 0 for seed in seed_values]
    results = list(train_one.starmap(jobs, order_outputs=False))
    path = PROJECT / output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
