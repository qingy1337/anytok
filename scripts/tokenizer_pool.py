"""Sample AnyTok's reproducible tokenizer pool and train one saved configuration.

Sampling uses equal algorithm/pretokenizer probabilities, integer vocabulary
sizes 32k..128k in 1k steps, BPE noise uniform in integer millionths 0..750000,
and Dirichlet(1,...,1) domain weights. Corpus quotas are raw UTF-8 bytes.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import time

DOMAINS = {
    "web": tuple(f"web-{lang}" for lang in ("en", "es", "fr", "ru", "zh", "ja", "ar", "hi")),
    "wikipedia": tuple(f"wiki-{lang}" for lang in ("en", "es", "fr", "ru", "zh", "ja", "ar", "hi")),
    "code": ("code",), "papers": ("papers-en",), "math": ("math-en",),
    "qa": ("qa-en",), "books": ("books-en",), "medical": ("medical-en",),
    "legal": ("legal-en",),
}
ALGORITHMS = ("bpe", "wordpiece", "unigram")
PRETOKENIZERS = ("regex", "line256", "no-regex")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def allocate(total, weights):
    norm = sum(weights)
    raw = [total * weight / norm for weight in weights]
    quotas = [math.floor(value) for value in raw]
    order = sorted(range(len(raw)), key=lambda i: raw[i] - quotas[i], reverse=True)
    for i in order[:total - sum(quotas)]:
        quotas[i] += 1
    assert sum(quotas) == total
    return quotas


def make_plan(recipe, count=1000, seed=20261105, total_bytes=200_000_000):
    sources = {source["name"]: source for source in recipe["buckets"]}
    assert set(sources) == {name for names in DOMAINS.values() for name in names}
    rng = random.Random(seed)
    configs = []
    for index in range(count):
        algorithm = rng.choice(ALGORITHMS)
        pretokenizer = rng.choice(PRETOKENIZERS)
        vocab_size = rng.randrange(32, 129) * 1000
        noise = rng.randint(0, 750_000) if algorithm == "bpe" else None
        tokenizer_seed = rng.randrange(2**63)
        # Normalized independent exponentials are exactly Dirichlet(alpha=1).
        weights = [rng.expovariate(1) for _ in DOMAINS]
        domain_quota = dict(zip(DOMAINS, allocate(total_bytes, weights)))
        quota = {}
        for domain, names in DOMAINS.items():
            quota.update(zip(names, allocate(domain_quota[domain],
                [sources[name]["train_target_bytes"] for name in names])))
        for name, amount in quota.items():
            if amount > sources[name]["train_available_bytes"]:
                raise ValueError(f"Sample {index}: unavailable quota for {name}; do not silently resample")
        config = {
            "index": index, "algorithm": algorithm, "pretokenizer": pretokenizer,
            "vocab_size": vocab_size, "noise_per_million": noise, "seed": tokenizer_seed,
            "domain_bytes": domain_quota, "bucket_bytes": quota,
        }
        config["config_sha256"] = digest(config)
        configs.append(config)
    pilot = []
    for algorithm in ALGORITHMS:
        for pretokenizer in PRETOKENIZERS:
            candidates = [c for c in configs if c["algorithm"] == algorithm and
                          c["pretokenizer"] == pretokenizer]
            if not candidates:
                continue
            typical = min(candidates, key=lambda c: abs(c["vocab_size"] - 80_000) / 96_000 +
                          abs((c["noise_per_million"] or 375_000) - 375_000) / 750_000)
            pilot.append(typical["index"])
            if algorithm == "bpe":
                extreme = max(candidates, key=lambda c: c["noise_per_million"])
                pilot.append(extreme["index"])
    plan = {
        "schema_version": 1, "seed": seed, "count": count,
        "total_bytes_per_tokenizer": total_bytes, "recipe": recipe,
        "recipe_sha256": digest(recipe), "tokenizers_wheel_sha256": None,
        "distribution": {
            "algorithm": "uniform categorical: bpe, wordpiece, unigram",
            "pretokenizer": "uniform categorical: regex, line256, no-regex",
            "vocab_size": "uniform discrete: 32000..128000 inclusive, step 1000",
            "bpe_noise_per_million": "uniform integer: 0..750000 inclusive",
            "domain_weights": "Dirichlet(1,1,1,1,1,1,1,1,1)",
            "within_domain": "proportional to recipe train_target_bytes",
        },
        "pilot_indices": list(dict.fromkeys(pilot)), "configs": configs,
    }
    return plan


def batches(recipe, quota, stats, validation):
    for source in recipe["buckets"]:
        name = source["name"]
        remaining = quota[name]
        used = documents = 0
        sha = hashlib.sha256()
        batch, size = [], 0
        if remaining:
            with gzip.open(Path(source["source_path"]) / "train.jsonl.gz", "rt", encoding="utf-8") as stream:
                for line in stream:
                    if remaining <= 3:
                        break
                    text = json.loads(line)["text"]
                    raw = text.encode("utf-8")
                    if len(raw) > remaining:
                        text = raw[:remaining].decode("utf-8", errors="ignore")
                        raw = text.encode("utf-8")
                    if not raw:
                        continue
                    if name not in validation:
                        validation[name] = text[:2048]
                    # Include document boundaries in the reproducibility hash.
                    sha.update(len(raw).to_bytes(8, "little"))
                    sha.update(raw)
                    batch.append(text)
                    size += len(raw)
                    used += len(raw)
                    remaining -= len(raw)
                    documents += 1
                    if size >= 2_000_000 or len(batch) >= 128:
                        yield batch
                        batch, size = [], 0
        if batch:
            yield batch
        if remaining > 3:
            raise RuntimeError(f"{name} ended with {remaining} bytes unfilled")
        stats[name] = {"quota_bytes": quota[name], "used_bytes": used,
                       "documents": documents, "documents_sha256": sha.hexdigest()}
    print(json.dumps({"stage": "corpus_consumed", "used_bytes": sum(s["used_bytes"] for s in stats.values())}), flush=True)


def train(config, recipe, destination):
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

    algorithm = config["algorithm"]
    common = dict(vocab_size=config["vocab_size"],
                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    specials = ["<|endoftext|>"] if algorithm == "bpe" else ["<|unk|>", "<|endoftext|>"]
    common["special_tokens"] = specials
    if algorithm == "bpe":
        tokenizer = Tokenizer(models.BPE())
        trainer = trainers.BpeTrainer(**common, min_frequency=2, max_token_length=64,
            noise_per_million=config["noise_per_million"], noise_seed=config["seed"], random_from_all=True)
        if json.loads(trainer.__getstate__())["BpeTrainer"].get("noise_per_million") != config["noise_per_million"]:
            raise RuntimeError("Patched noise trainer is not installed")
    elif algorithm == "wordpiece":
        tokenizer = Tokenizer(models.WordPiece(unk_token="<|unk|>", max_input_chars_per_word=1_000_000))
        trainer = trainers.WordPieceTrainer(**common, min_frequency=2, continuing_subword_prefix="")
    else:
        tokenizer = Tokenizer(models.Unigram())
        trainer = trainers.UnigramTrainer(**common, max_piece_length=64, unk_token="<|unk|>")
    if config["pretokenizer"] == "line256":
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(r"[^\n]{1,256}\n?|\n"), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
    else:
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False, use_regex=config["pretokenizer"] == "regex")
    tokenizer.decoder = decoders.ByteLevel()
    stats, validation = {}, {}
    start = time.monotonic()
    cpu_start = time.process_time()
    print(json.dumps({"stage": "training", "index": config["index"],
                      "algorithm": algorithm, "pretokenizer": config["pretokenizer"],
                      "vocab_size": config["vocab_size"], "noise": config["noise_per_million"]}), flush=True)
    tokenizer.train_from_iterator(batches(recipe, config["bucket_bytes"], stats, validation), trainer=trainer)
    train_seconds = time.monotonic() - start
    actual = tokenizer.get_vocab_size()
    if actual != config["vocab_size"]:
        raise RuntimeError(f"Requested {config['vocab_size']} tokens but obtained {actual}")
    validation["synthetic"] = "Hello world!\n中文 العربية हिन्दी 日本語 café 🙂\tdef f(x): return x+1\n"
    checks = {}
    for name, text in validation.items():
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if tokenizer.decode(ids, skip_special_tokens=False) != text:
            raise RuntimeError(f"Round-trip failure for {name}")
        if algorithm != "bpe" and tokenizer.token_to_id("<|unk|>") in ids:
            raise RuntimeError(f"Unknown tokens for {name}")
        checks[name] = {"bytes": len(text.encode("utf-8")), "tokens": len(ids)}
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "tokenizer.json"
    tokenizer.save(str(path))
    usage = resource.getrusage(resource.RUSAGE_SELF)
    elapsed = time.monotonic() - start
    peak_gib = usage.ru_maxrss / (1024 * 1024)
    cpu_seconds = time.process_time() - cpu_start
    # Peak RSS * elapsed overestimates this process's memory integral; allow
    # additional container overhead when estimating the whole pool from pilots.
    estimate = max(cpu_seconds, elapsed * 0.125) / 3600 * 0.04730 + max(peak_gib, 0.125) * elapsed / 3600 * 0.008
    manifest = {
        "status": "complete", "config": config, "source_stats": stats,
        "vocab_size_actual": actual, "train_seconds": train_seconds,
        "elapsed_seconds": elapsed, "cpu_seconds": cpu_seconds, "peak_rss_gib": peak_gib,
        "estimated_process_cost_usd": estimate,
        "total_used_bytes": sum(s["used_bytes"] for s in stats.values()),
        "validation": checks, "tokenizer_bytes": path.stat().st_size,
        "tokenizer_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"stage": "complete", "index": config["index"],
                      "train_seconds": train_seconds, "cpu_seconds": cpu_seconds,
                      "peak_rss_gib": peak_gib, "estimated_process_cost_usd": estimate}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("plans/tokenizer-pool-1000-v1.json"))
    parser.add_argument("--recipe", type=Path, default=Path("data/balanced-full-v3.json"))
    parser.add_argument("--wheel", type=Path, default=Path("build/all-candidate/tokenizers-0.22.2-cp39-abi3-manylinux_2_34_x86_64.whl"))
    parser.add_argument("--train-config", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if args.train_config:
        if not args.destination:
            parser.error("--destination is required with --train-config")
        train(json.loads(args.train_config.read_text()), json.loads(args.recipe.read_text()), args.destination)
        return
    plan = make_plan(json.loads(args.recipe.read_text()))
    plan["tokenizers_wheel_sha256"] = hashlib.sha256(args.wheel.read_bytes()).hexdigest()
    plan["plan_sha256"] = digest(plan)
    if args.output.exists() and json.loads(args.output.read_text()) != plan:
        raise RuntimeError("Refusing to overwrite a different saved pool")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    from collections import Counter
    print(json.dumps({"path": str(args.output), "plan_sha256": plan["plan_sha256"],
        "count": plan["count"], "pilot_indices": plan["pilot_indices"],
        "algorithms": dict(Counter(c["algorithm"] for c in plan["configs"])),
        "pretokenizers": dict(Counter(c["pretokenizer"] for c in plan["configs"]))}, indent=2))


if __name__ == "__main__":
    main()
