"""Train comparable byte-level subword tokenizers from the saved Modal corpus.

    MODAL_PROFILE=group-cron modal run scripts/modal_train_tokenizers.py --smoke
    MODAL_PROFILE=group-cron modal run scripts/modal_train_tokenizers.py
    MODAL_PROFILE=group-cron modal run scripts/modal_train_tokenizers.py --algorithm wordpiece
    MODAL_PROFILE=group-cron modal run scripts/modal_train_tokenizers.py --plan results/mixture_code50.json --only C

The five models share all tokenizer settings. Only their per-source raw UTF-8
byte quotas differ. Completed models are stored on the anytok-corpus volume.
The local entrypoint writes sample segmentations if dashboard passages exist.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import time

import modal


ROOT = Path("/corpus")
VOLUME_NAME = "anytok-corpus"
VOCAB_SIZE = 150_000
TOKENIZERS_VERSION = "0.22.2"
TRAINING_SETTINGS = {
    "algorithm": "byte-level BPE",
    "vocab_size": VOCAB_SIZE,
    "min_frequency": 2,
    "max_token_length": 64,
    "pre_tokenizer": "ByteLevel(add_prefix_space=False, use_regex=True)",
    "normalizer": None,
    "special_tokens": ["<|endoftext|>"],
    "tokenizers_version": TOKENIZERS_VERSION,
}


def training_settings(pretokenizer: str, algorithm: str = "bpe") -> dict:
    if pretokenizer not in {"regex", "no-regex", "line-chunks"}:
        raise ValueError("Pretokenizer must be regex, no-regex, or line-chunks")
    if algorithm not in {"bpe", "wordpiece", "unigram"}:
        raise ValueError("Algorithm must be bpe, wordpiece, or unigram")
    settings = TRAINING_SETTINGS.copy()
    if algorithm != "bpe":
        settings["algorithm"] = algorithm
        settings["special_tokens"] = ["<|unk|>", "<|endoftext|>"]
        settings["unk_token"] = "<|unk|>"
        if algorithm == "wordpiece":
            settings["continuing_subword_prefix"] = ""
            settings["max_input_chars_per_word"] = 1_000_000
            settings.pop("max_token_length")  # WordPieceTrainer has no length limit.
        else:
            settings.pop("min_frequency")  # UnigramTrainer has no frequency threshold.
    settings["pre_tokenizer"] = {
        "regex": "ByteLevel(add_prefix_space=False, use_regex=True)",
        "no-regex": "ByteLevel(add_prefix_space=False, use_regex=False)",
        "line-chunks": "Split(lines, max 256 characters) -> ByteLevel(add_prefix_space=False, use_regex=False)",
    }[pretokenizer]
    return settings

app = modal.App("anytok-train-tokenizers")
volume = modal.Volume.from_name(VOLUME_NAME)
image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    f"tokenizers=={TOKENIZERS_VERSION}"
)


def run_id(plan: dict, quota: dict[str, int], smoke: bool, settings: dict) -> str:
    payload = {"recipe": plan["source_recipe"], "quota": quota,
               "settings": settings, "smoke": smoke}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def iter_quota_batches(recipe: dict, quota: dict[str, int], stats: dict):
    """Stream source documents up to each raw-text byte quota in small batches."""
    for source in recipe["buckets"]:
        name = source["name"]
        remaining = quota[name]
        if remaining > source["train_available_bytes"]:
            raise ValueError(f"Quota exceeds available training text for {name}")
        used = docs = 0
        batch = []
        batch_bytes = 0
        path = Path(source["source_path"]) / "train.jsonl.gz"
        with gzip.open(path, "rt", encoding="utf-8") as handle:
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
                docs += 1
                if batch_bytes >= 2_000_000 or len(batch) >= 128:
                    yield batch
                    batch, batch_bytes = [], 0
        if batch:
            yield batch
        if remaining > 3:
            raise RuntimeError(f"Source {name} ended with {remaining:,} bytes unfilled")
        stats[name] = {"quota_bytes": quota[name], "used_bytes": used, "documents": docs}
        print(f"{name}: {used:,}/{quota[name]:,} bytes ({docs:,} documents)", flush=True)


def segmentation(tokenizer, sample: dict) -> dict:
    encoding = tokenizer.encode(sample["text"], add_special_tokens=False)
    pieces = []
    for token_id, token in zip(encoding.ids, encoding.tokens):
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        pieces.append({"id": token_id, "token": token, "text": decoded})
    if tokenizer.decode(encoding.ids, skip_special_tokens=False) != sample["text"]:
        raise RuntimeError(f"Round-trip mismatch on sample {sample['id']}")
    return {"count": len(encoding.ids), "pieces": pieces}


@app.function(image=image, volumes={str(ROOT): volume}, cpu=8, memory=65536, timeout=86_400)
def train_one(label: str, plan: dict, quota: dict[str, int], samples: list[dict],
              recipe: dict, settings: dict, smoke: bool = False) -> dict:
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

    if set(quota) != {source["name"] for source in recipe["buckets"]}:
        raise ValueError("Mixture and corpus recipe have different source buckets")
    rule = settings["pre_tokenizer"]
    use_regex = rule == TRAINING_SETTINGS["pre_tokenizer"]
    line_chunks = rule.startswith("Split(lines, max 256 characters)")
    version = "v5" if line_chunks else ("v3" if use_regex else "v4")
    job_id = run_id(plan, quota, smoke, settings)
    destination = ROOT / "tokenizers" / (f"smoke-{version}" if smoke else f"mixtures-{version}") / job_id / label.lower()
    tokenizer_path = destination / "tokenizer.json"
    manifest_path = destination / "manifest.json"
    if manifest_path.exists() and tokenizer_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["run_id"] != job_id:
            raise RuntimeError(f"Existing model has a different run ID: {destination}")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        print(f"Reusing completed {label} at {destination}", flush=True)
    else:
        algorithm = settings["algorithm"]
        if algorithm == "byte-level BPE":
            tokenizer = Tokenizer(models.BPE())
            trainer = trainers.BpeTrainer(
                vocab_size=VOCAB_SIZE, min_frequency=2, max_token_length=64,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                special_tokens=settings["special_tokens"], show_progress=False,
            )
        elif algorithm == "wordpiece":
            tokenizer = Tokenizer(models.WordPiece(
                unk_token="<|unk|>", max_input_chars_per_word=1_000_000))
            trainer = trainers.WordPieceTrainer(
                vocab_size=VOCAB_SIZE, min_frequency=2,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                continuing_subword_prefix="",
                special_tokens=settings["special_tokens"], show_progress=False,
            )
        elif algorithm == "unigram":
            tokenizer = Tokenizer(models.Unigram())
            trainer = trainers.UnigramTrainer(
                vocab_size=VOCAB_SIZE, max_piece_length=64,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                unk_token="<|unk|>",
                special_tokens=settings["special_tokens"], show_progress=False,
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        if line_chunks:
            tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
                pre_tokenizers.Split(Regex(r"[^\n]{1,256}\n?|\n"), behavior="isolated"),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ])
        else:
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
                add_prefix_space=False, use_regex=use_regex)
        tokenizer.decoder = decoders.ByteLevel()
        stats = {}
        start = time.monotonic()
        tokenizer.train_from_iterator(iter_quota_batches(recipe, quota, stats), trainer=trainer)
        elapsed = time.monotonic() - start
        destination.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(tokenizer_path))
        manifest = {"label": label, "run_id": job_id, "recipe": plan["source_recipe"],
                    "settings": settings, "source_stats": stats,
                    "total_used_bytes": sum(item["used_bytes"] for item in stats.values()),
                    "vocab_size_actual": tokenizer.get_vocab_size(), "train_seconds": round(elapsed, 2),
                    "model_path": str(tokenizer_path)}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        volume.commit()
        print(f"Trained {label}: {manifest['vocab_size_actual']:,} tokens in {elapsed:.1f}s", flush=True)
    return {"manifest": manifest,
            "segmentations": {sample["id"]: segmentation(tokenizer, sample) for sample in samples}}


@app.local_entrypoint()
def main(smoke: bool = False, only: str = "", plan: str = "results/mixture_examples.json",
         output: str = "", pretokenizer: str = "regex", algorithm: str = "bpe") -> None:
    if os.environ.get("MODAL_PROFILE") != "group-cron":
        raise RuntimeError("Set MODAL_PROFILE=group-cron before training")
    project = Path(__file__).resolve().parent.parent
    settings = training_settings(pretokenizer, algorithm)
    plan = json.loads((project / plan).read_text())
    recipe = json.loads((project / "data" / "balanced-full-v3.json").read_text())
    sample_path = project / "dashboard" / "sample_passages.json"
    samples = json.loads(sample_path.read_text()) if sample_path.exists() else []
    if plan["source_recipe"] != recipe["name"]:
        raise ValueError("Mixture plan does not match the corpus recipe")
    entries = [("Base", plan["baseline"])] + [(item["label"], item) for item in plan["mixtures"]]
    if only:
        requested = {label.strip().lower() for label in only.split(",")}
        entries = [entry for entry in entries if entry[0].lower() in requested]
        if {label.lower() for label, _ in entries} != requested:
            raise ValueError(f"Unknown mixture in: {only}")
    jobs = []
    for label, entry in entries:
        quota = entry["bucket_bytes"]
        if smoke:
            total = 40_000_000
            quota = {name: round(amount / plan["total_train_bytes_per_mixture"] * total)
                     for name, amount in quota.items()}
        print(f"Submitting {label}: {sum(quota.values()):,} bytes", flush=True)
        jobs.append((label, plan, quota, samples, recipe, settings, smoke))
    results = {}
    for result in train_one.starmap(jobs, order_outputs=False):
        label = result["manifest"]["label"]
        results[label] = result
        print(f"Completed {label}: {result['manifest']['model_path']}", flush=True)
    if smoke:
        print("Smoke run complete; dashboard output unchanged.")
        return
    if not output and (only or {label for label, _ in entries} != {"Base", "A", "B", "C", "D"}):
        print("Partial or experimental training complete; specify --output to save local segmentations.")
        return
    payload = {"recipe": plan["source_recipe"], "settings": settings,
              "samples": samples,
              "models": {label: result["manifest"] for label, result in results.items()},
              "segmentations": {label: result["segmentations"] for label, result in results.items()}}
    path = project / (output or "results/tokenizer_segmentations.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
