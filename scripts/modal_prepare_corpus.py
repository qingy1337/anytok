"""Prepare a bounded, labeled tokenizer-training corpus on Modal.

Run from the anytok repository root with the intended profile:

    MODAL_PROFILE=group-cron modal run scripts/modal_prepare_corpus.py --size pilot
    MODAL_PROFILE=group-cron modal run scripts/modal_prepare_corpus.py --size pilot --submit

The first command prints a plan. The second streams source documents into the
`anytok-corpus` Modal volume. A completed bucket is immutable and is skipped on
reruns. Pilot and full corpora occupy separate paths. Source revisions are
pinned below. Text is kept as received; no tokenizer-specific preprocessing is
applied here. The `full` option is 10 GB of training text, not full upstream
datasets, which are many terabytes.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Iterator

import modal


VOLUME_NAME = "anytok-corpus"
ROOT = Path("/corpus")
SOURCE_REVISIONS = {
    "HuggingFaceFW/fineweb": "9bb295ddab0e05d785b879661af7260fed5140fc",
    "HuggingFaceFW/fineweb-2": "af9c13333eb981300149d5ca60a8e9d659b276b9",
    "wikimedia/wikipedia": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
    "allenai/peS2o": "636a503e44a3ca1b58e01fb61eab0825cd574de0",
    "codeparrot/github-code-clean": "c48d40f9e70f0196f8236901ee35807f7d6c44c0",
    "HuggingFaceTB/finemath": "e92b25a616738fe95dc186b64dfb19f9c8525594",
    "common-pile/stackexchange_filtered": "c0ac7373830c688a43fc12d1988c4b19ccd884ab",
    "common-pile/project_gutenberg_filtered": "3cdf6879c807f4e4e063f2ceb23bc268d8c29ab7",
    "common-pile/pubmed": "648b8cfc93953ca0663a9c96a8d842a91b98fb64",
    "common-pile/caselaw_access_project_filtered": "50e1961a5ed8fdddcee64c9e66e1338de8953b63",
}
NON_ENGLISH = {
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "ru": "rus_Cyrl",
    "zh": "cmn_Hani",
    "ja": "jpn_Jpan",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
}
WIKI_LANGUAGES = ("en", *NON_ENGLISH)
PILOT_BYTES = 2_000_000_000
FULL_BYTES = 10_000_000_000
EXTRA_BYTES = 2_000_000_000
EVAL_BYTES_PER_BUCKET = 5_000_000
MAX_DOCUMENT_BYTES = 1_000_000
PERMISSIVE_CODE_LICENSES = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc"}
CODE_LANGUAGES = {"Python", "JavaScript", "TypeScript", "Java", "GO", "Rust", "C", "C++", "Shell"}

app = modal.App("anytok-corpus-prepare")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "datasets>=4.4,<6", "requests>=2.32"
)


def bucket_plan(total_bytes: int) -> list[dict]:
    specs = [{"name": "web-en", "kind": "fineweb", "train_target": total_bytes * 25 // 100}]
    specs.extend(
        {"name": f"web-{language}", "kind": "fineweb2", "language": language,
         "config": config, "train_target": total_bytes * 45 // 100 // len(NON_ENGLISH)}
        for language, config in NON_ENGLISH.items()
    )
    specs.extend(
        {"name": f"wiki-{language}", "kind": "wiki", "language": language,
         "train_target": total_bytes * 15 // 100 // len(WIKI_LANGUAGES)}
        for language in WIKI_LANGUAGES
    )
    specs.append({"name": "code", "kind": "code", "train_target": total_bytes * 10 // 100})
    specs.append({"name": "papers-en", "kind": "papers", "train_target": total_bytes * 5 // 100})
    for spec in specs:
        spec["eval_target"] = EVAL_BYTES_PER_BUCKET
    return specs


def extra_plan() -> list[dict]:
    """Additive pools for a 10 GB recipe with math, Q&A, and books."""
    return [
        {"name": "math-en", "kind": "math", "train_target": 1_000_000_000,
         "eval_target": EVAL_BYTES_PER_BUCKET},
        {"name": "qa-en", "kind": "qa", "train_target": 500_000_000,
         "eval_target": EVAL_BYTES_PER_BUCKET},
        {"name": "books-en", "kind": "books", "train_target": 500_000_000,
         "eval_target": EVAL_BYTES_PER_BUCKET},
    ]


def specialty_plan() -> list[dict]:
    """Pools large enough for a 30% share in a 2 GB tokenizer run."""
    return [
        {"name": "medical-en", "kind": "medical", "train_target": 800_000_000,
         "eval_target": EVAL_BYTES_PER_BUCKET},
        {"name": "legal-en", "kind": "legal", "train_target": 800_000_000,
         "eval_target": EVAL_BYTES_PER_BUCKET},
    ]


def balanced_recipe_targets(total_bytes: int) -> dict[str, int]:
    """Byte quotas for web 55%, wiki 10%, code 10%, papers 5%, and extras 20%."""
    web_total = total_bytes * 55 // 100
    english_web = web_total * 25 // 70
    other_web = web_total - english_web
    targets = {"web-en": english_web}
    for index, language in enumerate(NON_ENGLISH):
        targets[f"web-{language}"] = other_web // 7 + (index < other_web % 7)
    wiki_total = total_bytes * 10 // 100
    for index, language in enumerate(WIKI_LANGUAGES):
        targets[f"wiki-{language}"] = wiki_total // 8 + (index < wiki_total % 8)
    targets.update({"code": total_bytes * 10 // 100,
                    "papers-en": total_bytes * 5 // 100,
                    "math-en": total_bytes * 10 // 100,
                    "qa-en": total_bytes * 5 // 100,
                    "books-en": total_bytes * 5 // 100})
    assert sum(targets.values()) == total_bytes
    return targets


def specialty_recipe_targets(total_bytes: int) -> dict[str, int]:
    """Keep prior non-web shares; allocate 5% each to medical and legal."""
    targets = balanced_recipe_targets(total_bytes)
    web_names = [name for name in targets if name.startswith("web-")]
    web_total = sum(targets[name] for name in web_names)
    new_web = total_bytes * 45 // 100
    web_allocations = [new_web * targets[name] // web_total for name in web_names]
    for index in range(new_web - sum(web_allocations)):
        web_allocations[index] += 1
    targets.update(zip(web_names, web_allocations))
    targets["medical-en"] = total_bytes * 5 // 100
    targets["legal-en"] = total_bytes * 5 // 100
    assert sum(targets.values()) == total_bytes
    return targets


def plan_for_size(size: str) -> list[dict]:
    if size == "pilot":
        return bucket_plan(PILOT_BYTES)
    if size == "full":
        return bucket_plan(FULL_BYTES)
    if size == "extra":
        return extra_plan()
    if size == "specialty":
        return specialty_plan()
    raise ValueError("size must be 'pilot', 'full', 'extra', or 'specialty'")


def source_info(spec: dict) -> tuple[str, str, str]:
    kind = spec["kind"]
    if kind == "fineweb":
        return "HuggingFaceFW/fineweb", "sample-10BT", "hash-heldout"
    if kind == "fineweb2":
        return "HuggingFaceFW/fineweb-2", spec["config"], "upstream-test"
    if kind == "wiki":
        return "wikimedia/wikipedia", f"20231101.{spec['language']}", "hash-heldout"
    if kind == "code":
        return "codeparrot/github-code-clean", "permissive-selected-languages", "repository-hash-heldout"
    if kind == "math":
        return "HuggingFaceTB/finemath", "finemath-4plus", "url-hash-heldout"
    if kind == "qa":
        return "common-pile/stackexchange_filtered", "default", "document-hash-heldout"
    if kind == "books":
        return "common-pile/project_gutenberg_filtered", "default", "book-hash-heldout"
    if kind == "medical":
        return "common-pile/pubmed", "default", "article-hash-heldout"
    if kind == "legal":
        return "common-pile/caselaw_access_project_filtered", "default", "opinion-hash-heldout"
    return "allenai/peS2o", "v2", "upstream-validation"


def _hf_records(repo: str, config: str, split: str) -> Iterator[dict]:
    from datasets import load_dataset

    yield from load_dataset(
        repo, name=config, split=split, streaming=True,
        revision=SOURCE_REVISIONS[repo],
    )


def _paper_records(split: str) -> Iterator[dict]:
    import requests

    revision = SOURCE_REVISIONS["allenai/peS2o"]
    name = "train-00000-of-00020.json.gz" if split == "train" else "validation-00000-of-00002.json.gz"
    url = f"https://huggingface.co/datasets/allenai/peS2o/resolve/{revision}/data/v2/{name}"
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with gzip.GzipFile(fileobj=response.raw) as stream:
            for line in stream:
                if line:
                    yield json.loads(line)


def _code_records() -> Iterator[dict]:
    from datasets import load_dataset

    revision = SOURCE_REVISIONS["codeparrot/github-code-clean"]
    base = f"https://huggingface.co/datasets/codeparrot/github-code-clean/resolve/{revision}/data"
    for index in range(32):
        url = f"{base}/train-{index:05d}-of-00880.parquet"
        dataset = load_dataset("parquet", data_files=url, split="train", streaming=True)
        for record in dataset:
            if record.get("license", "").lower() in PERMISSIVE_CODE_LICENSES and record.get("language") in CODE_LANGUAGES:
                yield record


def _common_pile_records(repo: str, prefix: str, shard_count: int) -> Iterator[dict]:
    """Stream bounded slices across shards to avoid one site's or book range's bias."""
    import requests

    indices = list(range(shard_count))
    random.Random(20261022).shuffle(indices)
    revision = SOURCE_REVISIONS[repo]
    for index in indices:
        name = f"{prefix}-{index:04d}.json.gz"
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}"
        shard_bytes = 0
        with requests.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with gzip.GzipFile(fileobj=response.raw) as stream:
                for line in stream:
                    record = json.loads(line)
                    shard_bytes += len(record.get("text", "").encode("utf-8"))
                    yield record
                    if shard_bytes >= 90_000_000:
                        break


def _book_records() -> Iterator[dict]:
    for book in _common_pile_records(
        "common-pile/project_gutenberg_filtered", "project_gutenberg-dolma", 15
    ):
        if book.get("metadata", {}).get("language") != "en":
            continue
        book_id = str(book.get("id", ""))
        if not book_id:
            continue
        content = book.get("text", "")
        # Keep long books while assigning every chunk of a book to one split.
        for offset in range(0, len(content), 50_000):
            yield {"id": f"gutenberg:{book_id}:{offset // 50_000}",
                   "split_key": f"gutenberg:{book_id}", "text": content[offset:offset + 50_000]}


def _specialty_records(kind: str) -> Iterator[dict]:
    """Stream a deterministic, bounded slice of each shard from HF."""
    import requests

    if kind == "medical":
        repo, count = "common-pile/pubmed", 17
        filename = lambda i: f"data/{i:05d}_pubmedcentral.jsonl.gz"
    else:
        repo, count = "common-pile/caselaw_access_project_filtered", 27
        filename = lambda i: f"CAP-Dolma-{i:04d}.json.gz"
    indices = list(range(count))
    random.Random(20261023).shuffle(indices)
    for index in indices:
        url = f"https://huggingface.co/datasets/{repo}/resolve/{SOURCE_REVISIONS[repo]}/{filename(index)}"
        shard_bytes = 0
        with requests.get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            with gzip.GzipFile(fileobj=response.raw) as stream:
                for line in stream:
                    record = json.loads(line)
                    shard_bytes += len(record.get("text", "").encode("utf-8"))
                    yield record
                    if shard_bytes >= 100_000_000:
                        break


def records(spec: dict, split: str) -> Iterator[dict]:
    kind = spec["kind"]
    if kind == "fineweb":
        yield from _hf_records("HuggingFaceFW/fineweb", "sample-10BT", "train")
    elif kind == "fineweb2":
        yield from _hf_records("HuggingFaceFW/fineweb-2", spec["config"], split)
    elif kind == "wiki":
        yield from _hf_records("wikimedia/wikipedia", f"20231101.{spec['language']}", "train")
    elif kind == "code":
        yield from _code_records()
    elif kind == "math":
        yield from _hf_records("HuggingFaceTB/finemath", "finemath-4plus", "train")
    elif kind == "qa":
        yield from _common_pile_records(
            "common-pile/stackexchange_filtered", "stackexchange-dolma", 44
        )
    elif kind == "books":
        yield from _book_records()
    elif kind in {"medical", "legal"}:
        yield from _specialty_records(kind)
    else:
        yield from _paper_records(split)


def record_fields(spec: dict, record: dict) -> tuple[str, str, str]:
    if spec["kind"] == "code":
        identifier = f"{record.get('repo_name', '')}/{record.get('path', '')}"
        return record.get("code", ""), identifier, record.get("repo_name", "")
    if spec["kind"] == "qa":
        identifier = str(record.get("metadata", {}).get("provenance") or record.get("id") or "")
        return record.get("text", ""), identifier, identifier
    if spec["kind"] == "books":
        return record["text"], record["id"], record["split_key"]
    identifier = str(record.get("id") or record.get("url") or "")
    return record.get("text", ""), identifier, identifier


def held_out(key: str) -> bool:
    return hashlib.sha256(key.encode("utf-8")).digest()[0] < 26


def _write_record(handle, text: str, identifier: str) -> None:
    handle.write(json.dumps({"id": identifier, "text": text}, ensure_ascii=False).encode("utf-8"))
    handle.write(b"\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_bucket(root: Path, spec: dict) -> dict:
    repo, config, split_policy = source_info(spec)
    destination = root / spec["name"]
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["spec"] != spec or manifest["revision"] != SOURCE_REVISIONS[repo]:
            raise RuntimeError(f"Existing bucket has different settings: {destination}")
        print(f"Skipping completed {spec['name']}", flush=True)
        return manifest
    destination.mkdir(parents=True, exist_ok=True)
    train_tmp = destination / "train.jsonl.gz.tmp"
    eval_tmp = destination / "eval.jsonl.gz.tmp"
    counts = {"train_bytes": 0, "eval_bytes": 0, "train_docs": 0, "eval_docs": 0,
              "seen_train_rows": 0, "seen_eval_rows": 0, "skipped_docs": 0}
    separate_eval = split_policy in {"upstream-test", "upstream-validation"}
    try:
        with gzip.open(train_tmp, "wb") as train_file, gzip.open(eval_tmp, "wb") as eval_file:
            for record in records(spec, "train"):
                counts["seen_train_rows"] += 1
                text, identifier, split_key = record_fields(spec, record)
                if not text or not identifier:
                    counts["skipped_docs"] += 1
                    continue
                byte_count = len(text.encode("utf-8"))
                if byte_count < 100 or byte_count > MAX_DOCUMENT_BYTES:
                    counts["skipped_docs"] += 1
                    continue
                use_eval = not separate_eval and held_out(split_key)
                if use_eval and counts["eval_bytes"] < spec["eval_target"]:
                    _write_record(eval_file, text, identifier)
                    counts["eval_bytes"] += byte_count
                    counts["eval_docs"] += 1
                elif not use_eval and counts["train_bytes"] < spec["train_target"]:
                    _write_record(train_file, text, identifier)
                    counts["train_bytes"] += byte_count
                    counts["train_docs"] += 1
                if counts["train_bytes"] >= spec["train_target"] and (separate_eval or counts["eval_bytes"] >= spec["eval_target"]):
                    break
            if separate_eval:
                eval_split = "test" if split_policy == "upstream-test" else "validation"
                for record in records(spec, eval_split):
                    counts["seen_eval_rows"] += 1
                    text, identifier, _ = record_fields(spec, record)
                    if not text or not identifier:
                        counts["skipped_docs"] += 1
                        continue
                    byte_count = len(text.encode("utf-8"))
                    if byte_count < 100 or byte_count > MAX_DOCUMENT_BYTES:
                        counts["skipped_docs"] += 1
                        continue
                    _write_record(eval_file, text, identifier)
                    counts["eval_bytes"] += byte_count
                    counts["eval_docs"] += 1
                    if counts["eval_bytes"] >= spec["eval_target"]:
                        break
        if counts["train_bytes"] < spec["train_target"] or counts["eval_bytes"] < spec["eval_target"]:
            raise RuntimeError(f"Source exhausted before target for {spec['name']}: {counts}")
        train_path = destination / "train.jsonl.gz"
        eval_path = destination / "eval.jsonl.gz"
        os.replace(train_tmp, train_path)
        os.replace(eval_tmp, eval_path)
        manifest = {
            "spec": spec, "dataset": repo, "config": config,
            "revision": SOURCE_REVISIONS[repo], "split_policy": split_policy,
            "counts": counts,
            "files": {"train.jsonl.gz": _sha256(train_path), "eval.jsonl.gz": _sha256(eval_path)},
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest
    finally:
        train_tmp.unlink(missing_ok=True)
        eval_tmp.unlink(missing_ok=True)


@app.function(image=image, volumes={str(ROOT): volume}, cpu=2, memory=4096, timeout=86_400)
def prepare(size: str) -> list[dict]:
    root = ROOT / f"{size}-v1"
    manifests = []
    for spec in plan_for_size(size):
        print(f"Preparing {spec['name']}: {spec['train_target']:,} train bytes", flush=True)
        manifest = prepare_bucket(root, spec)
        volume.commit()
        manifests.append(manifest)
        print(f"Committed {spec['name']} ({manifest['counts']['train_bytes']:,} train bytes)", flush=True)
    summary = {"size": size, "total_train_bytes": sum(m["counts"]["train_bytes"] for m in manifests),
               "total_eval_bytes": sum(m["counts"]["eval_bytes"] for m in manifests),
               "buckets": [m["spec"]["name"] for m in manifests]}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    volume.commit()
    print(json.dumps(summary), flush=True)
    return manifests


@app.function(image=image, cpu=2, memory=4096, timeout=3_600)
def probe_sources(size: str = "full") -> dict[str, dict]:
    """Verify source adapters from the actual Modal environment."""
    examples = [
        {"name": "web-en", "kind": "fineweb"},
        {"name": "web-es", "kind": "fineweb2", "config": "spa_Latn"},
        {"name": "wiki-en", "kind": "wiki", "language": "en"},
        {"name": "code", "kind": "code"},
        {"name": "papers-en", "kind": "papers"},
        {"name": "math-en", "kind": "math"},
        {"name": "qa-en", "kind": "qa"},
        {"name": "books-en", "kind": "books"},
        {"name": "medical-en", "kind": "medical"},
        {"name": "legal-en", "kind": "legal"},
    ]
    result = {}
    selected = (examples[-2:] if size == "specialty" else
                examples[-5:-2] if size == "extra" else examples)
    for spec in selected:
        record = next(records(spec, "train"))
        content, identifier, _ = record_fields(spec, record)
        result[spec["name"]] = {"id": identifier, "bytes": len(content.encode("utf-8"))}
        print(f"Probed {spec['name']}: {result[spec['name']]['bytes']} bytes", flush=True)
    return result


@app.function(image=image, volumes={str(ROOT): volume}, cpu=2, memory=4096, timeout=86_400)
def verify_corpus(size: str) -> dict:
    """Check committed file hashes and every document count and byte total."""
    root = ROOT / f"{size}-v1"
    summary = json.loads((root / "summary.json").read_text())
    verified = []
    totals = {"train": 0, "eval": 0}
    for name in summary["buckets"]:
        directory = root / name
        manifest = json.loads((directory / "manifest.json").read_text())
        counts = manifest["counts"]
        for split in ("train", "eval"):
            filename = f"{split}.jsonl.gz"
            path = directory / filename
            if _sha256(path) != manifest["files"][filename]:
                raise RuntimeError(f"Checksum mismatch: {path}")
            actual_bytes = actual_docs = 0
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    actual_bytes += len(record["text"].encode("utf-8"))
                    actual_docs += 1
            if actual_bytes != counts[f"{split}_bytes"] or actual_docs != counts[f"{split}_docs"]:
                raise RuntimeError(f"Manifest count mismatch: {path}")
            totals[split] += actual_bytes
        verified.append(name)
        print(f"Verified {name}", flush=True)
    for split in ("train", "eval"):
        if totals[split] != summary[f"total_{split}_bytes"]:
            raise RuntimeError(f"Corpus summary {split}-byte mismatch")
    return {"size": size, "verified_buckets": len(verified),
            "train_bytes": summary["total_train_bytes"], "eval_bytes": summary["total_eval_bytes"]}


@app.function(image=image, volumes={str(ROOT): volume}, cpu=1, memory=1024, timeout=600)
def publish_recipes() -> list[str]:
    """Publish exact byte quotas that reuse the existing pools without copying text."""
    destinations = []
    for size, total in (("pilot", PILOT_BYTES), ("full", FULL_BYTES)):
        buckets = []
        for name, target in balanced_recipe_targets(total).items():
            pool = "extra" if name in {"math-en", "qa-en", "books-en"} else size
            directory = ROOT / f"{pool}-v1" / name
            manifest = json.loads((directory / "manifest.json").read_text())
            available = manifest["counts"]["train_bytes"]
            if available < target:
                raise RuntimeError(f"Insufficient data in {directory}: {available} < {target}")
            buckets.append({"name": name, "source_path": str(directory),
                            "train_target_bytes": target, "train_available_bytes": available,
                            "eval_path": str(directory / "eval.jsonl.gz")})
        recipe = {"name": f"balanced-{size}-v2", "profile": "group-cron",
                  "volume": VOLUME_NAME, "unit": "raw UTF-8 text bytes",
                  "total_train_target_bytes": total, "buckets": buckets}
        path = ROOT / "recipes" / f"balanced-{size}-v2.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(recipe, indent=2) + "\n")
        destinations.append(str(path))
    size, total = "full", FULL_BYTES
    buckets = []
    for name, target in specialty_recipe_targets(total).items():
        pool = ("specialty" if name in {"medical-en", "legal-en"} else
                "extra" if name in {"math-en", "qa-en", "books-en"} else size)
        directory = ROOT / f"{pool}-v1" / name
        manifest = json.loads((directory / "manifest.json").read_text())
        available = manifest["counts"]["train_bytes"]
        if available < target:
            raise RuntimeError(f"Insufficient data in {directory}: {available} < {target}")
        buckets.append({"name": name, "source_path": str(directory),
                        "train_target_bytes": target, "train_available_bytes": available,
                        "eval_path": str(directory / "eval.jsonl.gz"),
                        "dataset": manifest["dataset"], "revision": manifest["revision"]})
    recipe = {"name": "balanced-full-v3", "profile": "group-cron",
              "volume": VOLUME_NAME, "unit": "raw UTF-8 text bytes",
              "total_train_target_bytes": total, "buckets": buckets}
    path = ROOT / "recipes" / "balanced-full-v3.json"
    path.write_text(json.dumps(recipe, indent=2) + "\n")
    destinations.append(str(path))
    volume.commit()
    return destinations


@app.local_entrypoint()
def main(size: str = "pilot", submit: bool = False, probe: bool = False,
         verify: bool = False, publish: bool = False) -> None:
    plan = plan_for_size(size)
    total = sum(spec["train_target"] for spec in plan)
    print(f"Modal profile: {os.environ.get('MODAL_PROFILE', '(default)')}")
    print(f"Training budget: {total:,} raw UTF-8 bytes")
    print(f"Held-out budget: {EVAL_BYTES_PER_BUCKET:,} bytes per bucket")
    print(f"Volume path: {VOLUME_NAME}:/{size}-v1/")
    for spec in plan:
        repo, config, policy = source_info(spec)
        print(f"  {spec['name']}: {spec['train_target']:,} bytes from {repo} ({config}; {policy})")
    if probe:
        print(json.dumps(probe_sources.remote(size), indent=2))
    if verify:
        print(json.dumps(verify_corpus.remote(size), indent=2))
    if submit:
        if os.environ.get("MODAL_PROFILE") != "group-cron":
            raise RuntimeError("Set MODAL_PROFILE=group-cron before submitting")
        prepare.remote(size)
    if publish:
        if os.environ.get("MODAL_PROFILE") != "group-cron":
            raise RuntimeError("Set MODAL_PROFILE=group-cron before publishing recipes")
        print(json.dumps(publish_recipes.remote(), indent=2))
