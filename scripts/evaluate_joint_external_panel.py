"""Evaluate 32 unseen input tokenizers on the same recent external text.

Only document prefixes use generated embeddings. Continuations, predictions,
and the output head always use canonical SmolLM2 tokens. The original and
jointly trained canonical baselines score the identical continuations.
"""

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hf_hypernet.joint_training import make_lm_microbatch
from hf_hypernet.modeling_hypernet import ZettHypernet
from hf_hypernet.smollm_setup import separate_input_embeddings_and_freeze_output
from hf_hypernet.tokenizer_inputs import build_surface_forms, enable_byte_fallbacks
from scripts.train_joint_smollm_input import (
    SPECIAL_ID, SPECIAL_TOKEN, tokenizer_from_vocab,
)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                     prefix=".external-", delete=False) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def read_gzip_json(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def read_gzip_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@torch.inference_mode()
def score_group(model, examples, input_table, continuation_table):
    """Sum loss over the canonical continuation tokens only."""
    examples = sorted(examples, key=lambda x: len(x[0]) + len(x[1]))
    total_nats = 0.0
    total_tokens = 0
    index = 0
    while index < len(examples):
        length = 1 + len(examples[index][0]) + len(examples[index][1])
        if length <= 1024:
            batch_size, tier_max = 4, 1024
        elif length <= 2048:
            batch_size, tier_max = 2, 2048
        else:
            batch_size, tier_max = 1, 8192
        # Sorted lengths keep each GPU batch within roughly 4096 positions.
        stop = index + 1
        while stop < min(index + batch_size, len(examples)):
            next_length = 1 + len(examples[stop][0]) + len(examples[stop][1])
            if next_length > tier_max:
                break
            stop += 1
        micro = examples[index:stop]
        embeds, mask, labels, positions = make_lm_microbatch(
            micro, input_table, continuation_table, "cuda"
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(inputs_embeds=embeds, attention_mask=mask,
                         position_ids=positions, labels=labels,
                         use_cache=False).loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite evaluation loss")
        scored = sum(len(row[1]) for row in micro)
        total_nats += float(loss) * scored
        total_tokens += scored
        index = stop
    return {"nats": total_nats, "tokens": total_tokens,
            "bytes": sum(row[2] for row in examples),
            "bpb": total_nats / sum(row[2] for row in examples) / math.log(2)}


def score_sources(model, groups, input_table, continuation_table, sources):
    return {name: score_group(model, groups[name], input_table,
                              continuation_table) for name in sources}


def aggregate(groups):
    nats = sum(row["nats"] for row in groups.values())
    tokens = sum(row["tokens"] for row in groups.values())
    bytes_total = sum(row["bytes"] for row in groups.values())
    return {"nats": nats, "tokens": tokens, "bytes": bytes_total,
            "bpb": nats / bytes_total / math.log(2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", default=["education", "government"])
    parser.add_argument("--indices", nargs="+", type=int)
    parser.add_argument("--method", default=(
        "same whole-document midpoint whitespace cut for all 32; "
        "new tokenizer only on prefix; canonical continuation "
        "and output head; up to 8192 positions; score only "
        "continuation UTF-8 bytes"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA GPU required")
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.run / "run_config.json").read_text())
    sources = tuple(args.sources)
    if len(set(sources)) != len(sources):
        parser.error("Source names must be unique")
    metadata = {"checkpoint": str(args.checkpoint.resolve()),
                "corpus_sha256": digest(args.corpus),
                "examples_sha256": digest(args.examples),
                "specs_sha256": digest(args.specs),
                "evaluation_script_sha256": digest(Path(__file__)),
                "joint_training_helper_sha256": digest(
                    REPO / "hf_hypernet" / "joint_training.py"
                ),
                "sources": list(sources),
                "selected_indices": args.indices,
                "method": args.method}
    metadata_path = args.output / "metadata.json"
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != metadata:
            raise ValueError("Existing results have different settings")
    elif any(args.output.iterdir()):
        raise ValueError("Output directory contains unrelated files")
    else:
        save_json(metadata_path, metadata)
    source_tok = AutoTokenizer.from_pretrained(config["model"],
                                                local_files_only=True)
    corpus = {row["id"]: row for row in read_gzip_jsonl(args.corpus)}
    prepared = read_gzip_jsonl(args.examples)
    specs = read_gzip_json(args.specs)
    if len(specs) != 32 or len({row["index"] for row in specs}) != 32:
        raise ValueError("Expected 32 distinct saved tokenizer specs")
    if args.indices is not None:
        if len(set(args.indices)) != len(args.indices) or not set(args.indices) <= {row["index"] for row in specs}:
            raise ValueError("Invalid tokenizer indices")
        specs = [row for row in specs if row["index"] in args.indices]
    if (not prepared or len({row["id"] for row in prepared}) != len(prepared)
            or set(row["source"] for row in prepared) != set(sources)
            or any(row["id"] not in corpus for row in prepared)):
        raise ValueError("Prepared examples, corpus IDs, or sources differ")
    prepared_summary = args.examples.with_suffix(".summary.json")
    if prepared_summary.is_file():
        expected = json.loads(prepared_summary.read_text())
        if (expected["corpus_sha256"] != metadata["corpus_sha256"]
                or expected["specs_sha256"] != metadata["specs_sha256"]
                or expected["examples_sha256"] != metadata["examples_sha256"]
                or expected["selected_documents"] != len(prepared)
                or expected["selected_continuation_bytes"] != sum(
                    row["continuation_bytes"] for row in prepared)):
            raise ValueError("Prepared examples differ from their audited summary")
    baseline_path = args.output / "baselines.json"
    if baseline_path.exists():
        baselines = json.loads(baseline_path.read_text())
    else:
        canonical = {name: [] for name in sources}
        for row in prepared:
            canonical[row["source"]].append((
                row["canonical_prefix_ids"],
                row["canonical_continuation_ids"],
                row["continuation_bytes"],
            ))
        original = AutoModelForCausalLM.from_pretrained(
            config["model"], local_files_only=True, dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to("cuda").eval()
        original_table = original.get_input_embeddings().weight.detach()
        original_groups = score_sources(original, canonical, original_table,
                                        original_table, sources)
        original_input = original_table.cpu().clone()
        original_output = original.get_output_embeddings().weight.detach().cpu().clone()
        del original, original_table
        torch.cuda.empty_cache()
        trained = AutoModelForCausalLM.from_pretrained(
            config["model"], local_files_only=True, dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to("cuda")
        separate_input_embeddings_and_freeze_output(trained)
        trained.load_state_dict(torch.load(args.checkpoint / "lm.pt",
                                         map_location="cpu", weights_only=True))
        trained.eval()
        if not torch.equal(trained.get_input_embeddings().weight.cpu(),
                           original_input):
            raise ValueError("Canonical input table changed")
        if not torch.equal(trained.get_output_embeddings().weight.cpu(),
                           original_output):
            raise ValueError("Output head changed")
        trained_table = trained.get_input_embeddings().weight.detach()
        trained_groups = score_sources(trained, canonical, trained_table,
                                       trained_table, sources)
        baselines = {
            "original": {"overall": aggregate(original_groups),
                         "by_source": original_groups},
            "trained": {"overall": aggregate(trained_groups),
                        "by_source": trained_groups},
            "examples": len(prepared),
        }
        save_json(baseline_path, baselines)
        print(f"Matched canonical BPB: original "
              f"{baselines['original']['overall']['bpb']:.5f}, trained "
              f"{baselines['trained']['overall']['bpb']:.5f}", flush=True)
        del trained, trained_table
        torch.cuda.empty_cache()
    trained = AutoModelForCausalLM.from_pretrained(
        config["model"], local_files_only=True, dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to("cuda")
    separate_input_embeddings_and_freeze_output(trained)
    trained.load_state_dict(torch.load(args.checkpoint / "lm.pt",
                                     map_location="cpu", weights_only=True))
    trained.eval()
    trained_table = trained.get_input_embeddings().weight.detach()
    hypernet = ZettHypernet.from_pretrained(
        args.checkpoint / "hypernet", local_files_only=True,
    ).to("cuda").eval()
    fallback = enable_byte_fallbacks(hypernet, source_tok)
    results_path = args.output / "tokenizers.jsonl"
    rows = ([json.loads(line) for line in results_path.read_text().splitlines()
             if line.strip()] if results_path.exists() else [])
    completed = {row["index"] for row in rows}
    if len(completed) != len(rows):
        raise ValueError("Duplicate tokenizer result")
    for spec in sorted(specs, key=lambda row: row["index"]):
        index = spec["index"]
        if index in completed:
            continue
        started = time.perf_counter()
        tokenizer = tokenizer_from_vocab(spec["vocab"])
        groups = {name: [] for name in sources}
        for row in prepared:
            prefix = corpus[row["id"]]["text"][:row["cut_chars"]]
            target_ids = tokenizer.encode(
                prefix, add_special_tokens=False
            ).ids
            if tokenizer.decode(target_ids) != prefix:
                raise ValueError("Prepared prefix no longer roundtrips")
            if 1 + len(target_ids) + len(row["canonical_continuation_ids"]) > 8192:
                raise ValueError("Prepared example exceeds position limit")
            groups[row["source"]].append((
                target_ids, row["canonical_continuation_ids"],
                row["continuation_bytes"],
            ))
        forms_np, truncated = build_surface_forms(
            [piece for piece, _ in spec["vocab"]], source_tok, 10,
            fallback, {SPECIAL_TOKEN: SPECIAL_ID},
        )
        forms = torch.from_numpy(forms_np.astype(np.int64)).to("cuda")
        priors = torch.tensor([value for _, value in spec["vocab"]],
                              device="cuda", dtype=torch.float32)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated, output_embeddings, _ = hypernet(
                forms, priors, trained_table, None, deterministic=True
            )
        if output_embeddings is not None or not torch.isfinite(generated).all():
            raise FloatingPointError("Invalid generated input embeddings")
        torch.cuda.reset_peak_memory_stats()
        per_source = score_sources(trained, groups, generated, trained_table,
                                   sources)
        overall = aggregate(per_source)
        if (overall["bytes"] != baselines["trained"]["overall"]["bytes"]
                or overall["tokens"] != baselines["trained"]["overall"]["tokens"]):
            raise AssertionError("Tokenizer scores different continuations")
        result = {"index": index, "noise_std": spec["noise_std"],
                  "truncated_forms": truncated,
                  "overall": overall, "by_source": per_source,
                  "seconds": time.perf_counter() - started,
                  "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30}
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        rows.append(result)
        print(f"Tokenizer {index + 1}/32: {overall['bpb']:.5f} BPB, "
              f"{result['seconds']:.1f}s, {result['peak_gpu_gib']:.1f} GiB",
              flush=True)
        del generated, forms, priors, groups
    if len(rows) != len(specs):
        raise AssertionError("Incomplete tokenizer results")
    rows.sort(key=lambda row: row["index"])
    values = [row["overall"]["bpb"] for row in rows]
    baseline = baselines["trained"]["overall"]["bpb"]
    original = baselines["original"]["overall"]["bpb"]
    summary = {
        "tokenizers": len(rows),
        "examples_per_tokenizer": len(prepared),
        "continuation_bytes_per_tokenizer": rows[0]["overall"]["bytes"],
        "original_canonical_bpb": original,
        "trained_canonical_bpb": baseline,
        "transferred_mean_bpb": statistics.mean(values),
        "transferred_median_bpb": statistics.median(values),
        "transferred_min_bpb": min(values),
        "transferred_max_bpb": max(values),
        "mean_transfer_gap_vs_trained_canonical_bpb":
            statistics.mean(values) - baseline,
        "mean_gap_vs_original_canonical_bpb":
            statistics.mean(values) - original,
        "transferred_better_than_trained_canonical": sum(x < baseline for x in values),
        "transferred_better_than_original_canonical": sum(x < original for x in values),
        "mean_seconds_per_tokenizer": statistics.mean(row["seconds"] for row in rows),
        "by_source": {},
    }
    for name in sources:
        source_values = [row["by_source"][name]["bpb"] for row in rows]
        summary["by_source"][name] = {
            "bytes_per_tokenizer": rows[0]["by_source"][name]["bytes"],
            "original_canonical_bpb": baselines["original"]["by_source"][name]["bpb"],
            "trained_canonical_bpb": baselines["trained"]["by_source"][name]["bpb"],
            "transferred_mean_bpb": statistics.mean(source_values),
            "transferred_min_bpb": min(source_values),
            "transferred_max_bpb": max(source_values),
        }
    save_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
