"""Collect matched 2 GB BPE, WordPiece, and Unigram cuts for 23 real excerpts."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from tokenizers import Tokenizer

from evaluate_algorithms import model_dir
from generate_mixture_examples import DOMAIN_BUCKETS


ROOT = Path(__file__).resolve().parent.parent
ALGORITHMS = ("bpe", "wordpiece", "unigram")
LANGUAGES = {"en": "English", "es": "Spanish", "fr": "French", "ru": "Russian",
             "zh": "Mandarin", "ja": "Japanese", "ar": "Arabic", "hi": "Hindi"}


def load_samples() -> list[dict]:
    curated = ROOT / "dashboard/sample_passages.json"
    if curated.exists():
        return json.loads(curated.read_text())
    recipe = json.loads((ROOT / "data/balanced-full-v3.json").read_text())
    samples = []
    for bucket in recipe["buckets"]:
        name = bucket["name"]
        prefix, _, language = name.partition("-")
        if prefix in {"web", "wiki"}:
            label = f"{'Web' if prefix == 'web' else 'Wikipedia'} · {LANGUAGES[language]}"
        else:
            label = {"qa-en": "Q&A", "papers-en": "Papers"}.get(name, prefix.title())
        document = None
        with gzip.open(ROOT / "modal_results/eval" / f"{name}.jsonl.gz", "rt",
                       encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                if len(item["text"]) >= 100:
                    document = item
                    break
        if document is None:
            raise ValueError(f"No held-out document of at least 100 characters for {name}")
        content = document["text"][:320]
        samples.append({"id": name, "label": label, "text": content,
                        "source_dataset": bucket["dataset"], "source_id": document["id"],
                        "source_start": 0, "source_end": len(content)})
    return samples


def encode_sample(model: Tokenizer, sample: dict) -> dict:
    content = sample["text"]
    encoding = model.encode(content, add_special_tokens=False)
    if model.decode(encoding.ids, skip_special_tokens=False) != content:
        raise ValueError(f"Round-trip failed for {sample['id']}")
    pieces = []
    offset = 0
    for token_id, token in zip(encoding.ids, encoding.tokens):
        end = offset + len(token)
        pieces.append({"token": token,
                       "text": model.decode([token_id], skip_special_tokens=False),
                       "start": offset, "end": end})
        offset = end
    if offset != len(content.encode("utf-8")):
        raise ValueError(f"Byte offsets failed for {sample['id']}")
    return {"count": len(pieces), "pieces": pieces,
            "boundaries": [piece["end"] for piece in pieces[:-1]]}


def main() -> None:
    evaluation = json.loads((ROOT / "results/algorithm_eval_2gb.json").read_text())
    samples = load_samples()
    expected = {name for names in DOMAIN_BUCKETS.values() for name in names}
    if len(samples) != 23 or {sample["id"] for sample in samples} != expected:
        raise ValueError("Expected one held-out excerpt from each source bucket")
    models = {name: Tokenizer.from_file(str(model_dir(name, "2gb") / "tokenizer.json"))
              for name in ALGORITHMS}
    rows = []
    for sample in samples:
        bucket = sample["id"]
        counts = evaluation["buckets"][bucket]
        cuts = {name: encode_sample(model, sample) for name, model in models.items()}
        base = set(cuts["bpe"]["boundaries"])
        overlap = {}
        for name in ALGORITHMS[1:]:
            other = set(cuts[name]["boundaries"])
            overlap[name] = 100 * len(base & other) / len(base | other) if base | other else 100.0
        rows.append({
            "id": bucket,
            "label": sample["label"],
            "domain": counts["domain"],
            "text": sample["text"],
            "source_dataset": sample["source_dataset"],
            "source_id": sample["source_id"],
            "source_start": sample["source_start"],
            "source_end": sample["source_end"],
            "held_out_bytes": counts["bytes"],
            "held_out_documents": counts["documents"],
            "held_out_tokens": counts["tokens"],
            "cuts": cuts,
            "boundary_overlap_with_bpe_percent": overlap,
        })
    output = {"method": evaluation["method"],
              "total_held_out_tokens": evaluation["total_held_out_tokens"],
              "domains": evaluation["domains"],
              "rows": rows}
    destination = ROOT / "results/algorithm_cuts_2gb.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {destination}")
    for row in rows:
        counts = row["cuts"]
        print(f"{row['id']:12} {counts['bpe']['count']:4} / "
              f"{counts['wordpiece']['count']:4} / {counts['unigram']['count']:4} "
              f"sample tokens; boundary overlap {row['boundary_overlap_with_bpe_percent']}")


if __name__ == "__main__":
    main()
