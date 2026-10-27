"""Rebuild the 23 dashboard passages as exact held-out corpus excerpts.

Requires the eval files from the Modal corpus volume in
modal_results/eval/<bucket>.jsonl.gz. Document IDs and character offsets below
identify a contiguous slice of the original eval record; no text is rewritten.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from modal_prepare_corpus import SOURCE_REVISIONS


ROOT = Path(__file__).resolve().parent.parent
# bucket -> (source document ID, first character, one-past-last character)
EXCERPTS = {
    "web-en": ("<urn:uuid:997bd63a-951e-48ca-856a-1065c1784033>", 0, 316),
    "web-es": ("<urn:uuid:ca4d22c8-e739-412c-a9da-b50850d20760>", 0, 299),
    "web-fr": ("<urn:uuid:6584687d-eaa6-4a0b-982a-04822ea2ccfd>", 0, 306),
    "web-ru": ("<urn:uuid:49066caf-7272-4a23-94e3-ea46bf77147c>", 0, 365),
    "web-zh": ("<urn:uuid:ab3bb51e-8bc8-44af-afa0-8a64fc2c182a>", 0, 290),
    "web-ja": ("<urn:uuid:6bdad80b-dbf4-47d6-ada9-fd6f2f21cc99>", 2107, 2253),
    "web-ar": ("<urn:uuid:7e1d015b-c5bd-4224-b99d-3c650308ab5f>", 0, 313),
    "web-hi": ("<urn:uuid:1275f839-1244-4212-a6e5-6f8403a377ef>", 0, 269),
    "wiki-en": ("1445", 4029, 4329),
    "wiki-es": ("975", 18140, 18439),
    "wiki-fr": ("1047", 40432, 40732),
    "wiki-ru": ("9", 45171, 45472),
    "wiki-zh": ("610", 1321, 1623),
    "wiki-ja": ("1328", 753, 1052),
    "wiki-ar": ("2633", 6750, 7050),
    "wiki-hi": ("178", 42841, 43121),
    "code": ("uber-common/opentracing-python-instrumentation/tests/opentracing_instrumentation/test_mysqldb.py", 939, 1484),
    "papers-en": ("255294920", 75, 392),
    "math-en": ("https://www.coursehero.com/sg/college-algebra/solving-quadratic-equations-and-inequalities/", 680, 980),
    "qa-en": ("stackexchange-dolma-0040.json.gz:9655", 1146, 1440),
    "books-en": ("gutenberg:42427:5", 49480, 49768),
    "medical-en": ("PMC10209832", 24200, 24512),
    "legal-en": ("pa_80/html/0053-01.html", 8683, 8987),
}


def dataset_for(bucket: str) -> str:
    if bucket == "web-en":
        return "HuggingFaceFW/fineweb"
    if bucket.startswith("web-"):
        return "HuggingFaceFW/fineweb-2"
    if bucket.startswith("wiki-"):
        return "wikimedia/wikipedia"
    return {
        "code": "codeparrot/github-code-clean",
        "papers-en": "allenai/peS2o",
        "math-en": "HuggingFaceTB/finemath",
        "qa-en": "common-pile/stackexchange_filtered",
        "books-en": "common-pile/project_gutenberg_filtered",
        "medical-en": "common-pile/pubmed",
        "legal-en": "common-pile/caselaw_access_project_filtered",
    }[bucket]


def main() -> None:
    recipe = json.loads((ROOT / "data" / "balanced-full-v3.json").read_text())
    path = ROOT / "dashboard" / "sample_passages.json"
    previous = json.loads(path.read_text())
    labels = {sample["id"]: sample["label"] for sample in previous}
    bucket_names = [bucket["name"] for bucket in recipe["buckets"]]
    if set(bucket_names) != set(EXCERPTS) or set(bucket_names) != set(labels):
        raise ValueError("Excerpt list must match every recipe bucket and dropdown label")
    samples = []
    for name in bucket_names:
        source_id, start, end = EXCERPTS[name]
        eval_path = ROOT / "modal_results" / "eval" / f"{name}.jsonl.gz"
        matches = []
        with gzip.open(eval_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record["id"] == source_id:
                    matches.append(record)
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {name} eval document with ID {source_id}")
        full_text = matches[0]["text"]
        if not 0 <= start < end <= len(full_text):
            raise ValueError(f"Invalid excerpt offsets for {name}")
        excerpt = full_text[start:end]
        if not excerpt.strip() or len(excerpt) > 600:
            raise ValueError(f"Invalid excerpt text for {name}")
        dataset = dataset_for(name)
        samples.append({"id": name, "label": labels[name], "text": excerpt,
                        "source_dataset": dataset, "source_revision": SOURCE_REVISIONS[dataset],
                        "source_split": "eval", "source_id": source_id,
                        "source_start": start, "source_end": end})
        print(f"{name}: {len(excerpt)} chars from {source_id}")
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
