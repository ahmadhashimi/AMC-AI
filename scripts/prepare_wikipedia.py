import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

source = Path("data/raw/wikipedia_fa/articles.jsonl")
folder = Path("data/processed/wikipedia_fa_v1")
folder.mkdir(parents=True, exist_ok=True)

if any(folder.iterdir()):
    raise SystemExit("Output folder is not empty. Stopping to preserve files.")

seen = set()
counts = Counter()
files = {}

try:
    for split in ("train", "val", "test"):
        files[split] = (folder / f"{split}.jsonl.partial").open(
            "x", encoding="utf-8"
        )

    with source.open(encoding="utf-8") as articles:
        for number, line in enumerate(articles, start=1):
            article = json.loads(line)
            text = unicodedata.normalize("NFC", article["text"]).strip()
            digest = hashlib.sha256(text.encode("utf-8")).digest()

            if digest in seen:
                counts["duplicates_removed"] += 1
                continue

            seen.add(digest)

            # Assign whole articles, never individual sentences.
            bucket = int.from_bytes(digest[:8], "big") % 10000
            split = "train" if bucket < 9800 else (
                "val" if bucket < 9900 else "test"
            )

            article["text"] = text
            files[split].write(
                json.dumps(article, ensure_ascii=False) + "\n"
            )
            counts[split] += 1

            if number % 100000 == 0:
                print(f"Read {number:,} articles", flush=True)

    if number != 979869 or len(seen) != 979741:
        raise ValueError("Counts differ from our earlier checks.")

finally:
    for file in files.values():
        file.close()

for split in ("train", "val", "test"):
    (folder / f"{split}.jsonl.partial").rename(
        folder / f"{split}.jsonl"
    )

report = {
    "source": str(source),
    "source_articles": number,
    "unique_articles": len(seen),
    "counts": dict(counts),
    "normalization": "NFC Unicode normalization and outer whitespace trimming",
    "deduplication": "Exact normalized text only; near duplicates not checked",
    "split_method": "SHA-256 of normalized article text; approximately 98/1/1",
}

(folder / "preparation_report.json").write_text(
    json.dumps(report, indent=2) + "\n",
    encoding="utf-8",
)

print(json.dumps(report, indent=2))