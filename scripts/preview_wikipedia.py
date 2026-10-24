"""Download five Persian Wikipedia articles for review only; never trains a model."""
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "wikipedia_fa_review"
DATASET = "wikimedia/wikipedia"
CONFIG = "20231101.fa"


def main():
    if OUT.exists():
        raise SystemExit(f"Review folder already exists; preserving it: {OUT}")
    url = "https://datasets-server.huggingface.co/rows?" + urlencode({
        "dataset": DATASET, "config": CONFIG, "split": "train", "offset": 0, "length": 5,
    })
    request = Request(url, headers={"User-Agent": "AMC-dataset-review/0.1"})
    with urlopen(request, timeout=60) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise SystemExit("Sample exceeds the 2 MB review limit")
    payload = json.loads(raw)
    records = [item["row"] for item in payload["rows"]]
    if not records or len(records) > 5:
        raise SystemExit("Unexpected sample size")
    OUT.mkdir(parents=True)
    (OUT / "response.json").write_bytes(raw)
    metadata = {
        "dataset": DATASET, "config": CONFIG, "split": "train", "api_url": url,
        "articles": len(records), "response_sha256": hashlib.sha256(raw).hexdigest(),
        "dataset_card": "https://huggingface.co/datasets/wikimedia/wikipedia",
        "dataset_card_licenses": ["cc-by-sa-3.0", "gfdl"],
        "purpose": "Manual review only; not added to training data",
        "language_note": "Persian (fa); not verified as Afghan Dari",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    review = ["# Persian Wikipedia sample for AMC review", "",
              "Five articles, in dataset order; this is not a representative or approved training corpus.",
              "Persian is not automatically Afghan Dari. Check language, content, cleanup, and attribution.",
              "Source dataset: https://huggingface.co/datasets/wikimedia/wikipedia", ""]
    for row in records:
        review.extend([f"## {row['title']}", "", f"Source: {row['url']}", "",
                       f"Article ID: {row['id']}", "", row["text"], ""])
        print(f"{row['title']}: {len(row['text']):,} characters")
    (OUT / "REVIEW.md").write_text("\n".join(review), encoding="utf-8")
    print(f"Saved {len(records)} articles for review: {OUT / 'REVIEW.md'}")
    print("No training run started. Existing training files and checkpoints are unchanged.")


if __name__ == "__main__":
    main()
