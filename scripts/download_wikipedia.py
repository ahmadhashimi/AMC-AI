"""Download the complete November 2023 Persian Wikipedia subset as JSONL."""
import argparse
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw/wikipedia_fa/articles.jsonl"))
    args = parser.parse_args()
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    if args.out.exists() or partial.exists():
        parser.error("Output or partial download already exists; preserving it")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading the complete 20231101.fa subset...", flush=True)
    dataset = load_dataset("wikimedia/wikipedia", "20231101.fa", split="train")
    if len(dataset) != 979869:
        raise ValueError("Unexpected source article count")
    dataset.to_json(str(partial), force_ascii=False)
    partial.rename(args.out)
    print(f"Saved {len(dataset):,} articles to {args.out}")


if __name__ == "__main__":
    main()
