"""Download a complete November 2023 Wikipedia language subset as JSONL."""
import argparse
from pathlib import Path
import shutil

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
# Expected article counts guard against a silently changed or partial source.
# Source megabytes drive the disk check: the Arrow cache plus the JSONL copy
# need roughly three times the download.
SUBSETS = {"20231101.fa": (979869, "wikipedia_fa", 759),
           "20231101.ps": (20529, "wikipedia_ps", 53)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=sorted(SUBSETS), default="20231101.fa")
    parser.add_argument("--out", type=Path, help="Defaults to data/raw/<subset>/articles.jsonl")
    args = parser.parse_args()
    expected, folder, megabytes = SUBSETS[args.config]
    out = args.out or ROOT / "data/raw" / folder / "articles.jsonl"
    partial = out.with_suffix(out.suffix + ".partial")
    if out.exists() or partial.exists():
        parser.error("Output or partial download already exists; preserving it")
    needed = 1024**3 + 3 * megabytes * 1024**2
    if shutil.disk_usage(ROOT).free < needed:
        raise SystemExit(f"Need a 1 GiB reserve plus {3 * megabytes} MiB for this subset and its cache.")
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading the complete {args.config} subset...", flush=True)
    dataset = load_dataset("wikimedia/wikipedia", args.config, split="train")
    if len(dataset) != expected:
        raise ValueError(f"Expected {expected:,} articles, found {len(dataset):,}")
    dataset.to_json(str(partial), force_ascii=False)
    partial.rename(out)
    print(f"Saved {len(dataset):,} articles to {out}")


if __name__ == "__main__":
    main()
