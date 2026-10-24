"""Count tokens and uint16 storage requirements without writing token files."""
import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/wikipedia_fa_v1"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/wikipedia_fa_v1/tokenizer.json"))
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    if tokenizer.get_vocab_size() > 65536:
        parser.error("Vocabulary exceeds uint16 storage capacity")
    if tokenizer.token_to_id("<eos>") is None:
        parser.error("Tokenizer requires an <eos> token")
    report_path = args.data_dir / "preparation_report.json"
    expected = json.loads(report_path.read_text())["counts"] if report_path.exists() else {}
    total = 0
    for split in ("train", "val"):
        count = 0
        articles = 0
        with (args.data_dir / f"{split}.jsonl").open(encoding="utf-8") as corpus:
            bar = tqdm(corpus, total=expected.get(split), desc=split, unit="article")
            for articles, line in enumerate(bar, 1):
                text = json.loads(line)["text"]
                count += len(tokenizer.encode(text, add_special_tokens=False).ids) + 1
                if articles % 1000 == 0:
                    bar.set_postfix(tokens=f"{count:,}")
        if split in expected and articles != expected[split]:
            raise ValueError(f"{split}: article count differs from preparation report")
        print(f"{split}: {articles:,} articles, {count:,} tokens (including one EOS per article)", flush=True)
        total += count
    print(f"Token-file space needed: {total * 2 / 1024**3:.2f} GiB")
    print("No token files written and no model training started.")


if __name__ == "__main__":
    main()
