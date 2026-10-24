"""Conservative Unicode cleanup. Human review is mandatory for Dari dialect/OCR."""
import argparse
from pathlib import Path
import re
import unicodedata


def clean(text):
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\ufeff", "").replace("\u0640", "")  # BOM, tatweel
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Avoid aggressive Persian/Dari letter substitutions; they can alter names.
    text = "\n".join(re.sub(r"[\t ]+", " ", line).strip() for line in text.split("\n"))
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip() + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    files = sorted(args.input.glob("*.txt")) if args.input.is_dir() else [args.input]
    if not files:
        raise SystemExit("No .txt files found")
    args.output.mkdir(parents=True, exist_ok=True)
    seen = set()
    for file in files:
        cleaned = clean(file.read_text(encoding="utf-8"))
        if cleaned in seen:
            print("Skipping exact duplicate document:", file.name)
            continue
        seen.add(cleaned)
        dest = args.output / file.name
        dest.write_text(cleaned, encoding="utf-8")
        print(file.name, "=>", dest)
    print("Next: MANUALLY review Dari/OCR, licensing, and document-level train/val/test split.")


if __name__ == "__main__":
    main()
