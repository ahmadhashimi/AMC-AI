"""Extract selectable text from authorized PDFs; do not hallucinate OCR output."""
import argparse
from pathlib import Path
import fitz


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    pdfs = sorted(args.input.glob("*.pdf")) if args.input.is_dir() else [args.input]
    if not pdfs:
        raise SystemExit("No PDFs found. Supply legally authorized PDFs.")
    args.output.mkdir(parents=True, exist_ok=True)
    for pdf in pdfs:
        with fitz.open(pdf) as doc:
            pages = [page.get_text("text", sort=True).strip() for page in doc]
        extracted = "\n\n".join(p for p in pages if p)
        dest = args.output / (pdf.stem + ".txt")
        dest.write_text(extracted, encoding="utf-8")
        if len(extracted) < 80:
            print(f"WARNING: {pdf.name} yielded little text. Might be a scan requiring reviewed OCR.")
        print(f"{pdf.name} -> {dest} ({len(extracted)} characters)")


if __name__ == "__main__":
    main()
