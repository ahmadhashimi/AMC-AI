"""Fetch a bounded review sample only; never trains or changes Wikipedia data."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request

REPO = "taesiri/TinyStories-Farsi"
REVISION = "87493945c3b0f771d4c115be28e6f5bcd1c1d415"
FILE = "InProgress-TinyStoriesV2-GPT4-train-Translated-To-Farsi-w-Claude-2.0.csv"
DEST = Path("data/raw/tinystories_fa_review")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=50)
    args = p.parse_args()
    if not 1 <= args.limit <= 100:
        p.error("Review limit must be 1–100 rows")
    if DEST.exists():
        raise SystemExit(f"Preserving existing review sample: {DEST}")
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{FILE}"
    # Bound the response to 2 MiB even if the server ignores HTTP Range.
    request = urllib.request.Request(url, headers={"Range":"bytes=0-2097151"})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read(2 * 1024 * 1024)
    # A truncated CSV record may end inside a story; retain only complete rows
    # encountered before the review limit, with strict CSV parsing.
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig", errors="strict")), strict=True)
    if not {"English", "Persian"}.issubset(reader.fieldnames or []):
        raise ValueError(f"Unexpected columns: {reader.fieldnames}")
    rows = []
    for i, row in enumerate(reader):
        if row.get("English") is None or row.get("Persian") is None:
            raise ValueError("Incomplete review row")
        rows.append({"row":i, "English":row["English"], "Persian":row["Persian"]})
        if len(rows) == args.limit:
            break
    if len(rows) != args.limit:
        raise ValueError("Insufficient complete rows in bounded preview; nothing saved")
    flags = {"empty_persian":[], "translator_commentary":[], "latin_letters":[], "very_short_persian":[]}
    for row in rows:
        text = row["Persian"].strip()
        if not text:
            flags["empty_persian"].append(row["row"])
        if re.search(r"متن فارسی|ترجمه[ٔ‌ی ]|به فارسی|با تشکر|با تشكر|Here is|translation", text, re.I):
            flags["translator_commentary"].append(row["row"])
        if re.search(r"[A-Za-z]", text):
            flags["latin_letters"].append(row["row"])
        if len(text) < 100:
            flags["very_short_persian"].append(row["row"])
    report = {"dataset":REPO, "revision":REVISION, "source_file":FILE,
              "declared_license":"cdla-sharing-1.0", "rows":len(rows),
              "sampling":"First N training rows only; not a representative random sample",
              "flags":flags, "decision":"Review only; not approved as clean Afghan Dari training data",
              "limitations":"Heuristic flags are review cues, not proof of errors or translation correctness."}
    DEST.mkdir(parents=True)
    sample = "".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows)
    (DEST / "sample.jsonl").write_text(sample, encoding="utf-8")
    report["sample_sha256"] = hashlib.sha256(sample.encode()).hexdigest()
    (DEST / "review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    for row in rows[:3]:
        print(f"\nROW {row['row']}\nENGLISH: {row['English']}\nPERSIAN: {row['Persian']}")
    print("\nSaved review sample:", DEST / "sample.jsonl")
    print("No full dataset downloaded. No model training started.")


if __name__ == "__main__":
    main()
