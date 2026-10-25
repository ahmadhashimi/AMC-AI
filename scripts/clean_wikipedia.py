"""Preview conservative Wikipedia cleanup; --write saves a separate dataset."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import shutil

from tqdm import tqdm


SOURCE = Path("data/processed/wikipedia_fa_v1")
OUTPUT = Path("data/processed/wikipedia_fa_clean_v4")
TAIL_HEADINGS = {"منابع", "منبع", "پانویس", "پانویس‌ها", "پیوند به بیرون",
                 "پیوندهای بیرونی", "جستارهای وابسته"}


def remove_empty_prose_parentheses(match):
    run = match.group()
    # Keep grouped parentheses: these may encode trees or mathematical objects.
    if run.count("(") + run.count(")") > 2:
        return run
    return re.sub(r"[ \t]*\([ \t]*\)", "", run)


def clean(text):
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t\u00a0]+", " ", line).strip()
        if line.strip("= ") in TAIL_HEADINGS:
            break
        lines.append(line)
    result = re.sub(r"[ \t]*[()][() \t]*", remove_empty_prose_parentheses,
                    "\n".join(lines))
    lines = result.splitlines()
    while lines and (not lines[-1].strip() or
                     lines[-1].strip("= ") == "نگارخانه"):
        lines.pop()
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    # Reject unresolved extraction markup rather than guessing its meaning.
    if any(marker in result for marker in ("{{", "}}", "[[", "]]", "<ref", "</ref")):
        return None, "unresolved_markup"
    if not result:
        return None, "empty_after_cleanup"
    return result, "kept"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if not args.write:
        with (SOURCE / "train.jsonl").open(encoding="utf-8") as src:
            first = json.loads(next(src))
        print("First training article:", first["title"])
        print("Decision:", clean(first["text"])[1])
        chosen = set(random.Random(42).sample(range(9779), 5))
        counts = Counter()
        with (SOURCE / "val.jsonl").open(encoding="utf-8") as src:
            for index, line in enumerate(src):
                article = json.loads(line)
                text, reason = clean(article["text"])
                counts[reason] += 1
                if text is not None and text != article["text"]:
                    counts["changed"] += 1
                if index in chosen:
                    print("\nARTICLE:", article["title"])
                    print("BEFORE:\n", article["text"][:1000])
                    print("AFTER:\n", text if text is not None else reason)
        print("\nValidation preview counts:", dict(counts))
        print("Preview only. No dataset files changed.")
        return

    if OUTPUT.exists():
        raise SystemExit(f"Preserving existing output: {OUTPUT}")
    required = sum((SOURCE / f"{s}.jsonl").stat().st_size for s in ("train", "val", "test"))
    if shutil.disk_usage(SOURCE).free < required + 1024**3:
        raise SystemExit("Need room for a separate dataset plus 1 GiB reserve.")
    OUTPUT.mkdir()
    report = {"source": str(SOURCE), "splits": {},
              "policy": "Trim whitespace; remove tail at exact reference/navigation heading; remove isolated empty parentheses while preserving grouped parentheses; remove repeated trailing empty gallery headings; reject unresolved markup; exact dedup after cleaning. No minimum length filter.",
              "limitations": "No factual repair, dialect filtering, or near-duplicate detection. Split membership preserved; later exact duplicates dropped (train, val, test order)."}
    seen = set()
    for split in ("train", "val", "test"):
        counts = Counter()
        with (SOURCE / f"{split}.jsonl").open(encoding="utf-8") as src, (OUTPUT / f"{split}.jsonl.partial").open("x", encoding="utf-8") as dst:
            for line in tqdm(src, desc=f"Cleaning {split}", unit="article"):
                counts["input"] += 1
                article = json.loads(line)
                text, reason = clean(article["text"])
                if text is None:
                    counts[reason] += 1
                    continue
                digest = hashlib.sha256(text.encode("utf-8")).digest()
                if digest in seen:
                    counts["duplicate_after_cleanup"] += 1
                    continue
                seen.add(digest)
                counts["changed"] += text != article["text"]
                article["text"] = text
                dst.write(json.dumps(article, ensure_ascii=False) + "\n")
                counts["kept"] += 1
        report["splits"][split] = dict(counts)
    for split in ("train", "val", "test"):
        (OUTPUT / f"{split}.jsonl.partial").rename(OUTPUT / f"{split}.jsonl")
    (OUTPUT / "cleaning_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Saved cleaned text only. Existing tokenizer and token files unchanged.")


if __name__ == "__main__":
    main()
