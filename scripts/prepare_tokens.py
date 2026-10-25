import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

source = Path("data/processed/wikipedia_fa_clean_v4")
output = Path("data/tokenized/wikipedia_fa_clean_v4")
tokenizer_path = Path("artifacts/wikipedia_fa_v1/tokenizer.json")

preparation = json.loads((source / "cleaning_report.json").read_text(encoding="utf-8"))
expected = {split: {"articles": preparation["splits"][split]["kept"]}
            for split in ("train", "val")}

output.parent.mkdir(parents=True, exist_ok=True)
if output.exists():
    raise SystemExit("Output folder already exists; preserving it.")

# Initial estimate based on the original corpus; actual sizes checked below.
required = 585252810
if shutil.disk_usage(output.parent).free < required + 1024**3:
    raise SystemExit("Need approximately 0.55 GiB for tokens plus 1 GiB free reserve.")

tokenizer = Tokenizer.from_file(str(tokenizer_path))
eos = tokenizer.token_to_id("<eos>")
if eos is None or tokenizer.get_vocab_size() > 65536:
    raise SystemExit("Tokenizer is incompatible with this token format.")

output.mkdir()
report = {
    "source": str(source),
    "tokenizer": str(tokenizer_path),
    "dtype": "<u2",
    "bytes_per_token": 2,
    "eos_id": eos,
    "tokenizer_sha256": hashlib.sha256(
        tokenizer_path.read_bytes()
    ).hexdigest(),
    "splits": {},
}

for split, target in expected.items():
    count = 0
    articles = 0
    partial = output / f"{split}.bin.partial"

    with (source / f"{split}.jsonl").open(encoding="utf-8") as src:
        with partial.open("xb") as dst:
            bar = tqdm(
                src,
                total=target["articles"],
                desc=f"Saving {split}",
                unit="article",
            )
            for articles, line in enumerate(bar, 1):
                if articles % 1000 == 1 and shutil.disk_usage(output).free < 1024**3:
                    raise SystemExit("Stopped to preserve 1 GiB free space; partial files retained.")
                text = json.loads(line)["text"]
                ids = tokenizer.encode(
                    text, add_special_tokens=False
                ).ids
                ids.append(eos)
                dst.write(np.asarray(ids, dtype="<u2").tobytes())
                count += len(ids)

    if articles != target["articles"]:
        raise ValueError(f"{split}: article count differs from cleaning report.")
    if partial.stat().st_size != count * 2:
        raise ValueError(f"{split}: unexpected file size.")

    report["splits"][split] = {
        "articles": articles,
        "tokens": count,
        "bytes": partial.stat().st_size,
    }

for split in expected:
    (output / f"{split}.bin.partial").rename(output / f"{split}.bin")

(output / "metadata.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print("Token files saved:", output)
print(json.dumps(report, indent=2))
