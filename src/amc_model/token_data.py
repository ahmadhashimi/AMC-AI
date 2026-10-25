"""Read prepared token files without copying the corpus into RAM."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .tokenization import BPETokenizer


class EpochBatches:
    """Visit every next-token target once; pad only the final short window."""

    def __init__(self, ids, block_size, batch_size, seed):
        if len(ids) < 2 or min(block_size, batch_size) < 1:
            raise ValueError("Need at least two tokens and positive batch dimensions")
        self.ids, self.block_size, self.batch_size = ids, block_size, batch_size
        self.windows = (len(ids) - 1 + block_size - 1) // block_size
        self.order = np.random.default_rng(seed).permutation(self.windows)

    def __len__(self):
        return (self.windows + self.batch_size - 1) // self.batch_size

    def batch(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        windows = self.order[index*self.batch_size:(index+1)*self.batch_size]
        x = np.zeros((len(windows), self.block_size), dtype=np.int64)
        y = np.full_like(x, -100)
        for row, window in enumerate(windows):
            start = int(window) * self.block_size
            size = min(self.block_size, len(self.ids) - start - 1)
            x[row, :size] = self.ids[start:start+size]
            y[row, :size] = self.ids[start+1:start+size+1]
        return x, y


def load_token_files(folder, root):
    folder = Path(folder)
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    if metadata["dtype"] != "<u2" or metadata["bytes_per_token"] != 2:
        raise ValueError("Expected little-endian uint16 token files")
    path = Path(metadata["tokenizer"])
    if not path.is_absolute():
        path = Path(root) / path
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["tokenizer_sha256"]:
        raise ValueError("Tokenizer fingerprint differs from token-file metadata")
    tokenizer = BPETokenizer(path)
    if tokenizer.eos_id != metadata["eos_id"]:
        raise ValueError("End marker differs from metadata")
    data = {}
    for split in ("train", "val"):
        info = metadata["splits"][split]
        binary = folder / f"{split}.bin"
        if info["tokens"] < 2 or binary.stat().st_size != info["bytes"] or info["bytes"] != info["tokens"] * 2:
            raise ValueError(f"{split}: invalid token count or file size")
        ids = np.memmap(binary, dtype="<u2", mode="r")
        if int(ids[-1]) != tokenizer.eos_id:
            raise ValueError(f"{split}: missing final end marker")
        # Scan in bounded slices, including IDs far beyond the first sample.
        for start in range(0, len(ids), 1_000_000):
            if int(ids[start:start + 1_000_000].max()) >= tokenizer.vocab_size:
                raise ValueError(f"{split}: token ID outside tokenizer vocabulary")
        data[split] = ids
    return data, tokenizer, path, metadata
