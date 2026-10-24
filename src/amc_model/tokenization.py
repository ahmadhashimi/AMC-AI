"""A transparent UTF-8 byte tokenizer, plus optional self-trained BPE."""
from pathlib import Path


class ByteTokenizer:
    eos_id = 256
    vocab_size = 257
    kind = "byte"

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, tokens):
        return bytes(int(x) for x in tokens if 0 <= int(x) < 256).decode("utf-8", errors="replace")


class BPETokenizer:
    kind = "bpe"

    def __init__(self, path):
        try:
            from tokenizers import Tokenizer
        except ImportError as err:
            raise RuntimeError("Install optional package: pip install 'tokenizers>=0.20,<1'") from err
        self.tokenizer = Tokenizer.from_file(str(path))
        self.eos_id = self.tokenizer.token_to_id("<eos>")
        if self.eos_id is None:
            raise ValueError("AMC BPE tokenizer requires <eos> special token")
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, tokens):
        return self.tokenizer.decode([int(t) for t in tokens if int(t) != self.eos_id], skip_special_tokens=True)


def get_tokenizer(kind="byte", path=None):
    if kind == "byte":
        return ByteTokenizer()
    if kind == "bpe" and path:
        return BPETokenizer(Path(path))
    raise ValueError("Provide --tokenizer bpe --tokenizer-path path/to/tokenizer.json")
