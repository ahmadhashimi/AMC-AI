import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from amc_model.model import AMCTransformer, ModelConfig
from amc_model.tokenization import ByteTokenizer


def test_dari_roundtrip():
    t = ByteTokenizer()
    text = "سلام افغانستان! Hello."
    assert t.decode(t.encode(text)) == text


def test_masked_next_token_forward_backward():
    m = AMCTransformer(ModelConfig(vocab_size=257, context_length=16, n_layers=1, n_heads=2, d_model=32))
    x = torch.randint(0, 256, (2, 10))
    logits, loss = m(x, x)
    assert logits.shape == (2, 10, 257)
    loss.backward()
    assert m.blocks[0].attention.qkv.weight.grad is not None
