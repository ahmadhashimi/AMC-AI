"""Inspect actual trained tensors and gradient flow in AMC's model."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from amc_model.model import AMCTransformer, ModelConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = AMCTransformer(ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    print("Model parameters:", f"{model.count_parameters():,}")
    print("Layers:", len(model.blocks))
    print("Weight examples:")
    for name, param in model.named_parameters():
        print(f"{name:60s} shape={list(param.shape)} avg_abs={param.data.abs().mean().item():.6f}")
    x = torch.randint(0, model.cfg.vocab_size, (2, min(16, model.cfg.context_length)))
    y = torch.randint(0, model.cfg.vocab_size, x.shape)
    _, loss = model(x, y)
    loss.backward()
    print("Test loss:", float(loss))
    print("Input embedding gradient norm:", float(model.token_embeddings.weight.grad.norm()))
    print("QKV attention gradient norm:", float(model.blocks[0].attention.qkv.weight.grad.norm()))


if __name__ == "__main__":
    main()
