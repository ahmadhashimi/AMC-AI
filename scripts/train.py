"""End-to-end autoregressive pretraining from random initial weights."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from amc_model.model import AMCTransformer, ModelConfig
from amc_model.tokenization import get_tokenizer
from amc_model.token_data import load_token_files


def read_tokens(path, tokenizer):
    text = Path(path).read_text(encoding="utf-8")
    ids = tokenizer.encode(text) + [tokenizer.eos_id]
    return torch.tensor(ids, dtype=torch.long)


def sample_batch(data, block_size, batch_size, device):
    if len(data) < block_size + 1:
        raise ValueError("Corpus must contain at least block_size + 1 tokens")
    starts = torch.randint(len(data) - block_size, (batch_size,)).tolist()
    if isinstance(data, np.ndarray):
        # Only copy the sampled windows, never the whole memory-mapped file.
        windows = np.stack([data[i:i+block_size+1] for i in starts]).astype(np.int64)
        tokens = torch.from_numpy(windows)
        x, y = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
    else:
        x = torch.stack([data[i:i+block_size] for i in starts])
        y = torch.stack([data[i+1:i+block_size+1] for i in starts])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def evaluate(model, data, block_size, batch_size, device, batches=3):
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = sample_batch(data, block_size, batch_size, device)
        _, loss = model(x, y)
        losses.append(float(loss))
    model.train()
    return sum(losses) / len(losses)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, default=ROOT / "data/sample/train.txt")
    p.add_argument("--val", type=Path, default=ROOT / "data/sample/val.txt")
    p.add_argument("--tokenizer", choices=("byte", "bpe"), default="byte")
    p.add_argument("--tokenizer-path", type=Path)
    p.add_argument("--token-dir", type=Path, help="Prepared .bin files and metadata.json; selects their BPE tokenizer")
    p.add_argument("--check-only", action="store_true", help="Validate data and one forward pass; no training or checkpoint")
    p.add_argument("--config", type=Path, default=ROOT / "configs/tiny.json")
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=ROOT / "outputs/amc_tiny.pt")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.block_size) < 1:
        p.error("steps, batch-size and block-size must be positive")
    if not args.check_only and args.out.exists():
        p.error("Output checkpoint already exists; choose a new --out path")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    token_metadata = None
    if args.token_dir:
        print("Checking token files and tokenizer...", flush=True)
        data, tokenizer, args.tokenizer_path, token_metadata = load_token_files(args.token_dir, ROOT)
        args.tokenizer = "bpe"
        train_ids, val_ids = data["train"], data["val"]
    else:
        tokenizer = get_tokenizer(args.tokenizer, args.tokenizer_path)
        train_ids = read_tokens(args.train, tokenizer)
        val_ids = read_tokens(args.val, tokenizer)
    cfg_dict = json.loads(args.config.read_text(encoding="utf-8"))
    cfg_dict["vocab_size"] = tokenizer.vocab_size
    cfg = ModelConfig(**cfg_dict)
    if args.block_size > cfg.context_length:
        raise SystemExit("--block-size must not exceed model context_length")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model = AMCTransformer(cfg).to(device)
    print("Device:", device, "Train tokens:", len(train_ids), "Val tokens:", len(val_ids))
    print("Random-initialized trainable parameters:", model.count_parameters())
    print("No pretrained checkpoint loaded.")
    if args.check_only:
        model.eval()
        x, y = sample_batch(train_ids, args.block_size, args.batch_size, device)
        assert torch.equal(x[:, 1:], y[:, :-1])
        with torch.inference_mode():
            logits, loss = model(x, y)
        if not torch.isfinite(loss):
            raise ValueError("Forward-pass loss is not finite")
        print("Batch shape:", tuple(x.shape), "Logits:", tuple(logits.shape))
        print("Random-model check loss:", round(loss.item(), 4))
        print("PASS: data loading and forward pass. No weights updated; no checkpoint saved.")
        return
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        x, y = sample_batch(train_ids, args.block_size, args.batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step == 1 or step % max(1, args.steps // 5) == 0:
            val_loss = evaluate(model, val_ids, args.block_size, args.batch_size, device)
            print(f"step={step:5d} train_loss={loss.item():.4f} val_loss={val_loss:.4f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": cfg_dict,
        "tokenizer_kind": args.tokenizer,
        "tokenizer_path": str(args.tokenizer_path) if args.tokenizer_path else None,
        "steps": args.steps,
        "seed": args.seed,
        "token_metadata": token_metadata,
    }, args.out)
    print("Checkpoint:", args.out, "Elapsed seconds:", round(time.time() - start, 2))
    print("This smoke-test model is not suitable for answering real student questions.")


if __name__ == "__main__":
    main()
