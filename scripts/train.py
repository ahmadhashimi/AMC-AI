"""End-to-end autoregressive pretraining from random initial weights."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from amc_model.model import AMCTransformer, ModelConfig
from amc_model.tokenization import get_tokenizer


def read_tokens(path, tokenizer):
    text = Path(path).read_text(encoding="utf-8")
    ids = tokenizer.encode(text) + [tokenizer.eos_id]
    return torch.tensor(ids, dtype=torch.long)


def sample_batch(data, block_size, batch_size, device):
    if len(data) <= block_size + 1:
        raise ValueError("Corpus must contain more than block_size + 1 tokens")
    starts = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in starts]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in starts]).to(device)
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
    p.add_argument("--config", type=Path, default=ROOT / "configs/tiny.json")
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=ROOT / "outputs/amc_tiny.pt")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    tokenizer = get_tokenizer(args.tokenizer, args.tokenizer_path)
    cfg_dict = json.loads(args.config.read_text(encoding="utf-8"))
    cfg_dict["vocab_size"] = tokenizer.vocab_size
    cfg = ModelConfig(**cfg_dict)
    if args.block_size > cfg.context_length:
        raise SystemExit("--block-size must not exceed model context_length")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    train_ids = read_tokens(args.train, tokenizer)
    val_ids = read_tokens(args.val, tokenizer)
    model = AMCTransformer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print("Device:", device, "Train tokens:", len(train_ids), "Val tokens:", len(val_ids))
    print("Random-initialized trainable parameters:", model.count_parameters())
    print("No pretrained checkpoint loaded.")
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
    }, args.out)
    print("Checkpoint:", args.out, "Elapsed seconds:", round(time.time() - start, 2))
    print("This smoke-test model is not suitable for answering real student questions.")


if __name__ == "__main__":
    main()
