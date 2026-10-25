"""Train complete token-file passes, saving resumable checkpoints."""
import argparse
import atexit
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from tqdm import tqdm
from amc_model.model import AMCTransformer, ModelConfig
from amc_model.token_data import EpochBatches, load_token_files
from amc_model.monitoring import RunMonitor
from amc_model.run_lock import RunLock
from amc_model.schedule import learning_rate_at

# Values that reproduce the behaviour of runs started before these knobs
# existed. Only a changed value joins the resume identity, so those runs
# still resume with this script.
TUNING_DEFAULTS = {"warmup": 0, "min_lr_ratio": 1.0, "weight_decay": 0.0, "amp": "off"}


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def evaluate(model, ids, block_size, batch_size, device, amp=nullcontext()):
    # A fixed sample for monitoring, not a full validation-set evaluation.
    batches = EpochBatches(ids, block_size, batch_size, 1729)
    model.eval()
    total = count = 0
    for i in range(min(20, len(batches))):
        x, y = batches.batch(i)
        n = int((y != -100).sum())
        with amp:
            _, loss = model(torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))
        total += loss.item() * n
        count += n
    model.train()
    return total / count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--token-dir", type=Path, default=ROOT / "data/tokenized/wikipedia_fa_clean_v4")
    p.add_argument("--config", type=Path, default=ROOT / "configs/tiny.json")
    p.add_argument("--epochs", type=int, default=1, help="Total target epochs, including already completed epochs")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=TUNING_DEFAULTS["warmup"],
                   help="Steps of linear learning-rate warmup; 0 keeps the old constant rate")
    p.add_argument("--min-lr-ratio", type=float, default=TUNING_DEFAULTS["min_lr_ratio"],
                   help="Cosine-decay floor as a fraction of --lr; 1.0 disables decay")
    p.add_argument("--weight-decay", type=float, default=TUNING_DEFAULTS["weight_decay"],
                   help="AdamW decay on matrices only; biases and norms stay undecayed")
    p.add_argument("--amp", choices=("off", "bf16"), default=TUNING_DEFAULTS["amp"],
                   help="bf16 autocast; CUDA only")
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs/wikipedia_v4_epochs")
    p.add_argument("--resume", action="store_true", help="Resume out-dir/last.pt, including optimizer and cursor")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.block_size, args.save_every) < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        p.error("Epochs, batch dimensions, save interval and learning rate must be positive")
    if args.warmup < 0 or not 0.0 <= args.min_lr_ratio <= 1.0 or args.weight_decay < 0:
        p.error("Warmup and weight decay cannot be negative; min-lr-ratio must be within 0..1")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.amp == "bf16" and device != "cuda":
        p.error("bf16 autocast needs a CUDA device")
    amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if args.amp == "bf16" else nullcontext())
    print("Validating token files and computing data fingerprints...", flush=True)
    data, tokenizer, tokenizer_path, metadata = load_token_files(args.token_dir, ROOT)
    cfg_dict = json.loads(args.config.read_text(encoding="utf-8"))
    cfg_dict["vocab_size"] = tokenizer.vocab_size
    cfg = ModelConfig(**cfg_dict)
    if args.block_size > cfg.context_length:
        p.error("block-size exceeds model context length")
    identity = {"config": asdict(cfg), "batch_size": args.batch_size,
                "block_size": args.block_size, "lr": args.lr, "seed": args.seed,
                "tokenizer_sha256": metadata["tokenizer_sha256"],
                "data_sha256": {s: fingerprint(args.token_dir / f"{s}.bin") for s in ("train", "val")}}
    tuning = {"warmup": args.warmup, "min_lr_ratio": args.min_lr_ratio,
              "weight_decay": args.weight_decay, "amp": args.amp}
    identity.update({k: v for k, v in tuning.items() if v != TUNING_DEFAULTS[k]})
    model = AMCTransformer(cfg).to(device)
    batches = EpochBatches(data["train"], args.block_size, args.batch_size, args.seed)
    print(f"Device: {device}; parameters: {model.count_parameters():,}")
    print(f"One epoch: {len(data['train']) - 1:,} next-token targets; {len(batches):,} steps")
    if args.check_only:
        x, y = batches.batch(len(batches) - 1)
        with torch.inference_mode():
            _, loss = model(torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))
        assert torch.isfinite(loss)
        print("PASS: epoch batch forward pass. No training or checkpoint writes.")
        return
    checkpoint = args.out_dir / "last.pt"
    if not args.resume and args.out_dir.exists():
        p.error("Output directory exists; use --resume or choose a new --out-dir")
    if args.weight_decay:
        groups = [{"params": [p for p in model.parameters() if p.dim() >= 2],
                   "weight_decay": args.weight_decay},
                  {"params": [p for p in model.parameters() if p.dim() < 2], "weight_decay": 0.0}]
        optimizer = torch.optim.AdamW(groups, lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    epoch = cursor = step = 0
    if args.resume:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["identity"] != identity:
            p.error("Resume data, tokenizer, config or training settings do not match")
        if state["device"] != device:
            p.error("Use the same device type when resuming this run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        epoch, cursor, step = state["epoch"], state["next_batch"], state["steps"]
        torch.set_rng_state(state["rng"])
        if device == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        elif device == "mps":
            torch.mps.set_rng_state(state["mps_rng"])
        print(f"Resumed step {step:,}, epoch {epoch + 1}, batch {cursor:,}")
    if epoch >= args.epochs:
        print("Requested epochs already completed.")
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_lock = RunLock(args.out_dir)
    atexit.register(run_lock.close)
    stop = False
    monitor = RunMonitor(args.out_dir, {
        "epochs": args.epochs, "steps_per_epoch": len(batches),
        "train_tokens": len(data["train"]), "val_tokens": len(data["val"]),
        "parameters": model.count_parameters(), "device": device,
        "batch_size": args.batch_size, "block_size": args.block_size,
        "learning_rate": args.lr, "save_every": args.save_every,
        "warmup": args.warmup, "min_lr_ratio": args.min_lr_ratio,
        "weight_decay": args.weight_decay, "amp": args.amp,
        "tokenizer_vocab": tokenizer.vocab_size,
    })
    last_saved_step = step if args.resume else None
    last_saved_time = checkpoint.stat().st_mtime if args.resume else None
    train_loss = val_loss = rate = None

    def emit(event, status="running"):
        monitor.emit(event, status=status, step=step, epoch_completed=epoch,
                     next_batch=cursor, train_loss=train_loss,
                     validation_loss=val_loss, steps_per_second=rate,
                     checkpoint_step=last_saved_step, checkpoint_time=last_saved_time)

    original_excepthook = sys.excepthook

    def report_error(kind, error, traceback):
        try:
            monitor.emit("error", status="error", step=step, epoch_completed=epoch,
                         next_batch=cursor, error=f"{kind.__name__}: {error}",
                         checkpoint_step=last_saved_step, checkpoint_time=last_saved_time)
        except Exception:
            pass  # Preserve the original traceback even if disk writes fail.
        original_excepthook(kind, error, traceback)

    sys.excepthook = report_error

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    def save():
        nonlocal last_saved_step, last_saved_time
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "config": asdict(cfg), "identity": identity, "epoch": epoch,
                 "next_batch": cursor, "steps": step, "rng": torch.get_rng_state(),
                 "device": device, "tokenizer_kind": "bpe", "tokenizer_path": str(tokenizer_path),
                 "token_metadata": metadata}
        if device == "cuda":
            state["cuda_rng"] = torch.cuda.get_rng_state_all()
        elif device == "mps":
            state["mps_rng"] = torch.mps.get_rng_state()
        partial = checkpoint.with_suffix(".pt.partial")
        torch.save(state, partial)
        partial.replace(checkpoint)
        last_saved_step, last_saved_time = step, time.time()
        emit("checkpoint")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    emit("resume" if args.resume else "start")
    print("Training starts now. Ctrl+C requests a safe save after the current step.", flush=True)
    model.train()
    while epoch < args.epochs:
        batches = EpochBatches(data["train"], args.block_size, args.batch_size, args.seed + epoch)
        bar = tqdm(total=len(batches), initial=cursor, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="step")
        interval_start, interval_step = time.monotonic(), step
        while cursor < len(batches):
            x, y = batches.batch(cursor)
            optimizer.zero_grad(set_to_none=True)
            with amp:
                _, loss = model(torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; resume from the last saved checkpoint")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate_at(step, len(batches) * args.epochs, args.lr,
                                               args.warmup, args.min_lr_ratio)
            optimizer.step()
            step += 1
            cursor += 1
            bar.update(1)
            if step == 1 or step % 100 == 0:
                train_loss = loss.item()
                elapsed = time.monotonic() - interval_start
                rate = (step - interval_step) / elapsed if elapsed else None
                interval_start, interval_step = time.monotonic(), step
                bar.set_postfix(loss=f"{loss.item():.4f}")
                emit("train")
            if step % args.save_every == 0:
                save()
                val = evaluate(model, data["val"], args.block_size, args.batch_size, device, amp)
                val_loss = val
                emit("validation")
                tqdm.write(f"Saved step {step:,}; sampled validation loss={val:.4f}")
            if stop:
                save()
                emit("stopped", "paused")
                bar.close()
                print("Stopped safely. Resume with the same command plus --resume.")
                return
        bar.close()
        epoch += 1
        cursor = 0
        save()
    emit("completed", "completed")
    print("Completed full dataset passes. Checkpoint:", checkpoint)


if __name__ == "__main__":
    main()
