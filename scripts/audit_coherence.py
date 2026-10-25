"""Read-only model diagnostics; compares a checkpoint with fresh random weights."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from tqdm import tqdm
from amc_model.model import AMCTransformer, ModelConfig
from amc_model.token_data import EpochBatches
from amc_model.tokenization import BPETokenizer


@torch.inference_mode()
def evaluate(model, data, batches):
    model.eval()
    loader = EpochBatches(data, model.cfg.context_length, 4, 1729)
    total = correct = count = 0
    scrambled_total = 0
    hidden = []
    handle = model.final_norm.register_forward_hook(lambda module, inputs, output: hidden.append(output.detach().reshape(-1, model.cfg.d_model)))
    try:
        for i in tqdm(range(min(batches, len(loader))), desc="Measuring", unit="batch"):
            x, y = loader.batch(i)
            x, y = torch.from_numpy(x), torch.from_numpy(y)
            valid = y != -100
            logits, loss = model(x, y)
            # Keep only genuine tokens and only the normal-context activations.
            hidden[-1] = hidden[-1][valid.reshape(-1)].clone()
            n = int(valid.sum())
            count += n
            total += loss.item() * n
            correct += int((logits.argmax(-1)[valid] == y[valid]).sum())
            generator = torch.Generator().manual_seed(123 + i)
            _, shuffled_loss = model(x[:, torch.randperm(x.shape[1], generator=generator)], y)
            hidden.pop()  # Remove activations produced by the scrambled-context call.
            scrambled_total += shuffled_loss.item() * n
    finally:
        handle.remove()
    h = torch.cat(hidden).double()
    unit = torch.nn.functional.normalize(h, dim=1)
    cosine = float((unit.sum(0).square().sum() - len(unit)) / (len(unit) * (len(unit) - 1)))
    centered = h - h.mean(0)
    eigen = torch.linalg.eigvalsh(centered.T @ centered / (len(h) - 1)).clamp_min(0)
    # Participation ratio measures spread across covariance dimensions, not coherence.
    participation = float(eigen.sum().square() / eigen.square().sum())
    loss = total / count
    return {"evaluated_target_tokens":count, "loss":loss, "perplexity":math.exp(loss),
            "top1_next_token_accuracy":correct/count, "scrambled_context_loss":scrambled_total/count,
            "centered_hidden_participation_ratio":participation,
            "average_pairwise_hidden_cosine":cosine}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/wikipedia_baselines/step_48886.pt")
    p.add_argument("--batches", type=int, default=32)
    p.add_argument("--out", type=Path, default=ROOT / "outputs/research/coherence_audit_48886.json")
    args = p.parse_args()
    if not 1 <= args.batches <= 200:
        p.error("Use 1–200 diagnostic batches")
    if args.out.exists():
        p.error("Report already exists; choose another --out path")
    torch.set_num_threads(2)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = ModelConfig(**state["config"])
    tokenizer_path = Path(state["tokenizer_path"])
    tokenizer = BPETokenizer(tokenizer_path)
    meta = state["token_metadata"]
    if hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() != meta["tokenizer_sha256"]:
        raise ValueError("Checkpoint/tokenizer fingerprint mismatch")
    data_path = ROOT / "data/tokenized/wikipedia_fa_clean_v4/val.bin"
    if data_path.stat().st_size != meta["splits"]["val"]["bytes"]:
        raise ValueError("Validation file size differs from checkpoint metadata")
    ids = np.memmap(data_path, dtype=meta["dtype"], mode="r")
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if digest != state["identity"]["data_sha256"]["val"]:
        raise ValueError("Validation file fingerprint differs from checkpoint")
    assert cfg.vocab_size == tokenizer.vocab_size
    torch.manual_seed(42)
    fresh = AMCTransformer(cfg)
    trained = AMCTransformer(cfg)
    trained.load_state_dict(state["model"])
    print("Saved checkpoint step:", state["steps"], flush=True)
    report = {"checkpoint":str(args.checkpoint), "step":state["steps"],
              "checkpoint_sha256":hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "validation_sha256":digest, "config":state["config"],
              "total_parameters":trained.count_parameters(),
              "token_embedding_parameters":trained.token_embeddings.weight.numel(),
              "transformer_block_parameters":sum(p.numel() for p in trained.blocks.parameters()),
              "fresh":evaluate(fresh, ids, args.batches),
              "trained":evaluate(trained, ids, args.batches),
              "interpretation_limits":"Fixed validation sample, not full-set evaluation. Hidden participation ratio and cosine are descriptive; they do not establish softmax bottleneck causality. No weights updated."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps(report, indent=2))
    print("Saved diagnostic report:", args.out)


if __name__ == "__main__":
    main()
