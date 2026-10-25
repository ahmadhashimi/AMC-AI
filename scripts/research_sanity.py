"""A separate four-sentence memorization control, not a generalization benchmark."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from tqdm import tqdm
from amc_model.model import AMCTransformer, ModelConfig
from amc_model.tokenization import BPETokenizer

EXAMPLES = [
    ("کابل", "کابل پایتخت افغانستان است."),
    ("احمد", "احمد به مکتب رفت. او کتاب خود را خواند."),
    ("آب", "آب برای زندگی انسان ضروری است."),
    ("خورشید", "خورشید یک ستاره است. زمین به دور خورشید می‌گردد."),
]


def make_batch(tokenizer):
    sequences = [tokenizer.encode(text) + [tokenizer.eos_id] for _, text in EXAMPLES]
    length = max(len(ids) - 1 for ids in sequences)
    x = torch.zeros((len(sequences), length), dtype=torch.long)
    y = torch.full_like(x, -100)
    for row, ids in enumerate(sequences):
        x[row, :len(ids)-1] = torch.tensor(ids[:-1])
        y[row, :len(ids)-1] = torch.tensor(ids[1:])
    return x, y


@torch.inference_mode()
def measure(model, x, y):
    model.eval()
    logits, loss = model(x, y)
    mask = y != -100
    accuracy = float((logits.argmax(-1)[mask] == y[mask]).float().mean())
    model.train()
    return float(loss), accuracy


@torch.inference_mode()
def greedy(model, tokenizer, prompt, limit=60):
    model.eval()
    ids = tokenizer.encode(prompt)
    for _ in range(limit):
        logits, _ = model(torch.tensor([ids[-model.cfg.context_length:]]))
        token = int(logits[0, -1].argmax())
        if token == tokenizer.eos_id:
            break
        ids.append(token)
    return tokenizer.decode(ids)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs/research/sanity_seed42")
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    if args.steps < 1 or not 0 < args.lr < 1:
        p.error("Use positive steps and a learning rate between zero and one")
    if not args.check_only and args.out_dir.exists():
        p.error("Output already exists; use a new --out-dir to preserve it")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    tokenizer_path = ROOT / "artifacts/wikipedia_fa_v1/tokenizer.json"
    tokenizer = BPETokenizer(tokenizer_path)
    cfg = ModelConfig(**json.loads((ROOT / "configs/tiny.json").read_text()))
    cfg.vocab_size = tokenizer.vocab_size
    model = AMCTransformer(cfg)
    x, y = make_batch(tokenizer)
    assert x.shape[1] <= cfg.context_length
    baseline_loss, baseline_accuracy = measure(model, x, y)
    print(f"Fresh model: {model.count_parameters():,} parameters; {int((y != -100).sum())} target tokens", flush=True)
    print(f"Initial loss: {baseline_loss:.4f}; teacher-forced accuracy: {baseline_accuracy:.1%}", flush=True)
    print("Memorization test only. No Wikipedia checkpoint loaded or changed.", flush=True)
    if args.check_only:
        print("PASS: batch and forward check only. No training or files written.")
        return
    args.out_dir.mkdir(parents=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    report = {"experiment":"four_sentence_memorization", "generalization_test":False,
              "config":asdict(cfg), "seed":args.seed, "lr":args.lr,
              "tokenizer_sha256":hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
              "baseline":{"loss":baseline_loss,"teacher_forced_accuracy":baseline_accuracy},
              "measurements":[], "outputs":[], "progression":[]}

    def show_progress(step):
        outputs = []
        tqdm.write(f"\nSentence attempts at step {step}")
        for prompt, expected in EXAMPLES:
            generated = greedy(model, tokenizer, prompt, limit=30)
            row = {"prompt":prompt,"expected":expected,"generated":generated,
                   "exact_match":generated == expected}
            outputs.append(row)
            display = "".join(c if c.isprintable() or c in "\n\u200c" else repr(c)[1:-1] for c in generated)
            tqdm.write(f"{'MATCH' if row['exact_match'] else 'LEARNING'} | {prompt} → {display}")
        snapshot = {"step":step,"exact_sentences":sum(row["exact_match"] for row in outputs),"outputs":outputs}
        report["progression"].append(snapshot)
        with (args.out_dir / "progression.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot, ensure_ascii=False)+"\n")
        tqdm.write(f"Exact sentences: {snapshot['exact_sentences']}/4\n")

    show_progress(0)
    started = time.monotonic()
    bar = tqdm(range(1, args.steps + 1), desc="Memorization control", unit="step")
    try:
        for step in bar:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss in memorization control")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step == 1 or step % 50 == 0 or step == args.steps:
                value, accuracy = measure(model, x, y)
                row = {"step":step,"loss":value,"teacher_forced_accuracy":accuracy}
                report["measurements"].append(row)
                with (args.out_dir / "measurements.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row)+"\n")
                bar.set_postfix(loss=f"{value:.4f}", accuracy=f"{accuracy:.1%}")
            if step % 100 == 0 or step == args.steps:
                show_progress(step)
    except KeyboardInterrupt:
        print("Stopped control test; partial measurements preserved. Wikipedia is unchanged.")
        return
    for prompt, expected in EXAMPLES:
        generated = greedy(model, tokenizer, prompt)
        report["outputs"].append({"prompt":prompt,"expected":expected,"generated":generated,
                                  "exact_match":generated == expected})
        print(f"\nPrompt: {prompt}\nExpected: {expected}\nGenerated: {generated}")
    report["elapsed_seconds"] = time.monotonic() - started
    report["all_four_exact"] = all(r["exact_match"] for r in report["outputs"])
    report["interpretation"] = ("Memorization control passed; does not establish unseen-text coherence or rule out every bug."
                                if report["all_four_exact"] else
                                "Memorization control incomplete at this budget; inspect loss, tokenization and optimization before interpreting.")
    (args.out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    torch.save({"model":model.state_dict(), "config":asdict(cfg), "steps":args.steps,
                "tokenizer_kind":"bpe", "tokenizer_path":str(tokenizer_path),
                "seed":args.seed, "experiment":"memorization_control"}, args.out_dir / "model.pt")
    print("\n", report["interpretation"])
    print("Saved report:", args.out_dir / "report.json")


if __name__ == "__main__":
    main()
