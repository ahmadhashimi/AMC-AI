"""Train a new AMC multilingual BPE tokenizer from our own authorized text."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("artifacts/tokenizer.json"))
    p.add_argument("--vocab-size", type=int, default=8192)
    args = p.parse_args()
    if args.out.exists():
        raise SystemExit(f"Tokenizer already exists; preserving it: {args.out}")
    try:
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    except ImportError as exc:
        raise SystemExit("Install tokenizers: pip install 'tokenizers>=0.20,<1'") from exc
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<unk>", "<eos>", "<pad>"],
    )
    if args.corpus.suffix == ".jsonl":
        # Read article text only. Never train on JSON metadata or held-out splits.
        with args.corpus.open(encoding="utf-8") as corpus:
            tokenizer.train_from_iterator(
                (json.loads(line)["text"] for line in corpus), trainer=trainer
            )
    else:
        tokenizer.train([str(args.corpus)], trainer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.out))
    print("Tokenizer saved:", args.out)
    print("Vocabulary size:", tokenizer.get_vocab_size())
    print("Dari sample:", tokenizer.encode("سلام، محصلان افغانستان!", add_special_tokens=False).tokens)


if __name__ == "__main__":
    main()
