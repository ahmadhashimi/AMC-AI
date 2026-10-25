"""Train a new AMC multilingual BPE tokenizer from our own authorized text."""
import argparse
import itertools
import json
from pathlib import Path


def documents(spec, max_docs):
    """Yield text from one `path` or `path=REPEAT` source, capped at max_docs."""
    path, _, repeat = spec.partition("=")
    path = Path(path)
    repeat = int(repeat) if repeat else 1
    if repeat < 1:
        raise SystemExit(f"Repeat count must be positive: {spec}")
    if path.suffix == ".jsonl":
        # Read article text only. Never train on JSON metadata or held-out splits.
        with path.open(encoding="utf-8") as corpus:
            texts = [json.loads(line)["text"] for line in
                     (itertools.islice(corpus, max_docs) if max_docs else corpus)]
    else:
        texts = [path.read_text(encoding="utf-8")]
    characters = sum(len(t) for t in texts)
    print(f"  {path}: {len(texts):,} documents x{repeat} = {characters * repeat / 1e6:.1f}M characters")
    for _ in range(repeat):
        yield from texts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", nargs="+", required=True,
                   help="One or more sources as path or path=REPEAT; REPEAT oversamples a small language")
    p.add_argument("--max-docs", type=int, default=0, help="Cap documents read per source; 0 reads all")
    p.add_argument("--out", type=Path, default=Path("artifacts/tokenizer.json"))
    p.add_argument("--vocab-size", type=int, default=8192)
    args = p.parse_args()
    if args.out.exists():
        raise SystemExit(f"Tokenizer already exists; preserving it: {args.out}")
    try:
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    except ImportError as exc:
        raise SystemExit("Install tokenizers: pip install \'tokenizers>=0.20,<1\'") from exc
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<unk>", "<eos>", "<pad>"],
    )
    print("Reading sources:")
    stream = itertools.chain.from_iterable(documents(s, args.max_docs) for s in args.corpus)
    tokenizer.train_from_iterator(stream, trainer=trainer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.out))
    print("Tokenizer saved:", args.out)
    print("Vocabulary size:", tokenizer.get_vocab_size())
    print("Dari sample:", tokenizer.encode("سلام، محصلان افغانستان!", add_special_tokens=False).tokens)
    print("Pashto sample:", tokenizer.encode("ښه راغلاست، زما ګرانو زده کوونکو!", add_special_tokens=False).tokens)


if __name__ == "__main__":
    main()
