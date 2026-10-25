# AMC Dari Foundation Model 🇦🇫

**A nonprofit, open-source educational AI research project for Afghan students.**

**Model status:** working educational prototype with tiny sample-training experiments. Our Persian Wikipedia BPE tokenizer is trained (8,192 tokens); **Wikipedia model pretraining has not started**. No Claude, OpenAI, Qwen, or other pretrained checkpoint is used.

See [current progress and reproduction commands](docs/PROGRESS.md). The trained tokenizer is in `artifacts/wikipedia_fa_v1/tokenizer.json`; large datasets and model checkpoints are kept outside Git tracking.

For full Wikipedia passes, checkpoint resume, and the local live dashboard, see [training monitoring](docs/TRAINING_MONITORING.md). Start the dashboard in a separate terminal with `.venv/bin/python -u scripts/training_dashboard.py`, then open `http://127.0.0.1:8765`. Runs started before the metrics update need one safe stop/resume to begin recording loss history.

## Why two tracks?

1. **AMC Learning Lab (this repo):** a readable PyTorch decoder-only Transformer with its attention, embeddings, loss, backpropagation, checkpointing, generation, and inspectable gradients. Begin here.
2. **LitGPT (upstream training engine):** fork https://github.com/Lightning-AI/litgpt to study and eventually scale distributed pretraining. LitGPT is *software*, not pretrained model weights. See `docs/LITGPT.md`.

## Quickstart: a complete from-scratch experiment

Requires Python 3.10+ and PyTorch. A cloud GPU is recommended for bigger experiments; the tiny demonstration also works on CPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Model architecture + data are included. NO pretrained weights downloaded.
python scripts/train.py --steps 15 --batch-size 4 --block-size 64
python scripts/inspect_model.py --checkpoint outputs/amc_tiny.pt
python scripts/generate.py --checkpoint outputs/amc_tiny.pt --prompt 'آب'
```

This short run is for validating the pipeline, not high-quality answers. With just the sample data, generated text will be poor.

To interact with the saved model in your terminal:

```bash
.venv/bin/python scripts/generate.py --checkpoint outputs/amc_tiny.pt --interactive
```

Type a Dari or English prompt and press Enter. AMC prints a continuation including your prompt; each turn is independent. Type `/quit` to exit. This loads the existing checkpoint without training or changing data.

## Train AMC's OWN multilingual BPE tokenizer

```bash
pip install 'tokenizers>=0.20,<1'
python scripts/train_tokenizer.py --corpus data/sample/train.txt --out artifacts/tokenizer.json --vocab-size 1024
python scripts/train.py --tokenizer bpe --tokenizer-path artifacts/tokenizer.json --steps 15 --batch-size 4 --block-size 64 --out outputs/amc_bpe.pt
```

Both byte-based and BPE tokenization handle UTF-8 Dari and English. The byte tokenizer is transparent and requires no extra package; the BPE tokenizer is a more practical early experiment. Train a real tokenizer only after constructing a representative, licensed training corpus.

## Add textbooks

1. Check copyright and permission for training, record source in `data/manifest.csv` (example format in `data/manifest.example.csv`).
2. Put **authorized** PDFs in `data/raw/` (never commit the raw PDFs).
3. Extract digital PDF text:

```bash
python scripts/extract_pdf.py --input data/raw --output data/extracted
```

4. Manually inspect the output. If files are scans, arrange compliant OCR; the extraction script does **not** silently guess text.
5. Clean the reviewed `.txt` files:

```bash
python scripts/clean_corpus.py --input data/extracted --output data/cleaned
```

6. Split by *source textbook/document*, not random sentences, into `train.txt`, `val.txt`, and `test.txt`, and keep the test set entirely held out. Use only sources with training rights.

## Suggested development roadmap

- **Week 1:** run tiny experiment, inspect loss/gradients, register textbook licenses, extract a few authorized textbooks.
- **Weeks 2–4:** train and evaluate a Dari+English BPE tokenizer, acquire much larger and diverse language corpora, deduplicate and split by source.
- **Later:** scale small model to ~100M, create human-verified SFT examples, evaluate with native Dari speakers; only then consider a 4–7B model and institutional cloud-GPU funding.

## Key paths

- `src/amc_model/model.py` — all Transformer layers written in PyTorch.
- `src/amc_model/tokenization.py` — byte and BPE tokenizers.
- `scripts/train.py` — random initialization → batches → loss → backward → optimizer → checkpoint.
- `scripts/inspect_model.py` — parameter counts and gradient statistics.
- `scripts/generate.py` — token-by-token sampling.
- `scripts/extract_pdf.py` — PDF text extraction.
- `scripts/clean_corpus.py` — conservative text cleaning.
- `scripts/train_tokenizer.py` — train an AMC tokenizer from plain text or JSONL article text.
- `scripts/download_wikipedia.py` — download the full Persian Wikipedia subset.
- `scripts/prepare_wikipedia.py` — normalize, deduplicate, and split Wikipedia articles.
- `scripts/count_tokens.py` — count training/validation tokens with progress bars and a storage estimate.
- `docs/ROADMAP.md` — learning stages, real-data requirements, and evaluation criteria.
- `docs/LITGPT.md` — verified upstream training command and safe migration plan.

## Essential honesty

A 4–7B model trained **from scratch** needs a massive, varied training corpus and expensive GPU compute, well beyond a textbook collection. A tiny demonstration can illustrate learning but cannot provide strong educational answers. We will report measured evaluation results, not claim capabilities before testing.

## License and rights

Code scaffold: MIT license. Model and dataset licensing must be decided separately with clear rights audits. Do not upload copyrighted books, private student records, or audio without proper permission/consent.
