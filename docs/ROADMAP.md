# AMC roadmap

## Core objective added 2026-10-24: cross-language knowledge transfer

AMC learns technical knowledge (programming, mathematics, physics, chemistry, biology, general
knowledge) from high-quality **English** sources and learns to explain it naturally in **Afghan Dari**
and **Pashto**. This is a first-class training objective, planned from the tokenizer upward.

It does **not** change the from-scratch requirement: transfer comes from the data mixture, not from
pretrained weights. Random initialisation stands.

Full plan, datasets, evaluation design and risks: **`docs/CROSS_LANGUAGE_TRANSFER.md`**.

Headline metric: `transfer_gap = accuracy(English question) − accuracy(Dari question)`, reported
alongside factual correctness and language quality as three separate scores.

---

# AMC research roadmap and definition of success

## Step 0: Make every experiment reproducible
Store Git commit hash, seed, model config, tokenizer checksum, data manifest, license review, loss curves, GPU type, training duration, and checkpoint identifiers.

## Stage 1: Tiny learning model (this package)
- Python, PyTorch, randomly initialized layers
- Byte tokenizer for transparent UTF-8 learning; replace with trained multilingual BPE
- Run with *toy original examples only* to prove gradients and checkpoint saving
- Inspect matrices and gradients, understand failure modes
- **Not** suitable for production student answers

## Stage 2: Authentic Dari-English corpus
- Identify public-domain and explicitly licensed textbooks and web corpora
- Filter by source and language. Persian does not automatically equal Afghan Dari
- Retain provenance for each document and license
- Deduplicate at document and paragraph levels; detect personal data
- Retain math formatting and reviewed OCR output
- Train / validation / test split by source document and topic; keep untouched test set
- Native Dari speakers evaluate dialect quality
- Train a representative joint tokenizer, measure tokens per word and script coverage

## Stage 3: 50M–100M model
- Move from toy text to sufficiently diverse corpus
- Train full weights with monitored loss and validation performance
- Compare model against simple baselines, evaluate text generation and token efficiency
- Implement periodic checkpointing and restart-safe cloud storage

## Stage 4: Post-training
- Create reviewed JSON instruction examples with provenance and answer evidence
- Train SFT; evaluate on held-out grade levels and tasks
- Build an educational assistant that retrieves and cites textbook passages (RAG)
- Use preference optimization later, after reliable human comparative labels

## Stage 5: Scale and publish
- Design a feasible, costed 4B–7B pretraining program *only after dataset audits*
- LitGPT / FSDP / distributed GPU infrastructure, separate validation and red-teaming
- Publish weights only if the entire data/model license review allows it
- Release LoRA/QLoRA recipes and low-cost hosted learning workflows

## Student evaluation benchmarks
- Dari spelling/grammar and regional usage
- Mathematics with verified numerical answers
- Science explanations aligned with trusted textbooks
- English–Dari translation with reviewed reference answers
- Responses grounded in retrieved textbooks with citations
- Robustness to dialect variants, hallucinations, data leakage, and bias

Never claim the model has passed these benchmarks without measured scores.
