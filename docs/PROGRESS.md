# AMC progress — 2026-10-24

## Completed

- Educational PyTorch Transformer, attention, gradient flow, AdamW training, checkpoint inspection, and interactive CLI.
- 644,352-parameter byte-tokenizer experiments on eight sample passages: 15 steps and a separate 500-step run. These are pipeline demonstrations, not useful assistants.
- Persian Wikipedia `wikimedia/wikipedia`, configuration `20231101.fa`: 979,869 articles downloaded locally.
- User-run full JSONL check: every record parsed and had string ID, URL, title, and nonempty text. Count matched published dataset metadata. No duplicate IDs reported. This is structural/count validation, not a byte-for-byte source comparison.
- Exact normalized-text deduplication: 128 extra copies removed, leaving 979,741 articles. Near duplicates have not been checked.
- Deterministic article splits: 960,149 train; 9,779 validation; 9,813 test.
- Own ByteLevel BPE tokenizer trained on the training split only: 8,192 vocabulary entries, including 256 byte symbols and three special tokens.
- Reported round-trip check passed for `سلام، پایتخت افغانستان کابل است.`: IDs `[275, 832, 298, 3994, 1947, 4862, 329, 16]` decoded back exactly.
- Publication check: all scripts compiled, both model basic checks passed, and the saved BPE tokenizer round-tripped 100 validation articles plus four Persian/Dari, English, whitespace, numeral, and emoji examples exactly. This verifies lossless encoding for those inputs, not model language quality.

## Current stage

Tokenizer training is complete. A full training/validation token count was started by the user; its final result has not yet been recorded here. No complete token-ID files have been prepared, no formal Wikipedia model baseline has been recorded, and no Transformer has been pretrained on Wikipedia.

The byte checkpoints and the new BPE vocabulary are incompatible. A new model/configuration and measured baseline are needed for the Wikipedia experiment.

## Reproduce the data preparation

Run from the repository root. Use `.venv/bin/python` explicitly if a shell alias overrides `python`.

```bash
.venv/bin/python -m pip install -r requirements-data.txt
.venv/bin/python -u scripts/download_wikipedia.py
.venv/bin/python -u scripts/prepare_wikipedia.py
.venv/bin/python -u scripts/train_tokenizer.py \
  --corpus data/processed/wikipedia_fa_v1/train.jsonl \
  --out artifacts/wikipedia_fa_v1/tokenizer.json \
  --vocab-size 8192
.venv/bin/python -u scripts/count_tokens.py
```

The trained tokenizer is included in Git. Its training command stops if that output already exists; use a new output path for a separate experiment. Download and preparation commands also preserve existing outputs. Check disk space before downloading or creating token files. Count tokens directly from JSONL; do not create an additional full `train.txt` copy.

## Remaining

1. Record token counts and evaluate tokenizer round trips/compression on a fixed validation sample and Dari examples.
2. Add streaming or memory-mapped token preparation and a compatible training loader; the current `scripts/train.py` reads a whole plain-text corpus and is intended for the small learning lab.
3. Choose a model configuration, preserve its random initialization, and record baseline validation loss plus fixed-prompt completions.
4. Run a bounded Wikipedia pretraining pilot with optimizer checkpoints, resume support, and validation measurements.
5. Compare before/after using the same tokenizer, model, and evaluation inputs. Keep the test split reserved for final evaluation.
6. Consider instruction/conversation training after language learning is demonstrated.

## Provenance and repository scope

- Source: https://huggingface.co/datasets/wikimedia/wikipedia
- Source dataset card lists CC BY-SA 3.0 and GFDL. The MIT code license does not replace source-data terms.
- Persian Wikipedia is not a verified Afghan Dari corpus. Native Dari evaluation remains necessary.
- Large raw/processed datasets, token files, environments, and model checkpoints stay local and are ignored by Git. Small source metadata, scripts, the trained tokenizer, and preparation results are tracked.
- Do not interpret tokenizer completion or falling sample training loss as evidence of conversational ability.
