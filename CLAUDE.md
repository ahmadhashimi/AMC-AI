# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Read first

`docs/PROGRESS.md` is the authoritative state and handoff file: current verdict, what is approved, what is
blocked, live run status and the ordered next actions. Read it before doing anything, and update it (commands,
evidence, outcome, remaining blockers) after any meaningful change. Other docs are supporting detail and may
describe earlier stages. `README.md` is the public/human-facing overview — its examples call bare `python`,
which is wrong here (see below).

Project: a from-scratch, nonprofit educational Dari/Persian LM research baseline. No pretrained weights are
ever loaded. Current state is a 1.66M-parameter tiny Transformer; language quality is **not** established.

## Commands

Always run from the repository root with the explicit interpreter — the user's shell aliases `python`:

```bash
.venv/bin/python -u scripts/<script>.py      # Python 3.14.7
```

Engineering check set (what "tests pass" means here):

```bash
.venv/bin/python -m unittest discover -s tests -v    # 8 checks: stability + monitoring
.venv/bin/python -m unittest discover -s tests -k causal -v   # single test by substring
node tests/test_assessment.cjs                       # dashboard status-colour logic
.venv/bin/python -u scripts/verify_baseline.py       # checkpoint/tokenizer hashes + exact controls
git diff --check
```

`unittest discover` does **not** execute the pytest-style bare functions in `tests/test_basics.py`, and pytest
is not installed — do not report the unittest run as the whole suite.

Data pipeline (each stage writes a new versioned folder; `prepare_tokens.py` and `prepare_wikipedia.py` have
hardcoded paths, no argparse):

```
download_wikipedia.py → prepare_wikipedia.py → clean_wikipedia.py --write → prepare_tokens.py → train_epochs.py
```

Training on prepared token files:

```bash
.venv/bin/python -u scripts/train_epochs.py --check-only   # read-only preflight, starts nothing
.venv/bin/python -u scripts/train_epochs.py --resume       # restores weights, optimizer, cursor, RNG
.venv/bin/python -u scripts/training_dashboard.py          # second terminal, http://127.0.0.1:8765
```

`--epochs N` is the *total* target epoch count, not additional epochs. Ctrl+C once and wait for the safe-save
message. Details and the epoch-2 gate: `docs/TRAINING_MONITORING.md`.

## Before launching anything

```bash
pgrep -fl train_epochs ; df -h .
```

One trainer per output directory (`src/amc_model/run_lock.py` takes an advisory flock and will refuse a second
writer). Free disk is chronically close to the 1 GiB reserve the data scripts require, so check it rather than
assuming. Never delete user data to make room.

## Architecture

- `src/amc_model/model.py` — `ModelConfig` + dense decoder-only Transformer (causal attention, tied
  embedding/output weights) written out explicitly for teaching; `configs/tiny.json` holds the shape and
  `vocab_size` is overridden from the tokenizer at load.
- `src/amc_model/tokenization.py` — `ByteTokenizer` (no deps) and `BPETokenizer`; `get_tokenizer(kind, path)`.
- `src/amc_model/token_data.py` — `load_token_files()` validates `metadata.json` against the `.bin` files
  (dtype `<u2`, byte size, tokenizer sha256, EOS end marker, max token ID) and memory-maps them;
  `EpochBatches` shuffles fixed windows so every next-token target is visited exactly once per epoch, padding
  only the final short window with target `-100`.
- `src/amc_model/monitoring.py` — `RunMonitor.emit()` appends to `metrics.jsonl` and atomically replaces
  `status.json`; `scripts/training_dashboard.py` and `dashboard/` only read those files, so metrics survive
  without the dashboard. Status-colour heuristics live in `dashboard/assessment.js`.
- Checkpoints embed dataset/tokenizer/config fingerprints; `--resume` verifies them before continuing.
- `scripts/research_*.py` and `scripts/audit_coherence.py` are self-contained controlled experiments writing
  into their own `outputs/research/<name>/` folder. Note in `docs/PROGRESS.md` which evaluation sets have
  already been exposed as development data.

## Conventions for new data/experiment scripts

Follow the existing ones (`preview_finepdfs.py`, `review_finepdfs.py`, `prepare_curriculum_pilot.py`):

- Refuse to overwrite: `if OUT.exists(): raise SystemExit(...)`. Keep raw source under `data/raw/` untouched
  and write derived data to a new `_v<N>` folder. Never overwrite a baseline checkpoint.
- Bound every download (byte cap, row cap) and check the 1 GiB disk reserve first.
- Record provenance: source URL, dataset revision, sha256 of the saved bytes, declared licence.
- Emit `approved_for_training: false` plus an explicit `limitations` string until a human review gate passes.
  Heuristic flags are a triage aid; "unflagged" never means "verified clean". See `docs/DATA_POLICY.md`.
- No silent repair — no string reversal, invented text or guessed glyph mapping.

## Reporting rules

Falling loss, a passing memorization control or a green dashboard do not establish coherence, factual accuracy
or Afghan Dari quality — do not record a run as complete, coherent or scale-ready on that basis. Report what
was measured. Many uncommitted local changes are intentional: never `reset`, `stash` or `clean`, and commit or
push only when asked.
