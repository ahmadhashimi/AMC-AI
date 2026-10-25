# AMC progress and session handoff

Updated: 2026-10-24. **This is the authoritative continuation file.** Read it first after a context reset or in a new Codex session. Keep current decisions, outcomes, blockers and next actions here; other documents are supporting detail and may describe earlier stages.

## Goal and immediate priority

Build an Afghan-focused AMC model, ultimately with Dari/Pashto/English, speech and vision, and private weights. Current work is a small, from-scratch text research baseline. No general assistant, medical, vision, speech or MoE capability has been demonstrated.

Immediate priority: verify and preserve AMC's engineering baseline, investigate generalization honestly, then prepare **one** Afghan curriculum book with reviewed extraction. Do not scale downloads or training before the corresponding quality gate passes.

## Current verdict

- **Engineering checks PASS:** token/checkpoint fingerprints, finite weights, causal masking, ignored padding targets, full epoch target coverage, deterministic synthetic optimizer resume, single-writer lock and dashboard status checks.
- **Language/reasoning quality NOT established:** narrow memorization and single-fact controls pass; multi-fact transfer fails. Wikipedia generations remain incoherent in inspected samples.
- **One-book extraction BLOCKED FOR TRAINING:** the PDF downloaded and page text extracted, but visual comparison caught reversed word ordering and missing glyphs. No textbook tokens or textbook training were created.
- No existing baseline checkpoint was replaced by the new research experiment. No new commit or push was performed.

## Workspace and operational boundaries

- Project: `/Users/ahmadhashimi/Downloads/amc-dari-foundation`.
- Run commands from this directory using `.venv/bin/python` explicitly. The user's shell alias can override plain `python`.
- User wants visible progress bars and plain explanations. Keep original data/checkpoints and use separate experiment folders.
- User explicitly requested continuing training after the handoff update. Resuming the existing Wikipedia epoch is now authorized. Bulk downloads, publishing, deletion and unrelated new large experiments remain outside this request.
- Disk was about 1.2 GiB free before this trial. Check current free space; do not delete user data or bypass the reserve check.
- Many local changes are uncommitted, including previously staged files with later unstaged edits. Do not reset, clean or stash them. Git status is the authority.
- Source repo: https://github.com/ahmadhashimi/AMC-AI.git . Earlier initial commit was pushed; latest stabilization/research work is local. Weights/data/outputs remain Git-ignored.
- Latest action: launched `.venv/bin/python -u scripts/train_epochs.py --resume` after checking no AMC trainer/dashboard process was running. Starting checkpoint: step 48,886, epoch 1, validation loss 4.252895760536194. Read `outputs/wikipedia_v4_epochs/status.json` for live state; process state may change after this note. Do not start a duplicate trainer. Dashboard was not restarted in this action.

## Stable model and tokenizer

`src/amc_model/model.py`: educational dense Transformer, 3 layers, 4 heads, width 128, context 128, tied embedding/output weights, dropout 0. **1,660,032 trainable parameters** with the BPE tokenizer. No pretrained checkpoint is loaded for the research controls.

Tokenizer: `artifacts/wikipedia_fa_v1/tokenizer.json`, vocabulary 8,192; unknown 0, EOS 1, padding 2.

Tokenizer SHA256:
`c626eff71a6e14d015fe8cc3ed4a085818201d6701aadf7e8e0228b67776395c`

Frozen baseline manifest: `docs/research_baseline_v1.json`. It records checkpoint/config/tokenizer fingerprints and environment versions; it is small metadata, not a weight copy. Verify against it without replacing it:

```bash
.venv/bin/python -u scripts/verify_baseline.py
```

Last verification passed for all three preserved baselines:

| Baseline | Location | Verified outcome |
| --- | --- | --- |
| Wikipedia | `outputs/wikipedia_baselines/step_48886.pt` | Step 48,886, finite weights, preserved hash; not coherent assistant output |
| Four-sentence memorization | `outputs/research/sanity_seed42/model.pt` | 4/4 exact generations |
| Single-fact context | `outputs/research/context_seed42/model.pt` | 48/48 validation answers |

Wikipedia frozen checkpoint SHA256:
`38efad735f76636f946493cb7ebb3f8f34ea2d3d9718f26988e21a6536aec83d`

## Wikipedia data and training status

Source: `wikimedia/wikipedia`, configuration `20231101.fa`, 979,869 raw articles. Original raw/processed data are preserved. Persian Wikipedia is not a verified Afghan Dari corpus.

Final targeted cleanup is v4, built from original text: trims whitespace, removes reference/navigation tails, removes isolated empty parentheses while preserving grouped mathematical parentheses, removes repeated trailing empty gallery headings, rejects unresolved wiki markup and exactly deduplicates across splits. No factual repair, near-duplicate detection, dialect filtering or minimum length filter. Missing text and other quality defects remain possible.

Cleaned JSONL: `data/processed/wikipedia_fa_clean_v4/`.

| Split | Articles | Saved tokens | Binary bytes |
| --- | --- | --- | --- |
| train | 955,823 | 219,761,824 | 439,523,648 |
| val | 9,738 | 2,265,432 | 4,530,864 |
| test | 9,761 | Not tokenized by current preparation | - |

Tokens: `data/tokenized/wikipedia_fa_clean_v4/`, dtype `<u2`, EOS 1. Size/fingerprint/end-marker/sample-decode checks passed. Targeted cleanup checks found zero empty articles, trailing gallery headings and unresolved wiki markers. This does not mean perfect text.

`train_epochs.py` provides full target coverage, progress, optimizer/cursor/RNG resume and safe-save Ctrl+C. Batch 4 x block 128 means 512 targets per full step; one epoch is **429,223 steps** covering 219,761,823 next-token targets. Full epoch completion has not been verified. Do not call the run complete.

Checkpoint/log directory: `outputs/wikipedia_v4_epochs/`. Dashboard: `scripts/training_dashboard.py`, localhost port 8765. Dashboard green means sampled loss improvement, not accurate answers. The new `.trainer.lock` prevents two updated trainers writing the same output directory.

Read-only preflight (does not start training):

```bash
.venv/bin/python -u scripts/train_epochs.py --check-only
```

Resume only after checking no trainer already owns the run:

```bash
.venv/bin/python -u scripts/train_epochs.py --resume
```

Supporting details: `docs/TRAINING_MONITORING.md`.

## Research results: what we actually learned

### 1. Wikipedia audit

Report: `outputs/research/coherence_audit_48886.json`. Same 32 validation batches: fresh loss 9.0239, trained loss 4.3254, trained next-token top-1 accuracy 27.49%; scrambling the trained model's context raises loss to 8.9459. It learned useful context statistics. Fixed prompts still produce incoherent completions. These findings do not identify a unique cause or establish a softmax bottleneck.

### 2. Four-sentence memorization control, completed by user

`scripts/research_sanity.py`, fresh model, 500 steps. Four complete sentences matched by step 100 and through step 500; final training loss 0.009705. Report/progression/model: `outputs/research/sanity_seed42/`. This demonstrates basic fitting and reproduction, not generalization.

### 3. Single-fact generalization control, completed by user

`scripts/research_generalization.py`, fresh model, 1,500 steps. Original artificial color-question task: 96 training examples, 48 validation, 48 initially reserved test; disjoint object/color pairs, all individual words present in training.

At step 250: training 84/96, validation 27/48. By step 500 and through step 1,500: training 96/96, validation 48/48. Final validation loss 0.3038. Reports/data/model: `outputs/research/context_seed42/`.

### 4. Frozen-model stress evaluation, completed locally

Script: `scripts/research_stress.py`; report: `outputs/research/context_stress_seed42.json`. No weight updates. Each condition has 48 examples:

| Condition | Original context model |
| --- | --- |
| Initially reserved test pairs | 48/48 |
| Changed sole color | 48/48 |
| Two facts, query first | 36/48 |
| Two facts, query second | 0/48 |
| Reworded single-fact question | 48/48 |

Concrete weakness: selecting the queried object's color when multiple facts appear. Do not claim a definitive internal mechanism. The formerly reserved test and all stress cases are **now exposed development data**; they cannot be treated as untouched tests in subsequent tuning.

### 5. Balanced two-fact experiment, completed locally; NOT promoted

Script: `scripts/research_binding.py`; separate fresh model, seed 42, 1,500 steps, 1,056 training examples, AdamW learning rate 0.0003, batch 4. Training contains both query positions using only original training object/color pairs. Original weights unchanged. Files: `outputs/research/binding_seed42/{training_data.json,progression.jsonl,report.json,model.pt}`.

Final exact training answers: **1,050/1,056** (99.4%). Exposed development outcomes:

| Condition | Balanced experiment |
| --- | --- |
| Previously reserved test pairs | 3/48 |
| Changed sole color | 48/48 |
| Two facts, query first | 3/48 |
| Two facts, query second | 0/48 |
| Reworded single-fact question | 0/48 |

This attempt did not fix transfer and regressed multiple conditions. Keep the old baseline. Lower training loss and near-perfect training accuracy are insufficient. One seed, different data volume/token budget and adaptive development evaluations mean this is not a causal architecture comparison.

**Next model research action:** inspect train-versus-held-out object/color correlations and construct a balanced diagnostic with disjoint narrative identities, both query positions, varied templates and a newly sealed test set. Pre-register splits and metrics before another run. Do not repeatedly tune on the exposed 48-example test and claim new generalization. Measure exact training fit and development transfer together. No further training was launched after this failed experiment.

## Engineering verification completed

```bash
.venv/bin/python -m unittest discover -s tests -v
node tests/test_assessment.cjs
.venv/bin/python -u scripts/verify_baseline.py
git diff --check
```

Latest results: 8 Python unittest checks passed; all 8 dashboard assessment cases passed; baseline checkpoint/tokenizer hashes and exact controls passed; diff whitespace check passed. Unittest discovery does not execute pytest-style functions in `tests/test_basics.py`; do not misreport that as the whole repository test suite. Tests support the stated engineering checks, not absence of every bug or model quality.

## One-book curriculum pilot

Official catalogs saved under `data/raw/afghan_curriculum/` by `scripts/catalog_curriculum.py`: 532 unique PDF URLs, 571 references. Catalog language/filenames are hints, not verified language labels for every book. Do not download the full collection.

Downloaded one Grade 3 Dari textbook from:
https://moe.gov.af/sites/default/files/2020-02/G3-Dr-Dari.pdf

PDF: `data/raw/afghan_curriculum/dari/grade_03/G3-Dr-Dari.pdf`.
23,234,933 bytes, 118 PDF pages.
SHA256: `e04958b697a0f7a171e3418fe8bb7a1f732e64c48b3ec687f15348d19db1285e`.

Pilot script: `scripts/prepare_curriculum_pilot.py`. It limits downloads to 40 MiB, reserves 1 GiB disk space, refuses existing reports and preserves source/page provenance. PyMuPDF 1.28.2 is installed; Poppler is installed. Do not rerun into existing output just to inspect it.

Extraction folder: `data/extracted/afghan_curriculum/dari/grade_03/G3-Dr-Dari/`:
- `pages.jsonl`: raw sorted PyMuPDF page extraction, 118 rows, 108,611 characters; **not training-ready**.
- `report.json`: source/hash/page flags and manual review findings, `approved_for_training: false`, `BLOCKED_FOR_TRAINING`.
- `page_001.png`, `page_010.png`, `page_020.png`: rendered previews inspected visually.
- `poppler_comparison.txt`: alternate whole-book extraction for comparison, not approved output.

Observed defects: sorted PyMuPDF output reverses Dari word order inside lines; private-use glyphs remain; page 62 contains replacement characters. Native unsorted extraction improves word sequence but puts page components out of order. Poppler page 10 improves reading order but retains broken joins and unmapped glyphs. Neither variant has passed review. Pages 2, 3 and 62 were flagged by automatic low-text/replacement checks; automatic flags alone missed the reading-order problem.

**Next PDF action:** compare native and Poppler output for representative prose, poetry/two-column, exercise and damaged-font pages against rendered originals. Preserve raw text; select a verified reading-order strategy, identify unmapped glyphs explicitly and use reviewed OCR only where needed. No blind string reversal, invented text repair or claim of 100% accuracy. Create training JSONL only after this gate passes. Public access does not establish training/republication permission; source rights remain unverified.

## Other candidates and decisions

- `taesiri/TinyStories-Farsi`: only first 50 rows inspected, pinned revision `87493945c3b0f771d4c115be28e6f5bcd1c1d415`; translation commentary/awkward wording found, median 245 BPE tokens exceeds current 128 context. No full dataset or training approved. Sample: `data/raw/tinystories_fa_review/`.
- `Med-dataset/Med_Dataset`: inspected HF collection is not a demonstrated multimodal training corpus for this project. No medical training started.
- MoE, vision and speech deferred until a measured text baseline and suitable datasets exist. An application router is not an internal neural MoE router.
- Supporting source links and earlier hypotheses are in `docs/COHERENCE_RESEARCH.md`; current results and continuation decisions in this file take precedence over its older pending-status text.

## Continue from here, in order

1. Read this file and inspect `git status --short`; preserve all existing work.
2. Confirm current processes and disk space before launching anything.
3. Address the PDF extraction gate on this one book; keep failed variants out of training.
4. Design the newly sealed, balanced generalization diagnostic before another model run.
5. Update this same file with commands, evidence, outcomes and remaining blockers after each meaningful change. Do not mark AMC coherent or scale-ready based on a passing memorization task or falling loss.

## Latest action: Persian FinePDFs 100-document review

User selected FinePDFs and authorized starting with 100 documents. Completed the bounded preview, not full-corpus download or new training.

- Download command: `.venv/bin/python -u scripts/preview_finepdfs.py`. It preserves existing output, uses five bounded viewer requests, records source metadata and checks repository SHA before/after. Viewer response did not expose its own revision, so this is not proof of an immutable source snapshot; retain the saved sample hash.
- Dataset/config: `HuggingFaceFW/finepdfs`, `fas_Arab`, train. Repository SHA observed: `220bac3acbf07789502c621d2d33952f51ac7f86`. First 100 rows, not a representative random sample. 1,611,749 text characters. No viewer-truncated cells reported; one source document is marked truncated.
- Raw evidence: `data/raw/finepdfs_fa_review/sample.jsonl` and `review.json`. Sample SHA256: `91e1a1b03afc39cc6fe06c8ca514b668f9f81e50ebf760b40a27616784550858`. Dataset card declares ODC-BY; this does not independently settle every source document's rights.
- Review command: `.venv/bin/python -u scripts/review_finepdfs.py`. Output: `data/processed/finepdfs_fa_review_v1/normalized_review.jsonl` and `review.json`. Normalization only decomposes Arabic presentation glyphs and trims trailing whitespace; originals remain preserved. No guessing or reversal repairs.
- Flags: 7 documents with repeated lines, 25 with tables needing reading-order review, 4 with possible Urdu/mixed-language characters, 4 short texts, 1 source-truncated document. Conditions overlap: 35 documents flagged, 65 without these heuristic flags. **65 unflagged does not mean 65 verified clean.**
- Direct inspection found mixed-language repetition in row 0 and reversed table strings in row 2. Professionally produced data still needs inspection; do not claim this entire Persian subset is clean Afghan Dari or suitable for our tiny model.
- No FinePDFs training, tokenizer changes or merge into Wikipedia occurred. Next: inspect candidate prose among the 65 unflagged documents, preserve provenance, explicitly exclude damaged/table/repetition cases, and create a reviewed small dataset with a separately held-out evaluation before training.
- The attempted Wikipedia pause in the earlier interrupted turn did not execute a stop command. Latest status observed during this review: running at step 54500. Check live process/status before any further training; do not assume it was paused.

## FinePDFs candidate screening (read-only, nothing approved)

Script: `scripts/inspect_finepdfs_candidates.py` (`--check-only` measures and prints without writing).
Output: `data/processed/finepdfs_fa_candidates_v1/{report.json,screening.jsonl}`, `approved_for_training: false`.
It measures each of the 100 reviewed rows (characters, BPE tokens with the 8,192 Wikipedia tokenizer, Latin
share of script letters, digit share, share of characters sitting in lines under 25 characters) and adds
triage reasons to the earlier review flags. Thresholds are recorded in the report.

- 60 of 100 rows carry at least one reason; **40 pass every heuristic**. Newly added reasons beyond the
  earlier review: 26 heavy Latin mixing, 11 numeric/tabular, 8 too short, 4 fragmented layout.
- All 40 candidates exceed the 128-token context; median 2,411 tokens, largest 30,060.
- Human reading of the four largest candidates (rows 5, 84, 12, 81): continuous Iranian Persian academic
  prose (law journal, philosophy journal, a recorded lecture), **not Afghan Dari**, and defects survive the
  heuristics — row 12 has reversed title-block line order, row 81 has broken glyph joins and corrupted
  digits (`1911`/`1918` where the Persian calendar year belongs), and intra-word spacing plus kashida is
  pervasive across all of them.
- Assessment: this subset is Iranian academic/religious PDF text, far from Afghan school material, and
  heuristically clean rows still contain OCR corruption. **Recommendation: do not build a FinePDFs training
  set from this sample.** If it is revisited later, it is a secondary adult-register Persian corpus, not an
  Afghan Dari one, and needs per-document human reading.
- No download, tokenizer change, merge into Wikipedia or training was performed by this step.

### Operational note at the time of this screening

Free disk on the data volume was **1.0 GiB**, below the 1 GiB + budget reserve the data scripts require, so
the bounded download scripts will now refuse to run. The Wikipedia trainer (`train_epochs.py --resume`, one
process) was alive at step 62,600, sampled validation loss 4.2014. Checkpoint saves every 1,000 steps need
room; free space before anything else.

## PDF extraction gate: root cause found, reading order verified

Scripts added: `scripts/compare_pdf_extraction.py` (variant comparison + page renders) and
`scripts/extract_curriculum_text.py` (the selected strategy). Both support `--check-only` and refuse to
overwrite existing output. Disk reserve was freed first by deleting the superseded
`data/processed/wikipedia_fa_v1/` (1.8 GiB, user-approved; its `preparation_report.json` is identical to the
copy already kept at `docs/wikipedia_fa_preparation_report.json`, and it is rebuildable from `data/raw`).

### Why the earlier extraction looked reversed

`page.get_text('text', sort=True)` — the mode `prepare_curriculum_pilot.py` used — sorts character spans by
position, which reverses word order inside every right-to-left line. The **unsorted** character stream is
already in correct logical order. Variant evidence for pages 2, 17, 23, 55, 62 and 89 with 150 dpi renders:
`data/extracted/.../G3-Dr-Dari_compare_v1/`. Poppler `-raw` is word-reversed as well; `pdftotext -layout`
keeps the order but wraps every line in bidi control characters.

### The missing glyphs were a decomposed ligature, not a damaged font

Each lam-alef ligature is emitted as a **zero-width ALEF immediately before its LAM**, so `بالای` extracted as
`باالی`, `علاج` as `عالج`. Measured confirmation: the book contained **zero** `لا` sequences across 118 pages.
Swapping exactly that evidence-bearing pair (zero bounding-box width, followed by LAM) repairs 95 ligatures
and introduces 95 `لا` sequences. No lexicon, no guessing, no other character is rewritten.

### Selected strategy, now in `extract_curriculum_text.py`

Unsorted character stream → zero-width-ALEF ligature repair → blocks ordered in 12pt horizontal bands, right
to left. Output: `data/extracted/afghan_curriculum/dari/grade_03/G3-Dr-Dari_repaired_v1/`, 118 pages,
54,237 characters, `approved_for_training: false`.

Verification: non-space character count is 42,122 both before and after repair, so nothing was dropped or
invented. Page 55 prose matches its rendered image line by line; page 89's two-column poem now reads the
right hemistich before the left (`يك جفت كبوتران ابلق` / `استنـد جدا  جدا معلق`), matching the page.

### Still blocking a training JSONL

1. **313 of 2,638 lines begin with sentence punctuation** that belongs at the other end — neutral characters
   land on the wrong side at line boundaries. Needs a bidi-aware edge pass or line joining, then re-checking.
2. Tables (the `لغت‌ها` vocabulary grids) are still emitted cell-by-cell without row structure; exclude or
   reconstruct them deliberately.
3. **513 unmapped private-use glyphs** remain (U+F060 ×464 — the decorative activity bullet — plus seven
   others) and 70 replacement characters. Report them, map only what a human confirms against the renders.
4. Reading order was eyeballed on two pages. Check at least one exercise page, one front-matter page and
   page 62 before approving. Source rights are still unverified; public access is not training permission.
