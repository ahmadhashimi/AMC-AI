# AMC — Technical Handoff

Prepared 2026-10-24 for a second technical advisor (ChatGPT / Codex) joining the project.
Revision 3 — adds the Afghan-domain measurement and the cross-language knowledge transfer objective.
Revision 2 added the evaluation and financial safety system.
Status labels used throughout: **[IMPLEMENTED]** code exists and has been run, **[TESTED]** verified with
an explicit check or measurement, **[PLANNED]** decided but not built, **[NOT STARTED]**, **[UNKNOWN]**.

Nothing in this document is projected or aspirational unless labelled as such.

---

## 1. Project vision

AMC is a nonprofit, open-source effort to build a language model for **Afghan Dari (primary), Pashto
(secondary) and English (support)**, intended for free education of Afghan children — in particular the
girls barred from secondary school and university since 2022.

Design constraints that follow from the users:
- Must run **offline** on cheap hardware (unreliable electricity and internet).
- Must speak **Afghan Dari**, not Iranian Persian. These are mutually intelligible but lexically distinct,
  and large models default to Iranian usage.
- Must be **open source**, including the data recipe. Owner decision 2026-10-24, superseding an earlier
  "private weights" goal.

**The stated goal is a model trained from scratch.** Section 8 documents why that conflicts with available
data, which is the central unresolved issue in the project.

---

## 2. Current progress

### [TESTED] Engineering baseline
- 11 unit tests pass: `python -m unittest discover -s tests` (causal masking, padding-target exclusion, epoch
  coverage, optimizer-resume determinism, single-writer lock, monitoring, LR schedule, tokenizer round-trip,
  budget ceiling, budget carry-forward across resume).
- 8 dashboard assessment cases pass: `node tests/test_assessment.cjs`.
- `scripts/verify_baseline.py` checks checkpoint and tokenizer hashes against `docs/research_baseline_v1.json`.

### [TESTED] Trained model
A 1,660,032-parameter model trained to **step 94,000** on Persian Wikipedia. Sampled validation loss 4.10,
**plateaued** (flat across the last ~10,000 steps, measured). Generations are fluent-looking Persian with no
meaning, and drift to Iranian topics. Scored by `scripts/eval_checkpoint.py`: real-word rate 96.7%
(clean Persian reference 98.5%).

### [TESTED] PDF extraction, two defects root-caused and fixed
1. `page.get_text('text', sort=True)` reverses word order on every right-to-left line. The **unsorted**
   character stream is already in logical order.
2. Each lam-alef ligature is emitted as a **zero-width ALEF immediately before its LAM**, so `بالای`
   extracted as `باالی`. Measured confirmation: zero `لا` sequences across 118 pages before repair; 95
   ligatures repaired after, with non-space character count conserved exactly (42,122 before and after).
Verified by eye against 150 dpi page renders for prose (page 55) and two-column poetry (page 89).

### [TESTED] Afghan Dari vs Iranian Persian dialect detector
`scripts/eval_dialect.py`, 16 marker-word pairs (مکتب/مدرسه, صنف/کلاس, پوهنتون/دانشگاه, شفاخانه/بیمارستان …).
Measured separation:
- Curriculum corpus: 1,716 Afghan vs 8 Iranian markers → **99.5% Afghan**
- Persian Wikipedia: 6,371 Afghan vs 85,399 Iranian → **6.9% Afghan**
No public equivalent is known to exist. This is the project's most novel artifact.

### [IMPLEMENTED] Curriculum corpus
54 of 223 Dari books (grades 1–6) downloaded and extracted; extraction halted by owner, resumable.

### [NOT STARTED]
Joint Dari/Pashto/English tokenizer; the 6B-token corpus; any model above 1.66M parameters; any fine-tuning
run; speech; vision; instruction tuning.

### [TESTED] Evaluation and financial safety system (built 2026-10-24, revision 2)

Four capability tiers, reported separately and never conflated:

| Tier | Question | Automatable |
| --- | --- | --- |
| 1 Pipeline | Does training mechanically work? | yes |
| 2 Language | Does it model the language? | yes |
| 3 Instruction | Does it answer what was asked? | yes |
| 4 Education | Would it help an Afghan child? | **no — native-speaker rubric only** |

**Measured baseline** (`outputs/scorecards/baseline_94000.json`, step 94,000, 1.66M parameters):

| Tier | Metric | Value | Reference |
| --- | --- | --- | --- |
| 1 | weights finite | PASS | — |
| 2 | **Dari held-out perplexity** | **255.7** | random ≈ 8,192 |
| 2 | Science held-out perplexity | 200.4 | |
| 2 | Pashto held-out perplexity | 390.9 | |
| 2 | Wikipedia validation (training distribution) | ≈ 60 | **4× easier than Dari — quantifies the Iranian bias** |
| 2 | real-word rate | 94.4% | clean Persian 98.5% |
| 2 | distinct 4-gram / longest repeat | 0.985 / 3 | no degeneration |
| 3 | arithmetic accuracy | **0.0%** (35/60 produced a number) | — |
| 4 | educational usefulness | not assessed | — |

**Sealed held-out sets** — `data/eval/`, built by `scripts/build_eval_sets.py`, hashed in `manifest.json`.
Whole books are held out, not random pages, because pages within a book share vocabulary.

| Set | Rows | Status |
| --- | --- | --- |
| `dari_heldout` | 44 | sealed |
| `science_heldout` | 387 | sealed |
| `pashto_heldout` | 147 | sealed |
| `math_probes` | 200 | sealed, deterministic arithmetic with known answers |
| `english_heldout` | **0** | **NOT AVAILABLE — recorded, not silently skipped** |

Books excluded from every future training corpus: `G1-Dr-Dari.pdf`, `G2-Dr-Dari_compressed.pdf`,
`G5-Dr-Science.pdf`, `G5-Dr-Social_Science.pdf`, `G6-Dr-Science.pdf`, `G6-Dr-Social_Science_2.pdf`.

**Tooling added:**
- `scripts/eval_suite.py` — all four tiers, deterministic (seed 123, fixed prompts, temperature 0.8;
  **greedy for arithmetic**), writes a JSON scorecard.
- `scripts/compare_scorecards.py` — compares scorecards, flags regressions by metric-specific thresholds,
  renders `outputs/scorecards/dashboard.html`.
- `scripts/eval_dialect.py` — Afghan vs Iranian marker scoring (validated 99.5% / 6.9%).
- `src/amc_model/budget.py` — spend ceiling with wall-clock accounting that **carries across resume**;
  stops the run after a clean checkpoint save; also stops after 3 consecutive failures.
  Wired in as `train_epochs.py --budget-usd X --gpu-rate Y`. Two unit tests.
- `train_epochs.py` keeps `best.pt` separately whenever sampled validation improves.

**GO / HOLD / STOP criteria** are written in `docs/EVALUATION.md` for four decision points (pre-flight,
30 minutes, first checkpoint evaluation, run completion), with numeric thresholds. Headline gate for scaling:
Dari perplexity < 100, real-word rate ≥ 95%, **Afghan dialect share > 0.90**, no flagged regression, and a
native-speaker rubric mean ≥ 2.5 with neither *Afghan register* nor *Factual honesty* below 2.0.

### [TESTED] Afghan-domain measurement — revision 3, the project's most consequential result

`scripts/research_af_domains.py`, one streaming pass over **30,000 FineWeb-2 `fas_Arab` documents**
(`docs/research_af_domains.json`):

| Measurement | Result |
| --- | --- |
| `.af` TLD documents | 21 of 30,000 — **0.07%** |
| **Named Afghan publishers on other TLDs** | **63 — three times more than `.af`** |
| Afghan documents total | 84 of 30,000 — **0.28%** |
| Afghan character share | 0.23% |
| Dialect of the Afghan sample | **83.3% Afghan** (5 Afghan vs 1 Iranian markers, 3,611 chars) |
| **Linear projection to the full 58,887,122-document subset** | **≈165,000 documents ≈ 165M tokens** |

Afghan publishers found outside `.af`: `da.azadiradio.com` (36), `bbc.com`, `avapress.com`,
`afghanpaper.com`, `pajhwok.com`, `ariananews.co`, `khaama.com`, `etilaatroz.com`, `jomhornews.com`,
`afghanembassy.net`, `dw.com`. Inside `.af`: `tkg.af`, `president.gov.af`, `mod.gov.af`, `mohe.gov.af`,
`supremecourt.gov.af`, `bakhtarnews.com.af`, `aihrc.org.af`.

**Why this matters.** The obvious approach — filter on the `.af` TLD — returns 21 documents and would lead
an engineer to conclude Afghan web text does not exist. **Three quarters of Afghan content is on `.com`.**

**What it changes.** Afghan share of a hypothetical 6B-token corpus moves from 0.015% (curriculum only) to
**2.75%** at 1×, or **≈11%** at the 4× repetition that is standard practice. The step-94,000 model that
produced Iranian output trained on roughly 0.4% Afghan content.

**Confidence — read this before quoting the number.** The projection rests on **84 Afghan documents** from a
single **unshuffled** pass over the head of one config. The named-publisher list is hand-collected and
certainly incomplete, so the true yield is likely higher, but the sampling error on 84 documents is wide.
**This figure is provisional and should be superseded by a 200k–500k document rerun**, not quoted as settled.

Domain is also only a proxy for dialect: an `.af` site may publish Iranian Persian, and an Afghan author may
publish on a `.com`. The 83.3% dialect score on the selected sample supports the proxy but rests on 3,611
characters.

### [PLANNED] Cross-language knowledge transfer — core objective, added revision 3

Full plan in `docs/CROSS_LANGUAGE_TRANSFER.md`. AMC is to learn technical knowledge from English sources and
express it in Dari and Pashto. **This does not change the from-scratch requirement** — transfer is a property
of the data mixture, not of initialisation.

Headline metric: `transfer_gap = accuracy(English question) − accuracy(Dari question)`, reported beside
factual correctness and language quality as three separate scores.

**Scale caveat recorded in the plan:** cross-lingual transfer is weak below roughly 1B parameters, so the
30M pilot tests the pipeline and cannot demonstrate this objective. The required held-out sets
(`transfer_qa_dari`, `transfer_qa_pashto`, `transfer_qa_english`, `terminology_probe`) are **NOT BUILT**.

---

## 3. Model architecture

`src/amc_model/model.py` — dense decoder-only Transformer, written explicitly for inspection.

- Pre-norm blocks: `x = x + attn(ln1(x))`, `x = x + mlp(ln2(x))`
- Attention: single fused QKV projection, `F.scaled_dot_product_attention(..., is_causal=True)`
- MLP: Linear(d, 4d) → GELU → Linear(4d, d) → Dropout
- Learned absolute position embeddings (`nn.Embedding(context_length, d_model)`)
- **Tied** token-embedding and output weights
- Init: normal(0, 0.02); block output projections additionally scaled by `0.02/sqrt(2*n_layers)`
- Loss: `F.cross_entropy` with `ignore_index=-100` for padding targets

### Configurations

| File | layers | d_model | heads | context | vocab | parameters | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `configs/tiny.json` | 3 | 128 | 4 | 128 | 8192 at load | **1,660,032** | [TESTED] trained to step 94,000 |
| `configs/medium.json` | 8 | 512 | 8 | 512 | 8192 at load | **29,676,544** | [TESTED] forward pass only, never trained |

`vocab_size` in the config files is overridden at load from the tokenizer (`scripts/train_epochs.py:70`).

### Tokenizer
- `artifacts/wikipedia_fa_v1/tokenizer.json`, byte-level BPE, **8,192 entries**
- SHA256 `c626eff71a6e14d015fe8cc3ed4a085818201d6701aadf7e8e0228b67776395c`
- Specials: `<unk>`=0, `<eos>`=1, `<pad>`=2
- Trained on Persian Wikipedia only
- **[TESTED] Known deficiency:** 3.31 chars/token on Persian but **1.74 on Pashto** — Pashto costs double,
  halving effective context. A joint tokenizer is a prerequisite for any multilingual run.

---

## 4. Training from zero — confirmation

**Confirmed: all weights are randomly initialized.** `AMCTransformer.__init__` applies `_init_weights` to
every module; no checkpoint, no pretrained embedding, no distillation. No Llama/Qwen/GPT weights are
loaded anywhere in the training path.

**Dependencies that are pretrained or externally derived:**
- The BPE tokenizer is *trained by this project* on Persian Wikipedia — not pretrained externally, but it is
  derived from Iranian Persian text, which biases subword merges toward Iranian orthography.
- `scripts/train_qwen_dari.py` **[IMPLEMENTED, NEVER RUN]** references `Qwen/Qwen3-4B`. It is a parallel
  fine-tuning path, not part of the from-scratch pipeline. If from-scratch purity matters, this file is the
  only place a pretrained model appears.

---

## 5. Datasets

| # | Dataset | Language | Size | Licence | Status |
| --- | --- | --- | --- | --- | --- |
| 1 | Afghan curriculum books (MoE) | Dari | 54 books, 4,119 cleaned pages, **2,301,253 chars ≈ 0.70M tokens** | **Redistribution rights UNVERIFIED** | [TESTED] extracted, cleaned, dialect-verified 99.5% Afghan |
| 2 | Persian Wikipedia `20231101.fa` | Iranian Persian | 955,823 articles, **219,761,824 train tokens** + 2,265,432 val | CC BY-SA | [TESTED] cleaned v4, tokenized, fingerprint-checked |
| 3 | Pashto Wikipedia `20231101.ps` | Pashto | 20,529 articles, 63,982,936 chars ≈ 19.4M tokens | CC BY-SA | [IMPLEMENTED] downloaded, **not processed** |
| 4 | FinePDFs `fas_Arab` sample | Iranian Persian | 100 docs, 1.6M chars | ODC-BY | [TESTED] screened, **recommended against** — Iranian academic OCR |
| 5 | FineWeb-2 `fas_Arab` | Iranian Persian | 58,887,122 docs / 102.4 GB | ODC-BY | [NOT STARTED] |
| 6 | FineWeb-2 `pbt_Arab` | Pashto | 646,505 docs / 0.9 GB ≈ 250M tokens | ODC-BY | [NOT STARTED] |
| 7 | FineWeb-Edu `sample-10BT` | English | 9,672,101 docs / 28.5 GB | ODC-BY | [NOT STARTED] |

Sizes for 5–7 were read from the Hugging Face size API on 2026-10-24; token figures are estimates at
~3.3 chars/token (Persian/Pashto) and ~4 (English) unless marked as measured.

### Data quality, measured
- Curriculum Dari word validity: **92.3%** of words appear in a Persian Wikipedia reference vocabulary
  (clean Persian control: 98.5%). The gap is partly genuine Afghan vocabulary absent from Iranian Wikipedia.
- 3 of 54 books (G1/G2/G3 Maths) use fonts with **no Unicode mapping** and extract as Latin-Extended noise.
  They are excluded, not repaired. OCR would be required.
- Cleaning removed: 390 pages in broken books, 411 low-text, 239 with replacement characters, 65
  contents/index pages, 56 exact duplicates. Punctuation moved to the correct line end on 17,089 lines.
- **[UNKNOWN]** Whether line *order within a page* is always correct. Individual lines verified correct;
  owner reports whole pages read as gibberish, which would indicate a page-assembly ordering defect.
  **This is unresolved and matters.**

---

## 6. Training pipeline

```
download_wikipedia.py  → prepare_wikipedia.py → clean_wikipedia.py --write → prepare_tokens.py → train_epochs.py
build_curriculum_corpus.py → clean_curriculum_corpus.py → build_finetune_dataset.py
fetch_corpus.py (HF streaming) → prepare_corpus_tokens.py → train_epochs.py
```

**Tokenization/storage:** uint16 little-endian `.bin` files plus `metadata.json`. `load_token_files()`
validates dtype, byte size, tokenizer SHA256, EOS end-marker and maximum token ID before training, and
memory-maps the files.

**Batching:** `EpochBatches` shuffles fixed-size windows so every next-token target is visited exactly once
per epoch; only the final short window is padded, with target `-100`.

**Training** (`scripts/train_epochs.py`): AdamW, gradient clipping at 1.0, cosine LR with warmup
(`src/amc_model/schedule.py`), optional AdamW weight decay on matrices only, optional bf16 autocast (CUDA
only). All new options default to prior behaviour so older runs still resume.

**Checkpointing:** every 1,000 steps via temp-file-then-replace. Checkpoint carries model, optimizer, epoch,
batch cursor, RNG state, config and dataset/tokenizer fingerprints; `--resume` verifies them.
**`best.pt` is kept separately whenever sampled validation improves** [IMPLEMENTED, not yet exercised in a run].

**Monitoring:** `RunMonitor` appends `metrics.jsonl` and atomically replaces `status.json`;
`scripts/training_dashboard.py` serves a read-only local dashboard on port 8765.

**Evaluation:** `scripts/eval_checkpoint.py` produces a JSON scorecard — step, parameter count, real-word
rate, dialect Afghan share, and generations from 5 fixed prompts. `scripts/eval_dialect.py` scores any text.

---

## 7. Infrastructure

- **Development:** macOS (Darwin 25.5.0, Apple Silicon), Python 3.14.7 in `.venv`, torch 2.14.1,
  transformers 5.20.2, trl 1.16.0, peft 0.21.2.
- **[TESTED] Local throughput, 29.7M model, batch 4 × 512 ctx:** CPU 0.63 steps/s, Apple GPU (MPS)
  0.60 steps/s — MPS provides no speedup. One epoch locally ≈ 47 hours. Local training is not viable.
- **Cloud:** RunPod, connected via MCP and verified. Official plugin `runpod@runpod v1.8.0` installed.
  Zero pods created, **$0.00 spent to date**.
- **[TESTED] Live prices (2026-10-24, community cloud):** RTX 3090 24 GB **$0.22/hr** (in stock),
  A100 SXM 80 GB $1.39/hr, RTX 5090 $0.69/hr. RTX 4090 and A40 out of stock at time of reading.
- **Estimated cost** of a 300M-parameter run on ~6B tokens: 40–100 GPU-hours ≈ **$40–120**. This is an
  estimate, not measured.
- **Code distribution:** repo is public at `github.com/ahmadhashimi/AMC-AI`, branch
  `research/afghan-corpus`, commit `f83e8c0`. A pod can clone it with no credentials.
- **[UNKNOWN]** How trained weights would be retrieved from a pod. The RunPod MCP cannot transfer files;
  options are runpodctl, a Jupyter port, or pushing to the Hugging Face Hub.

---

## 8. Current problems

**P1 — [MATERIALLY IMPROVED in revision 3] Afghan text at pretraining scale.**
The revision-3 measurement raises reachable Afghan text from 0.9M to an estimated **165M tokens**, which is
enough for an Afghan-*sounding* model but still not enough for a from-scratch model that knows physics. The
paragraph below describes the position before that measurement and remains true for the knowledge problem.

**The original statement of the contradiction:**
Measured: 0.70M Afghan tokens held. A 300M model needs ~6,000M tokens; even a 30M model needs ~600M.
All large Persian corpora are **Iranian** (Wikipedia 6.9% Afghan by our own detector). Therefore any
from-scratch model trained on available data will be an Iranian Persian model. This was demonstrated, not
predicted: the step-94,000 checkpoint generates text about Tehran and Iranian universities.
**Unresolved decision:** accept an Iranian-leaning from-scratch model, spend weeks collecting Afghan text, or
adapt an existing open model on Afghan-only data. The owner has rejected the third option on sovereignty
grounds and has restated the from-scratch requirement.

**P2 — Page-assembly ordering is unverified.** See §5. Individual lines are correct Dari; the owner reports
pages reading as gibberish. If confirmed, the fix is to emit sentence/paragraph units and discard page
structure, which costs nothing and sidesteps the problem.

**P3 — Tokenizer is Persian-only.** Pashto costs 2× tokens. Must be retrained before any multilingual run.

**P4 — 3 Maths books unreadable** (fonts without Unicode maps). OCR required or accept the loss.

**P5 — Untested code on the critical path.** `prepare_corpus_tokens.py` and `train_qwen_dari.py` have never
been executed. `fetch_corpus.py` passed a 4-source smoke test after one bug fix. `eval_suite.py`,
`build_eval_sets.py`, `compare_scorecards.py` and `budget.py` have all been run successfully [TESTED].

**P6 — Curriculum redistribution rights unverified.** Mitigation chosen: publish the recipe (scripts, URLs,
SHA256) rather than the files.

**P7 — Residual extraction defects:** words split across line breaks by ZWNJ are left as-is; tables and
equations are not reconstructed; 18.5% of pages carry review flags.

---

## 9. Development roadmap

**Completed [TESTED]:** engineering baseline and tests; 1.66M model trained and evaluated; PDF extraction
root-causes and fixes; dialect detector; curriculum extraction (54/223 books); cleaning pipeline; evaluation
harness; RunPod integration; trainer upgrades (fused attention, cosine LR, bf16, weight decay, best-checkpoint).

**In progress:** none — all background jobs stopped at owner request.

**Next [PLANNED]:** finish the remaining 169 books; `.af` domain filter experiment on FineWeb-2; joint
tokenizer; corpus assembly; first scaled training run; publish dataset and dialect detector.

**Later [NOT STARTED]:** instruction tuning; speech (ASR/TTS — assessed as higher impact than text given
~37% adult literacy); Dari/Pashto OCR; vision; any medical capability.

---

## 10. Immediate priorities

Priority 0 is now satisfied: evaluation and spend controls exist, so a paid run can be judged and bounded.
The proposed 30M pilot is costed at **$1.30–2.00** (RTX 3090 at $0.22/hr, 6–9 hours) with a recommended
ceiling of `--budget-usd 10` against the owner's stated $50 maximum. **It is not authorised and no money
has been spent; total RunPod spend remains $0.00.**


1. **[DONE revision 3] The Afghan-domain experiment.** Result above: ≈165M tokens reachable, and a TLD-only
   filter misses 75% of it. **Next action: rerun at 200k–500k documents to replace the provisional figure**,
   and extend the named-publisher list, which is the main lever on yield.
2. **Resolve P2 (page ordering)** by having a Dari reader confirm whether whole pages read correctly. If not,
   rebuild the dataset from sentences instead of pages.
3. **Finish extracting the remaining 169 books** — free, unattended, resumable, triples the Afghan corpus.
4. **Train the joint Dari/Pashto/English tokenizer** (~32k vocab) — free, blocks everything multilingual.
5. **Decide P1 explicitly and record it in `docs/PROGRESS.md`.** Every other engineering decision depends on it.

---

## What a New Technical Advisor (ChatGPT / Codex) Needs to Know to Continue AMC Development

1. **The single most important fact: the project's stated goal (from scratch) and its data reality are in
   conflict.** 0.70M Afghan tokens exist against ~6,000M needed. This is measured, not estimated. Every
   productive conversation about AMC starts by acknowledging this rather than proposing more compute.

2. **Do not propose "just get more data" without naming a source.** The team has already enumerated what
   exists: Persian web (Iranian), Pashto web (250M tokens), Afghan curriculum (~3M tokens when complete),
   Afghan news (uncollected). The `.af` filter is the only untested lead that could change the arithmetic.

3. **The dialect detector is the project's quality gate.** Any proposal must be evaluable by it. A model that
   scores below ~90% Afghan on `eval_dialect.py` has not met the project's purpose, regardless of its loss.

5. **The valuable artifacts are the data and the tooling, not the model.** The extraction fixes (reversed
   reading order, zero-width-ALEF ligature) and the dialect detector are genuinely novel. The 1.66M model is
   a pipeline proof.

6. **Honesty discipline is a project requirement**, encoded in `CLAUDE.md` and `docs/DATA_POLICY.md`:
   every data artifact carries `approved_for_training: false` plus an explicit `limitations` string until a
   human gate passes; heuristic flags never constitute approval; falling loss is not coherence; no silent
   repair of text. Please preserve these conventions.

7. **Verify before asserting.** Commands: `.venv/bin/python -m unittest discover -s tests -v`,
   `node tests/test_assessment.cjs`, `.venv/bin/python -u scripts/verify_baseline.py`. Always run from the
   repository root with the explicit `.venv/bin/python` — the user's shell aliases `python`.

8. **Never report a tier-4 claim from a tier-2 number.** The scorecard carries an explicit warning field for
   this. "Loss went down" is tier 1 evidence that the optimiser works, nothing more. The project has a
   measured example of why: the step-94,000 model has respectable loss and generates Iranian text about
   Tehran with 0% arithmetic accuracy.

9. **What is NOT known** and should not be guessed: whether the 165M projection survives a larger sample;
   whether page-level line ordering is correct; how weights get off a RunPod pod; whether the Ministry will
   grant redistribution rights; what Qwen3-4B's baseline Dari ability is (never measured).
