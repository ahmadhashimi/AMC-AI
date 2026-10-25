# AMC evaluation and financial safety

Written 2026-10-24, before any paid training run. Nothing in this document authorises spending.

## The four tiers — never conflate them

A model can pass every tier below it and still be useless. Reporting must always name the tier.

| Tier | Question | How measured | Automatable |
| --- | --- | --- | --- |
| **1 Pipeline correctness** | Does training mechanically work? | Finite weights and loss, deterministic optimizer resume, epoch coverage, single-writer lock, checkpoint fingerprints | Yes — `unittest`, `verify_baseline.py`, `eval_suite.py` tier 1 |
| **2 Language modelling** | Does it model the language? | Held-out perplexity per language, real-word rate, repetition, script and dialect profile | Yes — `eval_suite.py` tier 2 |
| **3 Instruction following** | Does it answer what was asked? | Arithmetic probes with checkable answers, greedy decoding | Yes — `eval_suite.py` tier 3 |
| **4 Educational usefulness** | Would this help an Afghan child? | Native-speaker rubric below | **No. Never claim tier 4 from an automated number.** |

**Falling training loss is tier 1 evidence only.** It says the optimiser is working. It is not evidence of
tier 2, 3 or 4, and must never be reported as success.

## Preserved baseline

| Item | Value |
| --- | --- |
| Checkpoint | `outputs/wikipedia_v4_epochs/last.pt`, step 94,000, 1,660,032 parameters |
| Frozen copies | `outputs/wikipedia_baselines/step_48886.pt`, `outputs/research/*/model.pt` |
| Integrity check | `.venv/bin/python -u scripts/verify_baseline.py` against `docs/research_baseline_v1.json` |
| Rule | No run may write into an existing output directory without `--resume`; `best.pt` is kept separately from `last.pt` |

### Measured baseline, 2026-10-24 (`outputs/scorecards/baseline_94000.json`)

| Tier | Metric | Value | Reference |
| --- | --- | --- | --- |
| 1 | weights finite | **PASS** | — |
| 2 | **Dari held-out perplexity** | **255.7** | random model ≈ 8,192 |
| 2 | Science held-out perplexity | 200.4 | |
| 2 | Pashto held-out perplexity | 390.9 | |
| 2 | Wikipedia validation (training distribution) | ≈ 60 | **4× easier than Dari — the model is tuned to Iranian text** |
| 2 | real-word rate | 94.4% | clean Persian 98.5% |
| 2 | distinct 4-grams / longest repeat | 0.985 / 3 | no degenerate repetition |
| 2 | Afghan dialect share | too few markers to score | curriculum 99.5%, Wikipedia 6.9% |
| 3 | arithmetic accuracy | **0.0%** (35/60 produced a number) | — |
| 4 | educational usefulness | not assessed | — |

## Sealed evaluation sets — `data/eval/`

Built by `scripts/build_eval_sets.py`, hashed in `data/eval/manifest.json`. **Whole books** are held out,
not random pages, because pages within a book share vocabulary and a page-level split would leak.

| Set | Rows | Content | Status |
| --- | --- | --- | --- |
| `dari_heldout` | 44 | G1 and G2 Dari readers | sealed |
| `science_heldout` | 387 | G5/G6 Science and Social Science | sealed |
| `pashto_heldout` | 147 | Pashto Wikipedia articles | sealed |
| `math_probes` | 200 | Deterministic arithmetic with known answers | sealed |
| `english_heldout` | **0** | **NOT AVAILABLE — no English corpus held locally** | must be built before any English claim |

Held-out books (must be excluded from every training corpus): `G1-Dr-Dari.pdf`,
`G2-Dr-Dari_compressed.pdf`, `G5-Dr-Science.pdf`, `G5-Dr-Social_Science.pdf`, `G6-Dr-Science.pdf`,
`G6-Dr-Social_Science_2.pdf`.

**Known limitations:** maths probes are synthetic arithmetic, not curriculum word problems. Pashto is
encyclopedic, not classroom Pashto. No English set exists.

## Deterministic evaluation protocol

Every checkpoint is scored identically or the comparison is void:

```bash
.venv/bin/python -u scripts/eval_suite.py --checkpoint <path> --out outputs/scorecards/<name>.json
.venv/bin/python -u scripts/compare_scorecards.py     # writes outputs/scorecards/dashboard.html
```

Fixed: seed 123, temperature 0.8 for generation, **greedy (temperature 0) for arithmetic**, 120 new tokens,
six fixed prompts, same sealed sets. Changing any of these invalidates comparison with earlier scorecards.

The dashboard flags a **REGRESSION** when a metric moves the wrong way by more than its threshold
(2% for perplexity and real-word rate, 5% for dialect and diversity, 25% for longest-repeat).

## Native-speaker rubric — tier 4

Automated metrics cannot judge this. A fluent Afghan Dari (or Pashto) reader scores **20 generated samples**
from fixed prompts, blind to which checkpoint produced them, on five axes, 0–4 each:

| Axis | 0 | 2 | 4 |
| --- | --- | --- | --- |
| **Grammaticality** | Not parseable | Understandable with errors | Natural, correct |
| **Coherence** | Unrelated fragments | Starts well, loses thread | Holds one idea across sentences |
| **Afghan register** | Iranian Persian (دانشگاه، کلاس) | Mixed | Afghan Dari (پوهنتون، صنف، مکتب) |
| **Factual honesty** | Confident fabrication | Vague but harmless | Nothing invented, or declines |
| **Classroom suitability** | Unusable or inappropriate | Needs teacher correction | Could be read by a child |

Scoring rules: blind to checkpoint; the same 20 prompts every time; record the reviewer and date; two
reviewers where possible and report both. **A mean below 2.0 on *Afghan register* or *Factual honesty*
blocks release regardless of every other number.**

Record results in `docs/PROGRESS.md` with the checkpoint hash.

## Hallucination and terminology checks

- **Hallucination probe:** prompts asking for specific facts the model cannot know (dates, names, places).
  Any confident specific answer is a failure. Currently assessed by the human rubric only — **not automated**.
- **Afghan terminology:** `scripts/eval_dialect.py`, 16 marker pairs, validated at 99.5% on curriculum and
  6.9% on Iranian Wikipedia. Generation must exceed **0.90 Afghan share** to claim an Afghan model.
- **Repetition:** distinct-4-gram below 0.5 or longest-repeat above 10 indicates degeneration.

## Financial safety

`src/amc_model/budget.py` + `scripts/train_epochs.py --budget-usd X --gpu-rate Y`:

- Wall-clock billed hours × stated hourly rate, written to `<run>/budget.json` after every checkpoint.
- **Billed hours carry across resume** — stopping and restarting does not reset the meter.
- On reaching the ceiling the trainer **saves a checkpoint, emits a paused event, and exits cleanly**.
- Three consecutive recorded failures also stop the run.
- Non-finite loss already raises immediately (`train_epochs.py`), and gradient clipping uses
  `error_if_nonfinite=True`.
- Covered by two tests in `tests/test_stability.py` (ceiling stop, carry-forward across restart).

**The guard estimates the provider's bill from wall-clock time. It is not the bill.** Set the ceiling below
what you are willing to lose, and check the provider's own billing page.

## Proposed 30M pilot — NOT AUTHORISED, NO SPENDING

| Item | Value |
| --- | --- |
| Model | `configs/medium.json` — 8 layers, 512 wide, 512 context, **29,676,544 parameters** |
| Data | ~600M tokens (≈20 per parameter), mixed per `docs/CORPUS_MANIFEST.md`, held-out books excluded |
| Hardware | RTX 3090 24 GB, community cloud, **$0.22/hour** (live price 2026-10-24, in stock) |
| Data preparation | 2–4 hours |
| Training | 3–5 hours (≈293,000 steps at batch 4 × 512; extrapolated from the measured local 0.63 steps/s, **estimate, not measured on this GPU**) |
| **Estimated total** | **6–9 hours ≈ $1.30–2.00** |
| **Budget ceiling to set** | **`--budget-usd 10`** — 5× the estimate, so a slow run completes rather than dying |
| **Owner's stated maximum** | **$50** — never reached under this plan; anything approaching it means something is wrong, and the run should be stopped and investigated |

Checkpoint every 1,000 steps; `best.pt` kept on validation improvement; resume verified before the run.

## GO / HOLD / STOP criteria

Decided **before** each experiment, checked at the stated point, recorded in `docs/PROGRESS.md`.

### Before launching (pre-flight)
**GO** requires all of: `--check-only` passes on the exact config; the 11 unit tests pass; `verify_baseline.py`
passes; the training corpus provably excludes the six held-out books; budget ceiling set; the baseline
scorecard exists for comparison.
**HOLD** if any is missing. **STOP** if held-out data is found in the training corpus.

### At 30 minutes
**GO** training loss is falling and finite. **HOLD** loss flat — investigate learning rate before continuing.
**STOP** non-finite loss, repeated crashes, or spend above 2× the hourly estimate.

### At the first checkpoint evaluation (step ~10,000)
**GO** Dari held-out perplexity **below 255.7** (the preserved baseline) and real-word rate ≥ 90%.
**HOLD** perplexity between 255.7 and 300 — continue only to the next checkpoint, then re-decide.
**STOP** perplexity above the baseline at half the budget, or distinct-4-gram below 0.5.

### At run completion
**GO to scale** requires all of: Dari perplexity **< 100**; real-word rate ≥ 95%; Afghan dialect share
**> 0.90**; no regression flagged by `compare_scorecards.py`; native-speaker rubric mean ≥ 2.5 with neither
*Afghan register* nor *Factual honesty* below 2.0.
**HOLD** any single criterion missed — diagnose before spending more.
**STOP** dialect share below 0.5 (the model is Iranian, and more of the same data makes it worse).

### Permanent stop conditions
Spending without written authorisation; training on held-out data; reporting a tier-4 claim from a tier-2
number; any release where the rubric blocks it.
