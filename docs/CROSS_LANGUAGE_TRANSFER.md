# Cross-language knowledge transfer — core AMC objective

Added 2026-10-24 at owner instruction. This is now a first-class training objective, not a later phase.

## The objective, stated precisely

AMC learns **knowledge** (programming, mathematics, physics, chemistry, biology, general knowledge) from
high-quality **English** sources, and learns to **express that knowledge naturally in Afghan Dari and
Pashto**. A student asks a technical question in Dari; the model answers correctly in Dari, about a concept
it learned mostly from English text.

This is deliberate and well-founded: English has effectively unlimited high-quality technical material and
Dari has almost none, so the only route to a technically competent Dari model is transfer.

**It does not require pretrained weights.** Transfer is a property of the *data mixture*, not of
initialisation. The from-scratch requirement in §4 of `docs/HANDOFF.md` stands unchanged — random
initialisation, no Llama/Qwen/GPT checkpoints, no distillation.

## What makes transfer actually happen

Mixing monolingual English and monolingual Dari in one corpus produces **weak** transfer. The model has no
reason to connect `photosynthesis` to `فوتوسنتز` unless something in the data links them. Three mechanisms do
that, in increasing order of strength:

1. **Shared subword vocabulary** — one joint tokenizer across all three languages, so numerals, symbols,
   Latin technical terms and loanwords share representations. *Prerequisite, cheap, already planned.*
2. **Bilingual anchors** — the same content in both languages: parallel sentences, glossaries, terminology
   pairs, code-switched text. This is what builds the actual cross-lingual link.
3. **Task supervision** — question in Dari, answer in Dari, about English-sourced knowledge. This teaches the
   behaviour we want, rather than hoping it emerges.

**Honest limitation:** cross-lingual transfer strengthens with scale. Published results show it is weak below
roughly 1B parameters. At the planned 30M pilot, expect **little or no measurable transfer** — the pilot
tests the pipeline, not this objective. Claiming transfer from a 30M run would be unsupportable.

## Datasets required — what we have and what we need

### Have
| Source | Role |
| --- | --- |
| Afghan curriculum, 54/223 books, 0.70M tokens | Dari/Pashto expression, child register, **and the `لغت‌ها` glossary sections in every lesson** — usable as native terminology pairs |
| Persian Wikipedia 222M tokens | Persian fluency (Iranian register — tagged, down-weighted) |
| Pashto Wikipedia 19.4M tokens | Pashto fluency |

### Need — English knowledge sources
| Dataset | Content | Licence | Status |
| --- | --- | --- | --- |
| FineWeb-Edu `sample-10BT` | Classifier-filtered educational English, 9.7M docs | ODC-BY | identified, not downloaded |
| `HuggingFaceTB/cosmopedia` (openstax, khanacademy subsets) | Textbook-style explanations | Apache-2.0 | identified. **Synthetic — owner has previously rejected synthetic data; decision required** |
| `bigcode/the-stack-dedup` or `starcoderdata` | Programming | permissive, opt-out respected | **not yet evaluated** |
| `EleutherAI/proof-pile-2` / `HuggingFaceTB/finemath` | Mathematics | mixed | identified |
| `openai/gsm8k` | Grade-school maths word problems with answers | MIT | identified — ideal register |

### Need — bilingual anchors (the missing piece)
| Dataset | Pair | Status |
| --- | --- | --- |
| **Afghan curriculum English textbooks** (G4–G12 in the catalogue) | en ↔ Dari, same syllabus, same topics | **In our own catalogue, not yet downloaded. Highest-value bilingual source we have.** |
| OPUS collections (CCMatrix, QED, TED, Tanzil, GNOME/KDE) | en ↔ fa, limited en ↔ ps | **not yet surveyed** |
| **FLORES-200** | en ↔ `pes_Arab` (Persian), `pbt_Arab` (Southern Pashto) — professional translations | **evaluation set, not training** |
| Terminology glossaries | scientific terms en ↔ fa/ps | **does not exist in usable form — may have to be built from curriculum glossaries** |

**Unknown:** how much en↔ps parallel text exists anywhere. Pashto is far less resourced than Persian, and
this is the single largest risk to the Pashto half of the objective.

### Need — question/answer supervision
Verified Dari and Pashto Q&A about technical concepts. **This does not exist and must be created**, from:
- the curriculum's own exercise sections (`فعالیت‌ها`, `کار خانه‌گی`) — questions already written by Afghan
  educators, with the lesson text as grounding;
- human-written or human-verified pairs. Machine-translated Q&A is **not** acceptable for the evaluation
  sets, and is a decision for the owner for training data.

## Evaluation design

### The transfer gap — the headline metric
Ask the **same question in English and in Dari**, score correctness separately:

```
transfer_gap = accuracy(English question) − accuracy(Dari question)
```

- Gap near 0 with both high → knowledge transferred. **Objective met.**
- Large positive gap → the model knows the fact but cannot express it in Dari. A *language* problem.
- Both near 0 → it never learned the fact. A *knowledge* problem.

This separates the two failure modes, which a single accuracy number cannot.

### Three scores, never merged (item 5)
| Score | Measures | Method |
| --- | --- | --- |
| **Factual correctness** | Is the answer true? | Parallel Q&A sets with ground truth; exact or human match |
| **Language quality** | Is the Dari/Pashto good? | `eval_dialect.py` Afghan share, real-word rate, repetition, plus the native-speaker rubric in `docs/EVALUATION.md` |
| **Knowledge transfer** | Did English knowledge reach Dari? | transfer_gap above |

A model may score well on language and badly on facts, or the reverse. Reporting one number would hide
exactly the information this objective needs.

### Held-out sets to build (item 4)
| Set | Contents | Status |
| --- | --- | --- |
| `transfer_qa_dari` | ~200 technical questions in Dari with verified answers, on concepts whose training support is overwhelmingly English | **NOT BUILT** |
| `transfer_qa_pashto` | Same, in Pashto | **NOT BUILT** |
| `transfer_qa_english` | The *same* questions in English — required to compute the gap | **NOT BUILT** |
| `terminology_probe` | Technical term in English → expected Dari/Pashto term, from curriculum glossaries | **NOT BUILT — source material exists in our books** |
| FLORES-200 subset | Translation quality for fa and ps | **available, not downloaded** |

These must be **sealed before any training run that claims transfer**, and the source lessons excluded from
training, exactly as the six held-out books already are.

## Multilingual pretraining plan

Staged, because mixture ratios matter more than any single hyperparameter:

| Stage | Data | Purpose |
| --- | --- | --- |
| 0 | Joint 32k tokenizer over en + fa/Dari + ps | Shared subwords — prerequisite for everything below |
| 1 | English knowledge + Persian/Pashto fluency, monolingual | Base competence in all three |
| 2 | Stage 1 **plus bilingual anchors** (parallel text, glossaries, curriculum en↔Dari pairs) | Builds the cross-lingual link |
| 3 | Instruction tuning on verified Dari/Pashto Q&A | Teaches the target behaviour |

Working mixture for stage 1–2, to be revised once the `.af` experiment reports:
English 40%, Dari/Persian 35%, Pashto 20%, bilingual anchors 5%. The English share is higher than the
earlier Dari-led mix in `docs/CORPUS_MANIFEST.md` **because knowledge now has to come from somewhere**, and
that is a deliberate trade against Dari fluency.

**Scale note:** this objective realistically needs ≥1B parameters to show measurable transfer. The staged
plan is correct at any size, but the 30M and 300M runs should be reported as pipeline and language
milestones, never as transfer results.

## Risks

1. **Transfer may not appear at achievable scale.** Mitigation: measure the gap at every checkpoint; do not
   promise the capability before it is measured.
2. **en↔ps parallel data may not exist** in useful quantity. Mitigation: survey OPUS before committing;
   if it is absent, Pashto transfer becomes a translation problem, not a pretraining one.
3. **Q&A supervision must be human-verified**, which is slow and the main non-compute cost of this objective.
4. **Raising English to 40% reduces Dari fluency** at fixed compute. This is a real trade-off, not a free win.
5. **Code and maths corpora carry licence obligations** (opt-outs, attribution) that have not been reviewed.
