# What we are building, and how the data gets made

Updated: 2026-10-24. Priority order of languages: **Dari first, Pashto second, English third.**

The model architecture is a commodity — a dense decoder-only Transformer, already written and working. The
scarce thing, and therefore the project, is **Afghan Dari text at known quality**. Everything below is about
producing that.

## What we measured before writing this

| Finding | Evidence |
| --- | --- |
| Dari datasets effectively do not exist | HF index searched for `dari`, `fa-AF`, `prs`, `afghan`: one dataset at 3 downloads; `dari` hits are people named Dario and Moroccan Darija |
| Iranian Persian is abundant | FineWeb-2 `fas_Arab`: 58.9M documents, 102.4 GB, ODC-BY |
| Pashto exists but is small | Pashto Wikipedia 20,529 articles / 115 MB, now at `data/raw/wikipedia_ps/` |
| Our tokenizer cannot carry Pashto | Persian 8,192 BPE: **3.31 chars/token on Persian, 1.74 on Pashto** — Pashto costs double |
| Scraped Persian PDFs are a poor fit | FinePDFs `fas_Arab` sample: Iranian academic OCR, corruption survives heuristics (`finepdfs_fa_candidates_v1`) |
| The curriculum books are extractable | Ligature transposition and reading order solved; 42,122 characters conserved exactly (`G3-Dr-Dari_repaired_v1`) |
| Nothing medical exists in Dari | HF search: Dari none; Pashto two small SFT sets, apparently machine-translated |

## Data sources, in priority order

1. **Afghan curriculum PDFs — the unique asset.** 532 catalogued books at `data/raw/afghan_curriculum/`.
   Authentic Afghan Dari and Pashto, written for children, already graded by level. Nobody else has this as
   clean text. Estimated 10–20M tokens. Highest quality per token in the entire plan.
2. **Afghan web crawl — the volume.** Dari-language Afghan publishers and government: Pajhwok, TOLOnews,
   Ariana, Hasht-e Subh, Azadi, BBC's Afghanistan service, `.gov.af` sites. This is where authentic Afghan
   register at scale comes from, and it is the only source that is both large and genuinely Dari.
3. **Iranian Persian bulk — base competence only.** FineWeb-2 `fas_Arab` and similar. Used to teach the model
   how Persian grammar works, **tagged as non-Afghan** in the manifest, and down-weighted or dropped in the
   final adaptation stage so Iranian vocabulary does not dominate.
4. **Pashto.** Wikipedia (downloaded) plus Pashto pages from the same crawl.
5. **Synthetic expansion.** Exercises, questions and paraphrases generated from source 1. Only after sources
   1–2 exist, and every batch sampled by a human reader before it enters training.

## What "quality data" means here — six gates

Each gate is a script that writes a report and sets `approved_for_training: false` until a human passes it.

1. **Extraction correctness.** Built. Ligature repair, verified reading order, character conservation,
   explicit unmapped-glyph reporting. `scripts/extract_curriculum_text.py`.
2. **Dialect.** Afghan Dari vs Iranian Persian. No such classifier exists publicly; we can build one from
   known-Afghan text (curriculum, Afghan news) against known-Iranian text (the FinePDFs sample we already
   screened). **This classifier is itself a novel, publishable artifact** and is what keeps source 3 from
   silently turning the model Iranian.
3. **Deduplication.** Exact hashing, then near-duplicate detection across sources and against held-out sets.
4. **Cleanliness.** Boilerplate, navigation, PII, OCR damage, mixed-script junk.
5. **Rights.** Per document: URL, publisher, date, licence, permission evidence. Public access is not
   training permission (`docs/DATA_POLICY.md`).
6. **Human sample review.** A fixed number of documents read per source before that source is approved.
   Heuristics triage; they never approve.

## Crawler requirements

Honour `robots.txt`; one request at a time per domain with a delay; a hard cap on pages and bytes per domain
and per run; resumable from a saved frontier; store the raw response and the extracted text separately so
extraction can be redone without re-crawling; record URL, fetch time, HTTP status, content hash and declared
licence for every page; never log in, never bypass a paywall, never touch a `noindex`/disallowed path.

## Prerequisite decision: the tokenizer

The current 8,192-entry Persian tokenizer halves effective context for Pashto and was never trained on Afghan
text. Before any further training we retrain one joint tokenizer — Dari-weighted, with Pashto and English —
at roughly 32,000 entries, on the mixed corpus. Doing this after training would waste the training.

## Deferred, with the reason

- **Vision.** Needs paired image-text in Dari, which does not exist anywhere — except inside the curriculum
  PDFs, whose figures carry Dari captions. That makes vision a *later* product of the same extraction work,
  using a frozen open image encoder with a small trained projection, not a from-scratch effort.
  Object detection is a different model class entirely and would be a separate fine-tune, not pretraining.
- **Medical.** Nothing exists in Dari. A small model generating medical answers from its weights is unsafe;
  the defensible design is grounded retrieval over Dari-language MoPH/WHO material with citations. Scope to
  be decided; it is not a pretraining-stage concern.
- **Mixture of Experts.** An efficiency technique for models far larger than ours. Revisit above ~1B
  parameters. At 30M it adds complexity and buys nothing.

## Order of work

0. Joint Dari/Pashto/English tokenizer specification and training set.
1. Curriculum corpus: all 532 books through the repaired extraction and gates 1–6.
2. Crawler for Afghan Dari web, then the same gates.
3. Dialect classifier, applied retroactively to every source.
4. Synthetic expansion from the curriculum.
5. Only then train — 30M first as a pipeline proof, then scale.

## Decisions needed from the owner

1. Crawl scope: which Afghan publishers, and do we contact any of them for written permission first?
2. Synthetic data: which model generates it, and does its licence permit training on the output?
3. Medical: information retrieval with citations, or out of scope for v1?
