# What Afghanistan needs from AI — ranked

Ranked by impact multiplied by feasibility for a small open-source team. Dated 2026-10-24.

**The framing fact: adult literacy is roughly 37%, and far lower among women.** A text-only model is
unusable by most of the people it is meant to serve. Speech is therefore not a later nicety; it is what
makes everything else reach anybody.

| # | Capability | Why it matters here | Feasible for us |
| --- | --- | --- | --- |
| 1 | Dari/Pashto speech recognition and synthesis | Most people cannot read fluently; voice is the only interface that reaches them. Whisper is weak on both languages | Yes — fine-tune existing open models |
| 2 | Offline educational text model | Girls barred from secondary school and university since 2022; an offline tutor on a cheap phone reaches them | Yes — in progress |
| 3 | Dari/Pashto OCR | Demonstrated here: a clean government PDF extracted reversed and broken. Every Dari document is locked behind this | Yes — the ligature and reading-order work is already done |
| 4 | Translation Dari↔English, Pashto↔Dari | Unlocks world material for Afghan readers; connects the two national languages | Yes, with a larger model |
| 5 | Grounded health information | Maternal and child mortality among the world's highest | Yes — retrieval with citations, never generation from weights |
| 6 | Crop disease from a phone photo | Most Afghans farm; immediate economic value | Yes — fine-tune an existing vision model |
| 7 | Document understanding for aid and government | Large manual burden on NGOs and ministries | Medium; follows from 3 |
| 8 | Accessibility for low literacy | Multiplies every capability above | Yes; follows from 1 |
| 9 | Medical imaging | Hardware base is thin, data is specialist and private, errors are life-threatening with regulatory exposure | **No — recommend dropping** |
| 10 | Analytics, dashboards, PII tooling | Internal plumbing, not something a family touches | Not a product |

## Recommendation

After the text model reads Dari, go to **1 (speech)** and **3 (OCR)**, not to 9. Both are force multipliers:
speech makes the text model usable by the majority who cannot read it, and OCR manufactures the Dari
training data that does not otherwise exist. Both are fine-tuning jobs on existing open models — weeks, not
years. Medical imaging is the option that sounds most impressive and delivers least.

Code generation is deliberately not on this list. Afghanistan has no shortage of access to coding
assistants; it has a shortage of anything that speaks Dari.
