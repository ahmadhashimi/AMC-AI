# AMC corpus manifest v1

Every row below was confirmed against the Hugging Face size API on 2026-10-24. Token figures are estimates
from byte counts (~3.3 characters/token for Persian and Pashto, ~4 for English) except where marked measured.

## Sources

| # | Source | Config | Rows | Size | Licence | Est. tokens | Role |
|---|---|---|---|---|---|---|---|
| 1 | **Afghan curriculum** | Dari, grades 1–6 | 138 books | — | MoE, redistribution unverified | ~2M (measured in progress) | **Authentic Afghan Dari, child register. The unique asset.** |
| 2 | wikimedia/wikipedia | 20231101.fa | 979,869 | 759 MB | CC BY-SA | 220M **measured** | Persian base competence. Iranian register — tag, don't lead with it |
| 3 | wikimedia/wikipedia | 20231101.ps | 20,529 | 115 MB | CC BY-SA | ~35M | Pashto, encyclopedic |
| 4 | **HuggingFaceFW/fineweb-2** | **pbt_Arab** | **646,505** | 0.9 GB | ODC-BY | **~250M** | **Pashto web — 30× Pashto Wikipedia** |
| 5 | HuggingFaceFW/fineweb-2 | fas_Arab | 58,887,122 | 102.4 GB | ODC-BY | ~25B | Persian web. Filter for `.af` domains to mine Afghan Dari |
| 6 | **roneneldan/TinyStories** | default | 2,141,709 | 1.0 GB | CDLA-Sharing-1.0 | ~500M | **Simple English children's stories — the coherence source for a 30M model** |
| 7 | taesiri/TinyStories-Farsi | default | 152,041 | 0.3 GB | CDLA-Sharing-1.0 | ~80M | Persian children's stories. Known translation artifacts — sample before use |
| 8 | HuggingFaceTB/cosmopedia | stories | 4,992,964 | 11.9 GB | Apache-2.0 | ~3B | Synthetic textbooks and stories. The "generate high-quality data" step, already done |
| 9 | HuggingFaceTB/cosmopedia | khanacademy | 24,123 | small | Apache-2.0 | ~30M | School-subject explanations |
| 10 | HuggingFaceFW/fineweb-edu | sample-10BT | 9,672,101 | 28.5 GB | ODC-BY | ~10B | Classifier-filtered *educational* English |

## Why the register matters more than the volume

The TinyStories result: models of 1M–33M parameters trained on simple children's stories produce fluent,
coherent text, while same-size models trained on Wikipedia produce word salad. Our model is **29.7M
parameters**, and our current Wikipedia-trained checkpoint produces exactly that word salad.

So the corpus leads with **child and school register** (sources 1, 6, 7, 8, 9) and uses encyclopedic and web
text (2, 3, 4, 5, 10) for coverage, not as the backbone. Same compute, far better output.

## Budget for the 29.7M model

Roughly 20 tokens per parameter is ~600M tokens; three passes over a ~500M-token corpus reaches that.

| Share | Target | Drawn from | Available |
| --- | --- | --- | --- |
| 50% Dari/Persian | 250M | 1, 2, 7, plus `.af`-filtered 5 | ~300M+ |
| 25% English | 125M | 6, then 8/9/10 | far more than needed |
| 20% Pashto | 100M | 4, 3 | ~285M |
| 5% other | 25M | reserved | — |

Every share is comfortably covered. **Data volume is not the constraint; register, dialect and cleaning are.**

## What still has to be built

1. The dialect gate (Afghan Dari vs Iranian Persian) — nothing public does this.
2. `.af` domain filtering over source 5, which replaces writing a crawler entirely.
3. Dari children's stories — the one genuine gap. No Dari TinyStories exists anywhere.
