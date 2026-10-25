# AMC coherence research

## Question and current evidence

Why does next-token validation loss improve while free generation remains incoherent? Do not infer a single cause from the loss curve.

Wikipedia baseline preserved at `outputs/wikipedia_baselines/step_48886.pt` (step 48,886; SHA-256 `38efad735f76636f946493cb7ebb3f8f34ea2d3d9718f26988e21a6536aec83d`). This is a separate copy, not the mutable `last.pt`. It includes optimizer/cursor state. Continue the original run through its original output folder and command; keep this copy for comparison.

At step 44,000, ten fixed validation batches averaged loss 4.2192. Scrambling the input-token positions, while keeping targets unchanged, raised loss to 8.8814. Weight-finiteness, target-shift and causal-mask checks passed. Greedy output still confused Kabul with Japan and the Sun with a ship. This supports learned context sensitivity but does not prove semantic coherence or eliminate every implementation defect.

## Reference audit

Reviewed Sebastian Raschka's [LLMs-from-scratch repository](https://github.com/rasbt/LLMs-from-scratch), especially [chapter 4 model/data/generation code](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/01_main-chapter-code/gpt.py) and [chapter 5 trainer](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/01_main-chapter-code/gpt_train.py).

AMC uses the same core causal next-token objective: one-token-shifted targets, masked scaled dot-product attention, residual blocks, cross-entropy, gradient reset/backpropagation, AdamW, and separate evaluation mode. Differences include model size, tokenizer, tied embeddings, dropout settings, context length and data. Differences are not automatically bugs. The reference is an educational implementation, not evidence that our corpus/model combination must work. No repository clone, pretrained weights or external implementation has replaced AMC.

## Competing explanations

### External documentation and additional measurements

Context7's connector was unavailable in this session, so the audit used its [public PyTorch snippets](https://context7.com/pytorch/pytorch/llms.txt?tokens=10000), which show gradient reset, forward/loss, backward and optimizer update. Official [PyTorch cross-entropy documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html) requires raw logits and supports ignored targets (-100). [Hugging Face's causal-LM course](https://huggingface.co/learn/llm-course/en/chapter7/6) describes shifted next-token targets. These agree with the checked AMC implementation; they do not certify the entire project.

An important correction to earlier reassurance: the [TinyStories paper, section 4.2](https://arxiv.org/html/2305.07759v2) reports factual and consistency weaknesses in its smallest 1M model. The paper's training setup used a 512-token context and a simplified corpus. Small-model success is not a promise of broad factual competence. The [released TinyStories-1M configuration](https://huggingface.co/roneneldan/TinyStories-1M/blob/main/config.json) lists eight layers and width 64; model names alone are not matched architecture/parameter comparisons.

The [softmax saturation paper](https://arxiv.org/html/2404.07647v1) describes a width/rank mismatch in its experimental settings. AMC's current generation failure and loss curve are insufficient to diagnose that mechanism. Do not treat a rough 1,000-dimension observation as a universal threshold.

Ran `scripts/audit_coherence.py` on the preserved step-48,886 checkpoint, with a matched fresh-initialization reference. No training or parameter changes were made. Report: `outputs/research/coherence_audit_48886.json` (local, ignored by Git). Both conditions used the same 32 validation batches, totaling 16,384 target tokens.

| Metric | Fresh weights | Step 48,886 |
| --- | ---: | ---: |
| Sample loss | 9.0239 | 4.3254 |
| Top-1 next-token accuracy | 0.049% | 27.490% |
| Scrambled-context loss | 9.0227 | 8.9459 |
| Centered hidden participation ratio | 40.84 | 36.50 |
| Average pairwise hidden cosine | 0.1601 | 0.1258 |

The accuracy is next-token prediction, not factual correctness. This loss uses 32 batches, whereas dashboard monitoring uses 20, so their numerical values need not match. Hidden statistics are descriptive, lack a diagnostic threshold here, and do not establish representation degeneration or causality. Of AMC's 1,660,032 parameters, 1,048,576 are token embeddings and 594,816 are Transformer-block parameters. This parameter allocation is another capacity consideration, not proof of a defect.

The next gate remains the four-sentence control. After that, test data, context and capacity separately with matched budgets and saved outputs.

| Explanation | Test | What a positive result supports | What it cannot establish |
| --- | --- | --- | --- |
| Core optimization or generation defect | Four-sentence memorization control | Model can fit and reproduce tiny known sequences | Generalization, absence of all bugs |
| Sampling exaggerates errors | Greedy versus fixed-seed sampling on one checkpoint | Sensitivity to decoding | Fixing underlying knowledge |
| Corpus complexity/quality exceeds capacity | Same model, tokenizer, context, token budget and seeds on a separately prepared simpler corpus | Effect of dataset choice | That model width or softmax alone caused failure |
| Context too short | Same story corpus and width, longer context; match token budget | Benefit of longer context under those settings | A pure parameter-matched comparison, because position embeddings and compute change |
| Too little capacity | Same corpus/context/token budget; wider model | Effect of width/parameter capacity | Specific proof of a softmax bottleneck |
| Output-head rank limitation | Later controlled head alternatives and representation diagnostics | Evidence about output expressivity | A universal 1,000-dimension coherence threshold |

## First control: can the model memorize four sentences?

This trains a **fresh** 1,660,032-parameter model with the existing tokenizer on four short, original Persian/Dari sentences. It is not a continuation of Wikipedia, and the same four sentences are used for training and evaluation deliberately. Reported accuracy is teacher-forced training accuracy, not validation accuracy.

```bash
.venv/bin/python -u scripts/research_sanity.py
```

The progress bar shows loss and teacher-forced accuracy; measurements are saved every 50 steps. At 500 steps it prints greedy continuations and whether each exactly matches the known sentence. Outputs go to `outputs/research/sanity_seed42/`; existing output folders are never overwritten. If interrupted, it preserves partial measurements but does not support resume. Use a different output folder for a separate attempt.

Pass criterion: all four complete sentences generated exactly from their fixed prefixes. A miss at this finite budget is not proof of a fundamental defect: inspect the trajectory and specific mismatch. Do not interpret memorization as reasoning or general language ability. Run this control after safely pausing the main CPU run if you want a clean runtime measurement and no CPU contention.

The user completed this control. Local report inspection confirmed: all four sentences matched exactly by step 100 and at each recorded checkpoint through step 500. Initial loss was 8.9943; final loss was 0.009705 and final teacher-forced accuracy was 100%. Runtime was 8.17 seconds. Evidence is in `outputs/research/sanity_seed42/report.json` and `progression.jsonl`.

Conclusion: the current model/tokenizer/training/generation path can memorize and reproduce these short sequences. This argues against a completely broken pipeline. It does not establish generalization, eliminate subtler implementation defects, or identify whether Wikipedia's failure is mainly due to corpus complexity, context, capacity or another factor. A held-out generalization experiment is the next gate. The agent did not run control training; the user did.

## Second control: unseen context combinations

Prepared `scripts/research_generalization.py`. It uses the same tokenizer and 1,660,032-parameter architecture, initialized fresh. Original diagnostic text describes a person owning a colored object and asks for the color. There are 96 training examples, 48 validation examples and 48 reserved test examples. Object/color pairs are disjoint across splits; all individual names, colors and objects occur in training. This probes context copying and compositional transfer in a fixed template, not open-ended language understanding.

```bash
.venv/bin/python -u scripts/research_generalization.py
```

The user runs 1,500 steps. Every 250 steps it reports training-answer exact matches and held-out validation-answer exact matches, plus sample predictions. No test-set answers are evaluated. Default output is `outputs/research/context_seed42/`; the existing Wikipedia and memorization checkpoints are not loaded or modified. Existing output directories are refused. Setup-only checks passed. Training results are pending user execution.

This is a diagnostic rather than a matched Wikipedia comparison: corpus size, format, sequence lengths and repetition differ. A pass means successful transfer within this artificial task; a failure requires examining the budget and setup before attributing it to a specific architecture mechanism.

## Longer-term AMC target

The user's target is a strong Afghan-focused multimodal model with private weights. Track text, speech and vision capabilities separately; no broad capability or strongest-model claim is established by these controls. Current checkpoints remain local and Git-ignored. Source-data and code licenses still apply.

An application router that chooses independent tools/models differs from an internal MoE router choosing neural experts per token. Neither grants expertise without training/evaluation. A text-only medical collection does not provide image/speech supervision. Add an MoE architecture only as an isolated comparison, preserving the dense baseline and holding data/evaluation conditions fixed.

## Story-data inspection so far

Candidate: [taesiri/TinyStories-Farsi](https://huggingface.co/datasets/taesiri/TinyStories-Farsi), revision `87493945c3b0f771d4c115be28e6f5bcd1c1d415`; declared license `cdla-sharing-1.0`.

A bounded download saved the **first 50 training rows**, not a representative random sample, under `data/raw/tinystories_fa_review/`. Two rows contain translator commentary (12 and 43). Other examples show awkward word choices, such as translation of “maze” as “گیره”. No sample rows were empty. Token lengths with our BPE: 111–533, median 245. Many stories exceed the current 128-token context. This sample is inspection material, not an approved clean Dari training corpus. Do not treat automatic prefix removal as a translation-quality review.

The [TinyStories paper](https://arxiv.org/abs/2305.07759) supports testing simple narrative data with small models. It does not prove the quality of this translation or guarantee our results. No full story dataset has been downloaded, cleaned or trained yet.

## Progress and reporting

1. Preserve Wikipedia weights and the existing logs. Done for step 48,886.
2. Run the memorization control and inspect its report. Completed by the user; locally verified, 4/4 exact outputs.
3. Review/filter a larger story sample, fix a held-out split, record exact deduplication and provenance. Pending.
4. Pre-register matched-budget corpus comparisons using the same tokenizer, architecture, context, prompts, seeds, temperatures and generated lengths. Start with one seed as a screening experiment; repeat with multiple seeds before making research claims.
5. Compare context or width only in separate follow-up experiments. Do not silently change the running baseline.

For each checkpoint retain: model/data fingerprints, step and tokens processed, loss on the same held-out set, full fixed-prompt outputs, generation settings, runtime and failure notes. Different validation corpora have different difficulty; compare models on common evaluation sets rather than declaring the lower cross-corpus loss the winner. Blind review of readability, topic consistency and factual correctness should supplement loss and exact-match controls. Add story-domain prompts for story experiments while retaining general prompts as transfer tests.

The regular dashboard measures training progress, not a research verdict. Green status is not a coherence score.
