"""Full checkpoint evaluation: loss, perplexity, repetition, language, maths, dialect.

Four capability tiers are reported separately and must never be conflated:
  1 PIPELINE   — does training mechanically work (finite loss, resume, determinism)
  2 LANGUAGE   — does it model the language (perplexity, real words, repetition)
  3 INSTRUCTION— does it follow a prompt (answer extraction, format adherence)
  4 EDUCATION  — is it useful to a student (human rubric only; never automated)
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import torch
from amc_model.model import AMCTransformer, ModelConfig
from amc_model.tokenization import BPETokenizer
from eval_dialect import score as dialect_score

EVAL = ROOT / 'data/eval'
WORD = re.compile(r'[؀-ۿ]{2,}')
DIGITS = re.compile(r'[-+]?\d+')
PROMPTS = ['کابل پایتخت', 'برای یادگیری زبان دری', 'آب در زندگی روزمره',
           'امروز به مکتب رفتم و', 'شاگردان در صنف', 'معلم به شاگردان گفت']


def vocabulary(articles):
    counts = Counter()
    with open(ROOT / 'data/processed/wikipedia_fa_clean_v4/train.jsonl', encoding='utf-8') as f:
        for line in itertools.islice(f, articles):
            counts.update(WORD.findall(json.loads(line)['text'].replace('ـ', '')))
    return {w for w, c in counts.items() if c >= 3}


@torch.inference_mode()
def perplexity(model, tokenizer, texts, block, device, max_windows=200):
    """Mean next-token NLL over held-out text, and its exponential."""
    total = count = 0
    for text in texts:
        ids = tokenizer.encode(text)[:block * 8]
        for start in range(0, max(len(ids) - 1, 0), block):
            window = ids[start:start + block + 1]
            if len(window) < 2:
                continue
            x = torch.tensor([window[:-1]], device=device)
            y = torch.tensor([window[1:]], device=device)
            _, loss = model(x, y)
            total += loss.item() * y.numel()
            count += y.numel()
            if count >= max_windows * block:
                break
        if count >= max_windows * block:
            break
    if not count:
        return None, None
    mean = total / count
    return round(mean, 4), round(math.exp(min(mean, 20)), 2)


@torch.inference_mode()
def generate(model, tokenizer, prompt, new_tokens, temperature, device):
    ids = tokenizer.encode(prompt)
    for _ in range(new_tokens):
        logits, _ = model(torch.tensor([ids[-model.cfg.context_length:]], device=device))
        if temperature <= 0:
            ids.append(int(logits[0, -1].argmax()))
        else:
            probs = torch.softmax(logits[0, -1] / temperature, dim=-1)
            ids.append(int(torch.multinomial(probs, 1)))
    return tokenizer.decode(ids)


def repetition(text, n=4):
    """Distinct-n and the largest share taken by any single n-gram."""
    tokens = text.split()
    if len(tokens) <= n:
        return dict(distinct_4gram=None, top_4gram_share=None, longest_repeat=None)
    grams = Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    longest, run, previous = 1, 1, None
    for token in tokens:
        run = run + 1 if token == previous else 1
        longest = max(longest, run)
        previous = token
    return dict(distinct_4gram=round(len(grams) / sum(grams.values()), 4),
                top_4gram_share=round(max(grams.values()) / sum(grams.values()), 4),
                longest_repeat=longest)


def language_profile(text):
    arabic = sum(1 for c in text if '؀' <= c <= 'ۿ')
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    digits = sum(1 for c in text if c.isdigit())
    script = arabic + latin
    return dict(arabic_share=round(arabic / script, 4) if script else 0.0,
                latin_share=round(latin / script, 4) if script else 0.0,
                digit_share=round(digits / len(text), 4) if text else 0.0)


def math_correctness(model, tokenizer, probes, device, new_tokens=12):
    """Greedy decoding only: arithmetic has one right answer, so no sampling."""
    correct = attempted = 0
    examples = []
    for probe in probes:
        text = generate(model, tokenizer, probe['prompt'], new_tokens, 0.0, device)
        tail = text[len(probe['prompt']):]
        found = DIGITS.findall(tail)
        got = found[0] if found else None
        if got is not None:
            attempted += 1
            if got == probe['answer']:
                correct += 1
        if len(examples) < 5:
            examples.append(dict(prompt=probe['prompt'], expected=probe['answer'],
                                 got=got, raw=tail[:40]))
    return dict(probes=len(probes), produced_a_number=attempted, correct=correct,
                accuracy=round(correct / len(probes), 4) if probes else None,
                examples=examples)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, default=ROOT / 'artifacts/wikipedia_fa_v1/tokenizer.json')
    p.add_argument('--eval-dir', type=Path, default=EVAL)
    p.add_argument('--out', type=Path, help='Scorecard JSON path')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--new-tokens', type=int, default=120)
    p.add_argument('--vocab-articles', type=int, default=30000)
    p.add_argument('--skip-math', action='store_true')
    args = p.parse_args()

    torch.manual_seed(args.seed)          # deterministic: same seed, prompts and temperature every run
    device = 'cpu'
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = AMCTransformer(ModelConfig(**state['config'])).to(device)
    model.load_state_dict(state['model'])
    model.eval()
    tokenizer = BPETokenizer(args.tokenizer)
    known = vocabulary(args.vocab_articles)
    manifest = json.loads((args.eval_dir / 'manifest.json').read_text(encoding='utf-8'))

    # TIER 1 — pipeline correctness
    finite = all(torch.isfinite(t).all() for t in state['model'].values())
    tier1 = dict(weights_finite=bool(finite), step=state.get('steps'),
                 parameters=model.count_parameters(),
                 checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()[:16])

    # TIER 2 — language modelling on sealed held-out sets
    tier2 = {}
    for name in ('dari_heldout', 'science_heldout', 'pashto_heldout', 'english_heldout'):
        path = args.eval_dir / f'{name}.jsonl'
        texts = [json.loads(l)['text'] for l in path.read_text(encoding='utf-8').split('\n') if l.strip()]
        if not texts:
            tier2[name] = dict(status='NO DATA — set is empty')
            continue
        loss, ppl = perplexity(model, tokenizer, texts, model.cfg.context_length, device)
        tier2[name] = dict(rows=len(texts), loss=loss, perplexity=ppl)

    samples = [dict(prompt=q, text=generate(model, tokenizer, q, args.new_tokens, args.temperature, device))
               for q in PROMPTS]
    body = '\n'.join(s['text'] for s in samples)
    words = WORD.findall(body.replace('ـ', ''))
    share, afghan, iranian, _ = dialect_score(body)
    tier2['generation'] = dict(
        real_word_rate=round(sum(1 for w in words if w in known) / len(words), 4) if words else None,
        type_token_ratio=round(len(set(words)) / len(words), 4) if words else None,
        **repetition(body), **language_profile(body),
        dialect_afghan_share=round(share, 4), afghan_markers=afghan, iranian_markers=iranian)

    # TIER 3 — instruction following, measured by maths probes
    tier3 = dict(status='skipped')
    if not args.skip_math:
        probes = [json.loads(l) for l in (args.eval_dir / 'math_probes.jsonl')
                  .read_text(encoding='utf-8').split('\n') if l.strip()]
        tier3 = math_correctness(model, tokenizer, probes[:60], device)

    card = dict(
        checkpoint=str(args.checkpoint), evaluated=time.strftime('%Y-%m-%d %H:%M:%S'),
        settings=dict(seed=args.seed, temperature=args.temperature, new_tokens=args.new_tokens,
                      prompts=PROMPTS, eval_manifest_sha=manifest['sets']['dari_heldout']['sha256'][:16]),
        tier1_pipeline=tier1, tier2_language=tier2, tier3_instruction=tier3,
        tier4_education=dict(status='NOT AUTOMATED — requires a native-speaker rubric; '
                                    'see docs/EVALUATION.md'),
        references=dict(real_word_rate_clean_persian=0.985, dialect_curriculum=0.995,
                        dialect_wikipedia=0.069, random_model_loss=math.log(8192)),
        samples=samples,
        warning='Falling loss alone is not success. Tier 2 passing does not imply tier 3 or tier 4.')
    print(json.dumps({k: v for k, v in card.items() if k != 'samples'}, ensure_ascii=False, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(card, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print('\nscorecard:', args.out)


if __name__ == '__main__':
    main()
