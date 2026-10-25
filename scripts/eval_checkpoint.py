"""Score a checkpoint: validation loss, real-word rate, dialect, and samples."""
import argparse
from collections import Counter
import itertools
import json
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

WORD = re.compile(r'[؀-ۿ]{2,}')
PROMPTS = ['کابل پایتخت', 'برای یادگیری زبان دری', 'آب در زندگی روزمره',
           'امروز به مکتب رفتم و', 'شاگردان در صنف']


def real_word_rate(text, known):
    words = WORD.findall(text.replace('ـ', ''))
    return sum(1 for w in words if w in known) / len(words) if words else 0.0


def build_vocabulary(path, articles):
    counts = Counter()
    with open(path, encoding='utf-8') as f:
        for line in itertools.islice(f, articles):
            counts.update(WORD.findall(json.loads(line)['text'].replace('ـ', '')))
    return {w for w, c in counts.items() if c >= 3}


@torch.inference_mode()
def generate(model, tokenizer, prompt, new_tokens, temperature, device):
    ids = tokenizer.encode(prompt)
    for _ in range(new_tokens):
        window = ids[-model.cfg.context_length:]
        logits, _ = model(torch.tensor([window], device=device))
        probabilities = torch.softmax(logits[0, -1] / temperature, dim=-1)
        ids.append(int(torch.multinomial(probabilities, 1)))
    return tokenizer.decode(ids)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, default=ROOT / 'artifacts/wikipedia_fa_v1/tokenizer.json')
    p.add_argument('--new-tokens', type=int, default=60)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--vocab-articles', type=int, default=30000)
    p.add_argument('--out', type=Path, help='Write the scorecard JSON here')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = AMCTransformer(ModelConfig(**state['config']))
    model.load_state_dict(state['model'])
    model.eval()
    tokenizer = BPETokenizer(args.tokenizer)
    known = build_vocabulary(ROOT / 'data/processed/wikipedia_fa_clean_v4/train.jsonl', args.vocab_articles)

    samples = [dict(prompt=prompt, text=generate(model, tokenizer, prompt, args.new_tokens,
                                                 args.temperature, 'cpu')) for prompt in PROMPTS]
    body = '\n'.join(s['text'] for s in samples)
    share, afghan, iranian, _ = dialect_score(body)
    card = dict(
        checkpoint=str(args.checkpoint), evaluated=time.strftime('%Y-%m-%d %H:%M'),
        step=state.get('steps'), parameters=model.count_parameters(),
        config=state['config'],
        validation_loss_at_checkpoint=None,
        real_word_rate=round(real_word_rate(body, known), 4),
        dialect_afghan_share=round(share, 4), afghan_markers=afghan, iranian_markers=iranian,
        samples=samples,
        reference=dict(real_word_rate_clean_persian=0.985, afghan_share_curriculum=0.995,
                       afghan_share_wikipedia=0.069),
    )
    print(f"checkpoint     : {args.checkpoint}")
    print(f"step           : {card['step']:,}   parameters: {card['parameters']:,}")
    print(f"real-word rate : {card['real_word_rate']:.1%}   (clean Persian reference: 98.5%)")
    print(f"dialect        : {card['dialect_afghan_share']:.1%} Afghan "
          f"({afghan} Afghan vs {iranian} Iranian markers)")
    print(f"                 references: curriculum 99.5% Afghan, Wikipedia 6.9%\n")
    for s in samples:
        print(f"  [{s['prompt']}]\n  {s['text'][:240]}\n")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(card, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print('scorecard:', args.out)


if __name__ == '__main__':
    main()
