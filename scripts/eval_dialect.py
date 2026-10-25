"""Score text as Afghan Dari or Iranian Persian by counting marker words."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

# Everyday words where Afghan Dari and Iranian Persian differ. Both columns are
# correct Persian; which one a writer reaches for is the dialect signal.
MARKERS = [
    ('مکتب', 'مدرسه', 'school'), ('صنف', 'کلاس', 'class'),
    ('پوهنتون', 'دانشگاه', 'university'), ('شفاخانه', 'بیمارستان', 'hospital'),
    ('بایسکل', 'دوچرخه', 'bicycle'), ('موتر', 'خودرو', 'car'),
    ('تیلفون', 'تلفن', 'telephone'), ('کتابچه', 'دفترچه', 'notebook'),
    ('سرک', 'خیابان', 'street'), ('چوکی', 'صندلی', 'chair'),
    ('کمپیوتر', 'رایانه', 'computer'), ('روغتون', 'درمانگاه', 'clinic'),
    ('پوهنځی', 'دانشکده', 'faculty'), ('ولایت', 'استان', 'province'),
    ('لیسه', 'دبیرستان', 'high school'), ('دهقان', 'کشاورز', 'farmer'),
]


def score(text):
    counts = []
    afghan = iranian = 0
    for af, ir, gloss in MARKERS:
        a = len(re.findall(rf'(?<![؀-ۿ]){re.escape(af)}(?![؀-ۿ])', text))
        i = len(re.findall(rf'(?<![؀-ۿ]){re.escape(ir)}(?![؀-ۿ])', text))
        afghan += a
        iranian += i
        if a or i:
            counts.append((gloss, af, a, ir, i))
    total = afghan + iranian
    return (afghan / total if total else 0.0), afghan, iranian, counts


def read(path, field='text', limit=0):
    out = []
    for n, line in enumerate(Path(path).read_text(encoding='utf-8').split('\n')):
        if limit and n >= limit:
            break
        if line.strip():
            out.append(json.loads(line).get(field, ''))
    return '\n'.join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--files', nargs='+', help='JSONL files with a text field, or plain text files')
    p.add_argument('--limit', type=int, default=0, help='Lines to read per file')
    p.add_argument('--detail', action='store_true')
    args = p.parse_args()
    for path in args.files:
        text = read(path, limit=args.limit) if str(path).endswith('.jsonl') else Path(path).read_text(encoding='utf-8')
        share, af, ir, counts = score(text)
        verdict = 'AFGHAN DARI' if share > 0.65 else ('IRANIAN PERSIAN' if share < 0.35 else 'MIXED/UNCLEAR')
        print(f'\n{Path(path).name}')
        print(f'  Afghan markers {af:,} | Iranian markers {ir:,} | Afghan share {share:.1%}  -> {verdict}')
        if args.detail:
            for gloss, afw, a, irw, i in sorted(counts, key=lambda c: -(c[2]+c[4]))[:8]:
                print(f'    {gloss:12} {afw} ×{a:<5} vs {irw} ×{i}')


if __name__ == '__main__':
    main()
