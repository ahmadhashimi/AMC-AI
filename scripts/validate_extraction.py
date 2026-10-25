"""Render PDF pages beside their extracted text for human reading-order review."""
import argparse
from collections import Counter
import html
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import pymupdf
from extract_curriculum_text import SEPARATORS, order_blocks, page_blocks

BOOKS = ROOT / 'data/raw/afghan_curriculum/books'
OUT = ROOT / 'data/extracted/validation_v1'
MATH = re.compile(r'[=+×÷%]|\d\s*[-+]\s*\d')
EXERCISE = re.compile(r'فعالیت|کار\s*خانه|لغت')


def categorise(page, blocks, text):
    """Layout category, so the sample covers every hard case rather than only prose."""
    bands = Counter(round(b['bbox'][1] / 12) for b in blocks)
    columns = sum(1 for count in bands.values() if count > 1)
    digits = sum(c.isdigit() for c in text) / max(len(text), 1)
    if page <= 6:
        return 'front_matter'
    if columns >= 3:
        return 'multi_column_or_poetry'
    if digits > 0.05 or len(MATH.findall(text)) > 6:
        return 'equations_or_tables'
    if EXERCISE.search(text):
        return 'exercise'
    return 'prose'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--books', nargs='*', help='PDF filenames; default is a spread across subjects')
    p.add_argument('--per-category', type=int, default=2)
    p.add_argument('--dpi', type=int, default=130)
    p.add_argument('--band', type=float, default=12.0)
    p.add_argument('--out-dir', type=Path, default=OUT)
    args = p.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f'Preserving existing validation output: {args.out_dir}')

    available = sorted(BOOKS.rglob('*.pdf'))
    if not available:
        raise SystemExit(f'No PDFs under {BOOKS}; raw sources must be preserved before validating')
    chosen = ([b for b in available if b.name in set(args.books)] if args.books else
              [b for b in available if any(k in b.name for k in
                                           ('Dari', 'Science', 'Math', 'Quran', 'Pashto'))][:6])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'renders').mkdir(exist_ok=True)

    cards, counts = [], Counter()
    for pdf in chosen:
        document = pymupdf.open(pdf)
        picked = Counter()
        for number in range(1, document.page_count + 1):
            page = document[number - 1]
            blocks = order_blocks(page_blocks(page), args.band)
            text = '\n'.join(b['text'] for b in blocks).translate(SEPARATORS)
            if len(text.strip()) < 150:
                continue
            category = categorise(number, blocks, text)
            if picked[category] >= args.per_category:
                continue
            picked[category] += 1
            counts[category] += 1
            name = f'{pdf.stem[:28]}_p{number:03d}.png'
            page.get_pixmap(dpi=args.dpi).save(args.out_dir / 'renders' / name)
            cards.append(dict(book=pdf.name, page=number, category=category, render=f'renders/{name}',
                              text=text, blocks=len(blocks), characters=len(text),
                              flags=[f for f, hit in (
                                  ('replacement_characters', '�' in text),
                                  ('leading_punctuation', any(l.strip()[:1] in '.:؛،؟!'
                                                              for l in text.split('\n') if l.strip())),
                                  ('zwnj_at_line_edge', any(l.strip().endswith('‌') or
                                                            l.strip().startswith('‌')
                                                            for l in text.split('\n') if l.strip())),
                              ) if hit]))
        document.close()

    body = ''
    for i, card in enumerate(cards):
        flags = ''.join(f'<span class="flag">{html.escape(f)}</span>' for f in card['flags'])
        body += f'''<section><h2>{i+1}. {html.escape(card['book'])} — page {card['page']}
<small>{card['category']} · {card['blocks']} blocks · {card['characters']} chars</small></h2>{flags}
<div class="pair"><img src="{card['render']}" alt="rendered page">
<div class="text" dir="rtl">{html.escape(card['text'])}</div></div>
<div class="ask"><b>Reviewer:</b> does the extracted text read in the same order as the page?
&nbsp; ☐ correct &nbsp; ☐ lines out of order &nbsp; ☐ words reversed &nbsp; ☐ missing text
&nbsp; ☐ unusable</div></section>'''

    page_html = f'''<!doctype html><html><head><meta charset="utf-8">
<title>AMC extraction validation</title><style>
body{{font-family:system-ui,sans-serif;margin:2rem;background:#fafafa;color:#111;max-width:100rem}}
section{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:1rem;margin-bottom:2rem}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:1rem;align-items:start}}
img{{width:100%;border:1px solid #ccc}}
.text{{white-space:pre-wrap;font-size:1.05rem;line-height:1.9;background:#fcfcfc;
border:1px solid #eee;padding:1rem;max-height:46rem;overflow:auto}}
.flag{{display:inline-block;background:#fde2e1;padding:.15rem .5rem;border-radius:4px;margin-right:.4rem;
font-size:.85rem}}
.ask{{margin-top:.8rem;padding:.6rem;background:#fff8e1;border-radius:6px;font-size:.95rem}}
h2 small{{color:#666;font-weight:normal}}
.intro{{background:#eef4ff;padding:1rem;border-radius:8px;margin-bottom:2rem}}
@media(max-width:60rem){{.pair{{grid-template-columns:1fr}}}}
</style></head><body>
<div class="intro"><h1>AMC extraction validation</h1>
<p>Each card shows the rendered PDF page on the left and what the extractor produced on the right.
Read the page, then read the text. <b>The question is reading order</b>, not spelling: do the lines and
words appear in the same sequence a reader follows on the page?</p>
<p>Categories sampled: {', '.join(f'{k} ({v})' for k, v in sorted(counts.items()))}.
Pages a reviewer marks as out of order or unusable must be <b>excluded</b>, not repaired by guesswork.</p></div>
{body}</body></html>'''
    (args.out_dir / 'validation.html').write_text(page_html, encoding='utf-8')
    report = dict(books=[b.name for b in chosen], pages_sampled=len(cards),
                  by_category=dict(counts), band_points=args.band, dpi=args.dpi,
                  flag_counts=dict(Counter(f for c in cards for f in c['flags'])),
                  cards=[{k: v for k, v in c.items() if k != 'text'} for c in cards],
                  human_review_status='PENDING — no page is approved until a Dari reader marks it',
                  approved_for_training=False)
    (args.out_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                                              encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'cards'}, ensure_ascii=False, indent=2))
    print('\nopen:', args.out_dir / 'validation.html')


if __name__ == '__main__':
    main()
