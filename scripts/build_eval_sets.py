"""Build sealed held-out evaluation sets. Never train on anything this writes."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/eval'
CURRICULUM = ROOT / 'data/processed/curriculum_clean_v1/pages.jsonl'
PASHTO = ROOT / 'data/raw/wikipedia_ps/articles.jsonl'

# Whole books are held out, never random pages: pages inside one book repeat
# vocabulary and phrasing, so a page-level split leaks.
HELD_OUT_SUBJECTS = ('Science', 'Social_Science')


def read_jsonl(path, limit=0):
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding='utf-8').split('\n')):
        if limit and n >= limit:
            break
        if line.strip():
            rows.append(json.loads(line))
    return rows


def math_probes(count, seed):
    """Deterministic arithmetic with known answers, phrased as the books phrase it."""
    rng = random.Random(seed)
    probes = []
    for _ in range(count):
        kind = rng.choice(['add', 'sub', 'mul'])
        a, b = rng.randint(2, 99), rng.randint(2, 9 if kind == 'mul' else 99)
        if kind == 'sub' and b > a:
            a, b = b, a
        answer = {'add': a + b, 'sub': a - b, 'mul': a * b}[kind]
        symbol = {'add': '+', 'sub': '-', 'mul': '×'}[kind]
        probes.append(dict(prompt=f'{a} {symbol} {b} = ', answer=str(answer), kind=kind))
    return probes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', type=Path, default=OUT)
    p.add_argument('--seed', type=int, default=20261024)
    p.add_argument('--pashto-articles', type=int, default=300)
    p.add_argument('--math-probes', type=int, default=200)
    args = p.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f'Evaluation sets already exist and must stay sealed: {args.out_dir}')

    sets, provenance = {}, {}

    # 1. Dari and science: whole held-out books from the curriculum
    pages = read_jsonl(CURRICULUM)
    by_book = defaultdict(list)
    for row in pages:
        by_book[row['book']].append(row)
    science = [b for b in by_book if any(s in b for s in HELD_OUT_SUBJECTS)]
    rng = random.Random(args.seed)
    dari_books = sorted(b for b in by_book if b not in science and 'Dari' in b)
    held_dari = dari_books[:2]
    sets['dari_heldout'] = [dict(text=r['text'], book=r['book'], grade=r['grade'], page=r['page'])
                            for b in held_dari for r in by_book[b] if len(r['text']) > 400]
    sets['science_heldout'] = [dict(text=r['text'], book=r['book'], grade=r['grade'], page=r['page'])
                               for b in sorted(science) for r in by_book[b] if len(r['text']) > 400]
    provenance['held_out_books'] = sorted(held_dari) + sorted(science)

    # 2. Pashto: Wikipedia articles, sampled deterministically
    pashto = read_jsonl(PASHTO, limit=5000)
    rng.shuffle(pashto)
    sets['pashto_heldout'] = [dict(text=r['text'][:4000], title=r.get('title'))
                              for r in pashto[:args.pashto_articles] if len(r.get('text', '')) > 500]

    # 3. Mathematics: deterministic probes with checkable answers
    sets['math_probes'] = math_probes(args.math_probes, args.seed)

    # 4. English: NOT AVAILABLE locally. Recorded so it is not silently skipped.
    sets['english_heldout'] = []

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(created='2026-10-24', seed=args.seed, sets={}, provenance=provenance,
                    sealed=True,
                    rules='These files are evaluation only. Any training corpus must exclude every book '
                          'listed in provenance.held_out_books and must not contain these Pashto articles.',
                    limitations='english_heldout is empty: no English corpus is held locally. Math probes '
                                'are synthetic arithmetic, not curriculum word problems. Pashto is '
                                'encyclopedic Wikipedia, not Afghan classroom Pashto.')
    for name, rows in sets.items():
        path = args.out_dir / f'{name}.jsonl'
        content = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
        path.write_text(content, encoding='utf-8')
        manifest['sets'][name] = dict(rows=len(rows), characters=sum(len(r.get('text', '')) for r in rows),
                                      sha256=hashlib.sha256(content.encode()).hexdigest(),
                                      status='EMPTY — not available' if not rows else 'sealed')
    (args.out_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                                                encoding='utf-8')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'limitations'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
