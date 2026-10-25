"""Turn extracted curriculum pages into fine-tuning text, dropping broken books."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / 'data/extracted/afghan_curriculum_corpus_v1/pages.jsonl'
OUT = ROOT / 'data/processed/dari_finetune_v1'


def mojibake_share(text):
    """Books whose font carries no Unicode map extract as Latin Extended noise."""
    arabic = sum(1 for c in text if '؀' <= c <= 'ۿ')
    noise = sum(1 for c in text if 'Ā' <= c <= '˿')
    return noise / (arabic + noise) if arabic + noise else 1.0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pages', type=Path, default=PAGES)
    p.add_argument('--out-dir', type=Path, default=OUT)
    p.add_argument('--chunk-chars', type=int, default=3000, help='Join pages into documents of this size')
    p.add_argument('--min-chars', type=int, default=200, help='Drop chunks shorter than this')
    p.add_argument('--val-books', type=int, default=4, help='Whole books held out, never trained on')
    args = p.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f'Preserving existing dataset: {args.out_dir}')

    by_book = defaultdict(list)
    for line in args.pages.read_text(encoding='utf-8').split('\n'):
        if line.strip():
            row = json.loads(line)
            by_book[row['source_url'].rsplit('/', 1)[-1]].append(row)

    kept, dropped = {}, {}
    for name, pages in by_book.items():
        share = mojibake_share(''.join(p['text'] for p in pages))
        (dropped if share > 0.3 else kept)[name] = (pages, round(share, 3))
    print(f'books kept {len(kept)}, dropped for broken font {len(dropped)}: {sorted(dropped)}')

    # Hold out whole books, not random chunks: pages from one book are near
    # duplicates of each other, so a page-level split would leak.
    order = sorted(kept)
    validation = set(order[:args.val_books])
    splits = {'train': [], 'validation': []}
    for name in order:
        pages, _ = kept[name]
        split = 'validation' if name in validation else 'train'
        buffer = ''
        for page in sorted(pages, key=lambda r: r['page']):
            buffer += page['text'].strip() + '\n'
            if len(buffer) >= args.chunk_chars:
                splits[split].append(dict(text=buffer.strip(), book=name, grade=pages[0]['grade']))
                buffer = ''
        if len(buffer.strip()) >= args.min_chars:
            splits[split].append(dict(text=buffer.strip(), book=name, grade=pages[0]['grade']))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        path = args.out_dir / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
        characters = sum(len(r['text']) for r in rows)
        print(f'{split}: {len(rows):,} examples, {characters:,} characters -> {path.name}')
    (args.out_dir / 'report.json').write_text(json.dumps(dict(
        source=str(args.pages.relative_to(ROOT)), chunk_chars=args.chunk_chars,
        books_kept=sorted(kept), books_dropped={k: v[1] for k, v in dropped.items()},
        validation_books=sorted(validation),
        examples={k: len(v) for k, v in splits.items()},
        approved_for_training=False,
        limitations='Automated extraction; tables, equations and figure captions are not reconstructed. '
                    'Books whose fonts carry no Unicode map are excluded, not repaired. No dialect, '
                    'factual, PII or rights verification.'), ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')
    print(f'\nDataset: {args.out_dir.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
