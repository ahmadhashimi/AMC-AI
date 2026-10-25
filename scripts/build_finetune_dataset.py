"""Turn extracted curriculum pages into fine-tuning text, dropping broken books."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from amc_model.exclusions import filter_rows

ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / 'data/extracted/afghan_curriculum_corpus_v1/pages.jsonl'
OUT = ROOT / 'data/processed/dari_finetune_v1'
EVAL_MANIFEST = ROOT / 'data/eval/manifest.json'


def held_out_books():
    """Single source of truth for exclusions: the sealed evaluation manifest.

    An earlier version chose its own holdout and disagreed with the evaluation
    sets, putting 21 pages of a held-out book into training.
    """
    if not EVAL_MANIFEST.exists():
        raise SystemExit(f'Evaluation manifest missing: {EVAL_MANIFEST}. '
                         'Run build_eval_sets.py first so exclusions are defined.')
    return set(json.loads(EVAL_MANIFEST.read_text(encoding='utf-8'))['provenance']['held_out_books'])


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
    args = p.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f'Preserving existing dataset: {args.out_dir}')

    raw = [json.loads(l) for l in args.pages.read_text(encoding='utf-8').split('\n') if l.strip()]
    for row in raw:
        row.setdefault('book', row.get('source_url', '').rsplit('/', 1)[-1])
    # One enforcement point: book-name, document, paragraph and near-duplicate
    # exclusion all come from amc_model.exclusions, not from local logic.
    rows, exclusion = filter_rows(raw)
    print(f"evaluation exclusion: {exclusion['rows_kept']:,} of {exclusion['rows_in']:,} pages kept "
          f"(by book {sum(exclusion['dropped_by_book'].values())}, "
          f"document {sum(exclusion['dropped_by_document'].values())}, "
          f"paragraph {sum(exclusion['dropped_by_paragraph'].values())}, "
          f"near-duplicate {sum(exclusion['dropped_by_shingle'].values())})")
    by_book = defaultdict(list)
    for row in rows:
        by_book[row['book']].append(row)

    kept, dropped = {}, {}
    for name, pages in by_book.items():
        share = mojibake_share(''.join(p['text'] for p in pages))
        (dropped if share > 0.3 else kept)[name] = (pages, round(share, 3))
    print(f'books kept {len(kept)}, dropped for broken font {len(dropped)}: {sorted(dropped)}')

    # Whole books are excluded, never random chunks: pages from one book are near
    # duplicates of each other, so a page-level split would leak. The exclusion
    # list comes from the sealed evaluation manifest, not from a local choice.
    excluded = held_out_books()   # already applied by filter_rows; kept for the report
    order = sorted(kept)
    present = sorted(excluded & set(order))
    print(f'held out by evaluation manifest: {len(excluded)} books; '
          f'{len(present)} of them present here and removed: {present}')
    splits = {'train': [], 'validation': []}
    for name in order:
        if name in excluded:
            continue
        pages, _ = kept[name]
        split = 'validation' if name in sorted(set(order) - excluded)[:2] else 'train'
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
        source=str(Path(args.pages).resolve().relative_to(ROOT)), chunk_chars=args.chunk_chars,
        books_kept=sorted(set(kept) - excluded), books_dropped={k: v[1] for k, v in dropped.items()},
        books_excluded_as_evaluation=sorted(excluded), evaluation_exclusion=exclusion,
        examples={k: len(v) for k, v in splits.items()},
        approved_for_training=False,
        limitations='Automated extraction; tables, equations and figure captions are not reconstructed. '
                    'Books whose fonts carry no Unicode map are excluded, not repaired. No dialect, '
                    'factual, PII or rights verification.'), ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')
    print(f'\nDataset: {args.out_dir.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
