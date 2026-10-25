"""Download catalogued curriculum books and extract them with the repaired reader."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pymupdf
from tqdm import tqdm
from extract_curriculum_text import order_blocks, page_blocks

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / 'data/raw/afghan_curriculum/catalog.json'
RAW = ROOT / 'data/raw/afghan_curriculum/books'
OUT = ROOT / 'data/extracted/afghan_curriculum_corpus_v1'
MAX_PDF_BYTES = 60 * 1024**2
AGENT = 'AMC-research/0.1 (nonprofit Afghan education corpus; contact via repository)'


def fetch(url, destination):
    request = urllib.request.Request(url, headers={'User-Agent': AGENT})
    with urllib.request.urlopen(request, timeout=180) as response:
        data = response.read(MAX_PDF_BYTES + 1)
    if len(data) > MAX_PDF_BYTES:
        raise ValueError(f'Exceeds {MAX_PDF_BYTES // 1024**2} MiB cap')
    if not data.startswith(b'%PDF'):
        raise ValueError('Response is not a PDF')
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix('.partial')
    partial.write_bytes(data)
    partial.replace(destination)
    return data


def extract(path, book, band):
    document = pymupdf.open(path)
    rows, repairs, private_use = [], 0, Counter()
    for number in range(1, document.page_count + 1):
        blocks = order_blocks(page_blocks(document[number - 1]), band)
        text = '\n'.join(b['text'] for b in blocks)
        repairs += sum(b['repairs'] for b in blocks)
        private_use.update(f'U+{ord(c):04X}' for c in text if 0xE000 <= ord(c) <= 0xF8FF)
        flags = []
        if '�' in text:
            flags.append('replacement_characters')
        if len(text.strip()) < 200:
            flags.append('low_text')
        rows.append(dict(book_id=book['id'], grade=book['grade'], title=book['title'],
                         medium=book['medium_hint'], source_url=book['url'],
                         page=number, text=text, review_flags=flags,
                         approved_for_training=False))
    document.close()
    return rows, repairs, private_use


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--medium', default='dari', choices=('dari', 'pashto', 'unknown', 'all'))
    p.add_argument('--grades', type=int, nargs=2, default=(1, 6), metavar=('LOW', 'HIGH'))
    p.add_argument('--limit', type=int, default=0, help='Stop after N books; 0 means all selected')
    p.add_argument('--band', type=float, default=12.0)
    p.add_argument('--delay', type=float, default=2.0, help='Seconds between downloads')
    p.add_argument('--out-dir', type=Path, default=OUT)
    p.add_argument('--check-only', action='store_true', help='List the selection and stop')
    args = p.parse_args()

    books = json.loads(CATALOG.read_text(encoding='utf-8'))['books']
    low, high = args.grades
    selected = [b for b in books
                if (args.medium == 'all' or b.get('medium_hint') == args.medium)
                and isinstance(b.get('grade'), int) and low <= b['grade'] <= high]
    selected.sort(key=lambda b: (b['grade'], b['url']))
    if args.limit:
        selected = selected[:args.limit]
    print(f'Selected {len(selected)} books: medium={args.medium}, grades {low}-{high}')
    if args.check_only:
        for b in selected[:20]:
            print(f"  G{b['grade']} {b['title'][:30]:32} {b['url'].rsplit('/', 1)[-1][:50]}")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pages_path = args.out_dir / 'pages.jsonl'
    done = set()
    if pages_path.exists():  # resume: keep finished books, never re-download them
        # Only \n delimits JSONL rows; splitlines() would also break on \x0c and
        # \u2028, which JSON leaves unescaped inside strings.
        done = {json.loads(line)['book_id']
                for line in pages_path.read_text(encoding='utf-8').split('\n') if line.strip()}
        print(f'Resuming: {len(done)} books already extracted')

    summary, failures = [], []
    repairs_total, pages_total = 0, 0
    private_use = Counter()
    with pages_path.open('a', encoding='utf-8') as sink:
        for book in tqdm(selected, desc='Books', unit='book'):
            if book['id'] in done:
                continue
            if shutil.disk_usage(ROOT).free < 1024**3 + MAX_PDF_BYTES:
                print('\nStopping: disk reserve reached. Rerun after freeing space; progress is kept.')
                break
            name = book['url'].rsplit('/', 1)[-1][:120]
            path = RAW / f"grade_{book['grade']:02d}" / name
            try:
                data = path.read_bytes() if path.exists() else fetch(book['url'], path)
                rows, repairs, glyphs = extract(path, book, args.band)
            except (urllib.error.URLError, ValueError, RuntimeError, pymupdf.FileDataError) as error:
                failures.append(dict(id=book['id'], url=book['url'], error=f'{type(error).__name__}: {error}'))
                continue
            for row in rows:
                sink.write(json.dumps(row, ensure_ascii=False) + '\n')
            sink.flush()
            characters = sum(len(r['text']) for r in rows)
            summary.append(dict(id=book['id'], grade=book['grade'], title=book['title'],
                                url=book['url'], sha256=hashlib.sha256(data).hexdigest(),
                                bytes=len(data), pages=len(rows), characters=characters,
                                ligature_repairs=repairs,
                                flagged_pages=sum(bool(r['review_flags']) for r in rows)))
            repairs_total += repairs
            pages_total += len(rows)
            private_use.update(glyphs)
            if not path.exists() or data:
                time.sleep(args.delay)  # be polite to moe.gov.af

    report = dict(catalog=str(CATALOG.relative_to(ROOT)), medium=args.medium,
                  grades=[low, high], books_selected=len(selected),
                  books_extracted_this_run=len(summary), books_failed=len(failures),
                  pages=pages_total, characters=sum(s['characters'] for s in summary),
                  lam_alef_repairs=repairs_total,
                  unmapped_private_use=dict(private_use.most_common(20)),
                  approved_for_training=False, failures=failures, books=summary,
                  limitations='Automated extraction with verified reading order and ligature repair. '
                              'Tables, figure captions and decorative glyphs are not reconstructed. '
                              'No dialect, factual, PII or rights verification. Every book still needs '
                              'human sampling against the rendered pages before training.')
    stamp = time.strftime('%Y%m%d-%H%M%S')
    (args.out_dir / f'report-{stamp}.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('books', 'failures')},
                     ensure_ascii=False, indent=2))
    if failures:
        print('Failed:', *[f"  {f['url'].rsplit('/', 1)[-1]}: {f['error'][:90]}" for f in failures], sep='\n')
    print(f'\nCorpus: {pages_path.relative_to(ROOT)} (approved_for_training is false)')


if __name__ == '__main__':
    main()
