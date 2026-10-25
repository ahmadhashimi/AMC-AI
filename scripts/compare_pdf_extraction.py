"""Compare PDF text-extraction variants against rendered pages; approves nothing."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import fitz

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / 'data/raw/afghan_curriculum/dari/grade_03/G3-Dr-Dari.pdf'
OUT = ROOT / 'data/extracted/afghan_curriculum/dari/grade_03/G3-Dr-Dari_compare_v1'
# Representative pages chosen from the earlier extraction: front matter with
# replacement characters, poetry, an exercise page, continuous prose, a page
# with damaged glyphs, and a two-column poem.
PAGES = (2, 17, 23, 55, 62, 89)


def diagnostics(text):
    pua = Counter(f'U+{ord(c):04X}' for c in text if 0xE000 <= ord(c) <= 0xF8FF)
    return dict(chars=len(text),
                lines=len([x for x in text.splitlines() if x.strip()]),
                replacement_characters=text.count('�'),
                private_use_total=sum(pua.values()),
                private_use_codepoints=dict(pua.most_common()))


def pymupdf_variants(page):
    blocks = [b for b in page.get_text('blocks') if b[6] == 0]
    # Right-to-left page order: top to bottom, then rightmost block first.
    blocks.sort(key=lambda b: (round(b[1], 1), -b[0]))
    return {
        'pymupdf_native': page.get_text('text'),
        'pymupdf_sorted': page.get_text('text', sort=True),
        'pymupdf_blocks_rtl': '\n'.join(b[4].strip() for b in blocks if b[4].strip()),
    }


def pdftotext(mode, number):
    if not shutil.which('pdftotext'):
        return None
    done = subprocess.run(['pdftotext', mode, '-f', str(number), '-l', str(number),
                           str(PDF), '-'], capture_output=True, text=True, timeout=120)
    if done.returncode != 0:
        return None
    return done.stdout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pages', type=int, nargs='+', default=list(PAGES))
    p.add_argument('--dpi', type=int, default=150)
    p.add_argument('--check-only', action='store_true', help='Print diagnostics; write nothing')
    args = p.parse_args()

    if OUT.exists() and not args.check_only:
        raise SystemExit(f'Preserving existing output: {OUT}')
    if shutil.disk_usage(ROOT).free < 1024**3 + 64 * 1024**2:
        raise SystemExit('Need a 1 GiB reserve plus a 64 MiB render budget.')
    pdf_sha = hashlib.sha256(PDF.read_bytes()).hexdigest()

    document = fitz.open(PDF)
    renders, records = {}, []
    for number in args.pages:
        page = document[number - 1]
        variants = pymupdf_variants(page)
        for mode in ('-layout', '-raw'):
            text = pdftotext(mode, number)
            if text is not None:
                variants['poppler' + mode.replace('-', '_')] = text
        records.append(dict(page=number, pdf_sha256=pdf_sha,
                            variants={k: dict(text=v, **diagnostics(v)) for k, v in variants.items()}))
        renders[number] = page.get_pixmap(dpi=args.dpi)
    document.close()

    summary = dict(
        source_pdf=str(PDF.relative_to(ROOT)), pdf_sha256=pdf_sha,
        pages=args.pages, render_dpi=args.dpi,
        variants=sorted({k for r in records for k in r['variants']}),
        per_page={r['page']: {k: {m: v[m] for m in
                                 ('chars', 'lines', 'replacement_characters', 'private_use_total')}
                              for k, v in r['variants'].items()} for r in records},
        reading_order_verdict='undecided: requires visual comparison with the rendered pages',
        approved_for_training=False,
        limitations='Extraction variants only. No OCR, no glyph remapping, no reordering repair, no rights '
                    'clearance. Private-use codepoints are reported, not resolved. A variant that scores '
                    'well on these counts can still have wrong reading order.',
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.check_only:
        print('\ncheck-only: nothing written.')
        return
    OUT.mkdir(parents=True)
    (OUT / 'variants.jsonl').write_text(
        ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
                                      encoding='utf-8')
    for number, pixmap in renders.items():
        pixmap.save(OUT / f'page_{number:03d}.png')
    print(f'\nWrote {OUT.relative_to(ROOT)}: variants.jsonl, summary.json and {len(renders)} renders.')


if __name__ == '__main__':
    main()
