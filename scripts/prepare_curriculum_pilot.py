"""Download one official book and preserve page text for review, never training."""
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

import pymupdf as fitz
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://moe.gov.af/sites/default/files/2020-02/G3-Dr-Dari.pdf'
RAW = ROOT / 'data/raw/afghan_curriculum/dari/grade_03/G3-Dr-Dari.pdf'
OUT = ROOT / 'data/extracted/afghan_curriculum/dari/grade_03/G3-Dr-Dari'
LIMIT = 40 * 1024**2


def main():
    if (OUT / 'report.json').exists():
        raise SystemExit('Review report already exists; preserving this extraction.')
    if shutil.disk_usage(ROOT).free < LIMIT + 1024**3:
        raise SystemExit('Need a 1 GiB reserve plus the 40 MiB download limit.')
    RAW.parent.mkdir(parents=True, exist_ok=True)
    if not RAW.exists():
        partial = RAW.with_suffix('.pdf.partial')
        with urllib.request.urlopen(URL, timeout=60) as response, partial.open('xb') as file:
            length = int(response.headers.get('Content-Length', 0))
            if length > LIMIT:
                raise ValueError('PDF exceeds download limit')
            with tqdm(total=length or None, desc='One book', unit='B', unit_scale=True) as bar:
                size = 0
                while chunk := response.read(65536):
                    size += len(chunk)
                    if size > LIMIT:
                        raise ValueError('PDF exceeds download limit')
                    file.write(chunk)
                    bar.update(len(chunk))
        with partial.open('rb') as file:
            if not file.read(5) == b'%PDF-':
                raise ValueError('Response is not a PDF; partial preserved for inspection')
        with fitz.open(partial) as doc:
            if not len(doc):
                raise ValueError('No pages in PDF')
        partial.rename(RAW)
    sha = hashlib.sha256(RAW.read_bytes()).hexdigest()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with fitz.open(RAW) as doc:
        for index, page in enumerate(tqdm(doc, desc='Extracting', unit='page')):
            text = page.get_text('text', sort=True).strip()
            flags = []
            if len(text) < 80:
                flags.append('little_selectable_text')
            if '\ufffd' in text:
                flags.append('replacement_character')
            rows.append(dict(source_url=URL, pdf_sha256=sha, page=index + 1,
                             text=text, review_flags=flags, approved_for_training=False))
        preview_pages = sorted(set([0, min(9, len(doc)-1), min(19, len(doc)-1)]))
        for index in preview_pages:
            doc[index].get_pixmap(matrix=fitz.Matrix(1.3, 1.3)).save(OUT / f'page_{index+1:03}.png')
    with (OUT / 'pages.jsonl').open('x', encoding='utf-8') as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + '\n')
    report = dict(source_url=URL, pdf=str(RAW.relative_to(ROOT)), sha256=sha,
                  bytes=RAW.stat().st_size, pages=len(rows),
                  extracted_characters=sum(len(row['text']) for row in rows),
                  flagged_pages=[row['page'] for row in rows if row['review_flags']],
                  extractor=f'PyMuPDF {fitz.VersionBind}; get_text(text, sort=True)',
                  approved_for_training=False,
                  limitations='Selectable text only. Flags do not validate spelling, reading order, tables or diagrams. No OCR, repairs, tokenization or model training.',
                  rights_status='Public source; training/republication permission not verified.')
    with (OUT / 'report.json').open('x', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write('\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
