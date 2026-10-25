"""Extract textbook pages with evidence-based ligature repair and RTL block order."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import pymupdf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / 'data/raw/afghan_curriculum/dari/grade_03/G3-Dr-Dari.pdf'
OUT = ROOT / 'data/extracted/afghan_curriculum/dari/grade_03/G3-Dr-Dari_repaired_v1'
ALEF, LAM = 'ا', 'ل'


def repair_line(chars):
    """A zero-width ALEF before a LAM is half of a decomposed lam-alef ligature.

    The PDF emits the ligature's ALEF with an empty bounding box ahead of its
    LAM, which reads as 'باالی' instead of 'بالای'. Swapping that documented
    pair is the only text change made here; nothing else is rewritten.
    """
    out, repairs = [], 0
    index = 0
    while index < len(chars):
        code, width = chars[index]
        if (code == ALEF and width == 0 and index + 1 < len(chars)
                and chars[index + 1][0] == LAM):
            out.append(LAM)
            out.append(ALEF)
            repairs += 1
            index += 2
            continue
        out.append(code)
        index += 1
    return ''.join(out), repairs


def page_blocks(page):
    blocks = []
    for block in page.get_text('rawdict')['blocks']:
        if 'lines' not in block:
            continue
        lines, repairs = [], 0
        for line in block['lines']:
            chars = [(c['c'], c['bbox'][2] - c['bbox'][0])
                     for span in line['spans'] for c in span['chars']]
            text, count = repair_line(chars)
            repairs += count
            if text.strip():
                lines.append(text.rstrip())
        if lines:
            blocks.append(dict(bbox=block['bbox'], text='\n'.join(lines), repairs=repairs))
    return blocks


def order_blocks(blocks, band):
    # Group blocks into horizontal bands, then read each band right to left.
    # shortcut: fixed point band instead of measured line height; revisit if a
    # book with different leading mixes blocks from neighbouring lines.
    return sorted(blocks, key=lambda b: (round(b['bbox'][1] / band), -b['bbox'][0]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pdf', type=Path, default=PDF)
    p.add_argument('--out-dir', type=Path, default=OUT)
    p.add_argument('--band', type=float, default=12.0, help='Vertical band in points for block ordering')
    p.add_argument('--check-only', action='store_true', help='Report only; write nothing')
    args = p.parse_args()

    if args.out_dir.exists() and not args.check_only:
        raise SystemExit(f'Preserving existing output: {args.out_dir}')
    if shutil.disk_usage(ROOT).free < 1024**3:
        raise SystemExit('Need a 1 GiB disk reserve before writing extraction output.')
    pdf_sha = hashlib.sha256(args.pdf.read_bytes()).hexdigest()

    document = pymupdf.open(args.pdf)
    rows, private_use, repairs, replacements = [], Counter(), 0, 0
    for number in tqdm(range(1, document.page_count + 1), desc='Pages', unit='page'):
        page = document[number - 1]
        blocks = order_blocks(page_blocks(page), args.band)
        text = '\n'.join(b['text'] for b in blocks)
        repairs += sum(b['repairs'] for b in blocks)
        replacements += text.count('�')
        found = Counter(f'U+{ord(c):04X}' for c in text if 0xE000 <= ord(c) <= 0xF8FF)
        private_use.update(found)
        flags = []
        if '�' in text:
            flags.append('replacement_characters')
        if found:
            flags.append('unmapped_private_use_glyphs')
        if len(text.strip()) < 200:
            flags.append('low_text')
        rows.append(dict(source_url='https://moe.gov.af/sites/default/files/2020-02/G3-Dr-Dari.pdf',
                         pdf_sha256=pdf_sha, page=number, text=text,
                         ligature_repairs=sum(b['repairs'] for b in blocks),
                         review_flags=flags, approved_for_training=False))
    document.close()

    body = '\n'.join(r['text'] for r in rows)
    lam_alef = body.count(LAM + ALEF)
    report = dict(
        source_pdf=str(args.pdf.relative_to(ROOT)), pdf_sha256=pdf_sha,
        pages=len(rows), characters=len(body),
        extraction='PyMuPDF character stream, lines kept in emitted order, blocks read in '
                   f'{args.band}pt horizontal bands from right to left.',
        lam_alef_ligatures_repaired=repairs, lam_alef_sequences_after_repair=lam_alef,
        replacement_characters=replacements,
        unmapped_private_use_total=sum(private_use.values()),
        unmapped_private_use_codepoints=dict(private_use.most_common()),
        pages_flagged=sum(bool(r['review_flags']) for r in rows),
        approved_for_training=False,
        limitations='Reading order verified by eye on pages 55 and 89 only; tables and the decorative '
                    'answer labels are not reconstructed. Private-use glyphs are reported, not resolved. '
                    'No OCR, no spelling repair, no rights clearance. Every page still needs human reading '
                    'against the rendered original before any training set is built.',
    )
    assert lam_alef >= repairs, 'Repaired ligatures must appear as lam+alef in the output'
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.check_only:
        print('\ncheck-only: nothing written.')
        return
    args.out_dir.mkdir(parents=True)
    (args.out_dir / 'pages.jsonl').write_text(
        ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    (args.out_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                                              encoding='utf-8')
    print(f'\nWrote {args.out_dir.relative_to(ROOT)}; approved_for_training is false.')


if __name__ == '__main__':
    main()
