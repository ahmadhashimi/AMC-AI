"""Normalize presentation glyphs and flag a saved sample; no quality approval."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/raw/finepdfs_fa_review'
OUT = ROOT / 'data/processed/finepdfs_fa_review_v1'


def normalize(text):
    # Apply compatibility decomposition only to Arabic presentation forms.
    # Whole-document NFKC can also change mathematical and other symbols.
    text = ''.join(unicodedata.normalize('NFKC', c)
                   if 0xFB50 <= ord(c) <= 0xFDFF or 0xFE70 <= ord(c) <= 0xFEFF
                   else c for c in text)
    return '\n'.join(line.rstrip() for line in text.splitlines()).strip()


def main():
    if OUT.exists():
        raise SystemExit('Preserving existing review output')
    source_bytes = (SOURCE / 'sample.jsonl').read_bytes()
    provenance = json.loads((SOURCE / 'review.json').read_text())
    assert hashlib.sha256(source_bytes).hexdigest() == provenance['sample_sha256']
    rows = []
    counts = Counter()
    for line in source_bytes.decode().splitlines():
        raw = json.loads(line)
        text = normalize(raw['source']['text'])
        flags = list(raw['review_flags'])
        lines = [re.sub(r'^\s*\d+[.)]\s*', '', x.strip()) for x in text.splitlines() if len(x.strip()) >= 20]
        repeated = Counter(lines)
        if repeated and max(repeated.values()) >= 5:
            flags.append('repeated_lines_review')
        if any(x.lstrip().startswith('|') for x in text.splitlines()):
            flags.append('table_reading_order_review')
        if any(c in text for c in 'ٹڈڑںےھ'):
            flags.append('possible_urdu_or_mixed_language')
        counts.update(set(flags))
        rows.append(dict(row_index=raw['row_index'], id=raw['source'].get('id'),
                         url=raw['source'].get('url'), text=text,
                         source_text_sha256=hashlib.sha256(raw['source']['text'].encode()).hexdigest(),
                         review_flags=sorted(set(flags)), approved_for_training=False))
    OUT.mkdir(parents=True)
    (OUT / 'normalized_review.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    report = dict(dataset=provenance['dataset'], revision=provenance['revision'],
                  source_sample_sha256=provenance['sample_sha256'], rows=len(rows),
                  flags=dict(counts), rows_without_heuristic_flags=sum(not r['review_flags'] for r in rows),
                  normalization='Arabic presentation-form decomposition and trailing whitespace only; original text preserved in raw sample.',
                  approved_for_training=False,
                  limitations='Unflagged rows are not verified clean. No correction of reading order, dialect, factual content, OCR hallucinations, personal data or source rights. Source page offsets apply to original text only.')
    (OUT / 'review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
