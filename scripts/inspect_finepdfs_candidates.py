"""Screen the reviewed FinePDFs sample for prose candidates; approves nothing."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from amc_model.tokenization import BPETokenizer

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/processed/finepdfs_fa_review_v1'
OUT = ROOT / 'data/processed/finepdfs_fa_candidates_v1'
TOKENIZER = ROOT / 'artifacts/wikipedia_fa_v1/tokenizer.json'

# Triage heuristics only. Passing them does not mean a document is clean,
# is Afghan Dari, or has verified reading order.
LIMITS = dict(min_chars=1000, max_latin_share=0.10, max_fragment_share=0.25,
              max_digit_share=0.06, min_short_line_len=25)


def measure(text, tokenizer):
    persian = sum(1 for c in text if '؀' <= c <= 'ۿ')
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    script = persian + latin
    digits = sum(1 for c in text if c.isdigit())
    lines = [x for x in text.splitlines() if x.strip()]
    fragment_chars = sum(len(x) for x in lines if len(x) < LIMITS['min_short_line_len'])
    body = sum(len(x) for x in lines)
    return dict(
        chars=len(text),
        tokens=len(tokenizer.encode(text)),
        latin_share=round(latin / script, 4) if script else 0.0,
        digit_share=round(digits / len(text), 4) if text else 0.0,
        fragment_share=round(fragment_chars / body, 4) if body else 0.0,
        lines=len(lines),
    )


def reasons(m):
    out = []
    if m['chars'] < LIMITS['min_chars']:
        out.append('too_short')
    if m['latin_share'] > LIMITS['max_latin_share']:
        out.append('heavy_latin_mixing')
    if m['fragment_share'] > LIMITS['max_fragment_share']:
        out.append('fragmented_layout_reading_order_unverified')
    if m['digit_share'] > LIMITS['max_digit_share']:
        out.append('numeric_or_tabular_review')
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--show', type=int, default=5, help='Candidate excerpts to print for human review')
    p.add_argument('--excerpt-chars', type=int, default=700)
    p.add_argument('--check-only', action='store_true', help='Measure and print; write nothing')
    args = p.parse_args()

    if OUT.exists() and not args.check_only:
        raise SystemExit(f'Preserving existing output: {OUT}')
    provenance = json.loads((SOURCE / 'review.json').read_text(encoding='utf-8'))
    source_bytes = (SOURCE / 'normalized_review.jsonl').read_bytes()
    tokenizer = BPETokenizer(TOKENIZER)

    rows = [json.loads(line) for line in source_bytes.decode().splitlines()]
    assert len(rows) == provenance['rows'], 'Row count differs from the review report'
    screened, counts = [], Counter()
    for row in rows:
        m = measure(row['text'], tokenizer)
        why = sorted(set(row['review_flags'] + reasons(m)))
        counts.update(why or ['no_heuristic_flag'])
        screened.append(dict(row_index=row['row_index'], id=row['id'], url=row['url'],
                             metrics=m, reject_reasons=why,
                             approved_for_training=False))
    assert all(0.0 <= s['metrics']['latin_share'] <= 1.0 for s in screened)
    assert sum(not s['reject_reasons'] for s in screened) == counts['no_heuristic_flag']

    candidates = sorted((s for s in screened if not s['reject_reasons']),
                        key=lambda s: -s['metrics']['chars'])
    over_context = sum(s['metrics']['tokens'] > 128 for s in candidates)
    report = dict(
        source=str(SOURCE.relative_to(ROOT)),
        dataset=provenance['dataset'], revision=provenance['revision'],
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        rows=len(rows), candidates=len(candidates),
        candidates_over_128_token_context=over_context,
        median_candidate_tokens=sorted(s['metrics']['tokens'] for s in candidates)[len(candidates)//2] if candidates else 0,
        thresholds=LIMITS, reason_counts=dict(counts),
        candidate_rows=[s['row_index'] for s in candidates],
        approved_for_training=False,
        limitations='Character-statistics triage over the first 100 viewer rows of one Persian subset. '
                    'It cannot verify reading order, dialect, factual content, OCR hallucinations, personal '
                    'data or source rights. Candidates still require human reading before any training.',
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    for s in candidates[:args.show]:
        text = next(r['text'] for r in rows if r['row_index'] == s['row_index'])
        print(f"\n--- ROW {s['row_index']}  {s['metrics']}\n{s['url']}\n{text[:args.excerpt_chars]}")
    if args.check_only:
        print('\ncheck-only: nothing written.')
        return
    OUT.mkdir(parents=True)
    (OUT / 'screening.jsonl').write_text(
        ''.join(json.dumps(s, ensure_ascii=False) + '\n' for s in screened), encoding='utf-8')
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'\nWrote {OUT.relative_to(ROOT)}; no dataset approved and no training started.')


if __name__ == '__main__':
    main()
