"""Check that no held-out evaluation document appears in any training corpus."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from amc_model.exclusions import (corpus_boilerplate, evaluation_fingerprints,
                                  evaluation_shingles, fingerprint, reference_boilerplate, shingles)

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / 'data/eval'
TRAINING = [
    ROOT / 'data/processed/dari_finetune_v1/train.jsonl',
    ROOT / 'data/raw/corpus_v1/fineweb_edu.jsonl',
]
# Files the evaluation sets were cut FROM. They necessarily contain the held-out
# pages and are audited for the record, never treated as training corpora.
SOURCES = [ROOT / 'data/processed/curriculum_clean_v1/pages.jsonl']


def rows(path, limit=0):
    if not path.exists():
        return []
    out = []
    for n, line in enumerate(path.read_text(encoding='utf-8').split('\n')):
        if limit and n >= limit:
            break
        if line.strip():
            out.append(json.loads(line))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training', nargs='*', type=Path, default=TRAINING)
    p.add_argument('--limit', type=int, default=0, help='Rows to read per training file')
    p.add_argument('--out', type=Path, default=ROOT / 'docs/contamination_audit.json')
    p.add_argument('--strict', action='store_true',
                   help='Exit non-zero on any contamination, so a pipeline stops rather than trains')
    args = p.parse_args()

    manifest = json.loads((EVAL / 'manifest.json').read_text(encoding='utf-8'))
    held_books = set(manifest['provenance']['held_out_books'])
    eval_prints, eval_paragraphs = evaluation_fingerprints()
    eval_grams = evaluation_shingles()

    findings, summary = [], {}
    for path in list(args.training) + SOURCES:
        if not path.exists():
            summary[str(path.relative_to(ROOT))] = dict(status='absent')
            continue
        data = rows(path, args.limit)
        boilerplate = reference_boilerplate() | corpus_boilerplate(data)
        grams = {g: n for g, n in eval_grams.items() if g not in boilerplate}
        overlap, paragraph_hits, shingle_hits = Counter(), Counter(), Counter()
        book_hits = Counter()
        for row in data:
            text = row.get('text', '')
            book = row.get('book') or ''
            if book in held_books:
                book_hits[book] += 1
            if not text:
                continue
            if fingerprint(text) in eval_prints:
                overlap[eval_prints[fingerprint(text)]] += 1
                continue
            blocks = [b.strip() for b in text.split('\n') if len(b.strip()) >= 120]
            hit = next((eval_paragraphs[fingerprint(b)] for b in blocks
                        if fingerprint(b) in eval_paragraphs), None)
            if hit:
                paragraph_hits[hit] += 1
                continue
            gram = next((grams[g] for g in shingles(text) if g in grams), None)
            if gram:
                shingle_hits[gram] += 1
        is_source = path in SOURCES
        dirty = overlap or book_hits or paragraph_hits or shingle_hits
        status = ('SOURCE — contains evaluation pages by design, not a training corpus' if is_source
                  else 'CLEAN' if not dirty else 'CONTAMINATED')
        summary[str(path.relative_to(ROOT))] = dict(
            rows=len(data), status=status,
            exact_overlap_with_eval=dict(overlap),
            paragraph_overlap=dict(paragraph_hits),
            near_duplicate_overlap=dict(shingle_hits),
            boilerplate_shingles_ignored=len(boilerplate),
            held_out_books_present=dict(book_hits))
        if status == 'CONTAMINATED':
            findings.append(f'{path.relative_to(ROOT)}: '
                            f'{sum(overlap.values())} pages identical to evaluation rows, '
                            f'{sum(paragraph_hits.values())} paragraph matches, '
                            f'{sum(shingle_hits.values())} near-duplicate matches, '
                            f'{sum(book_hits.values())} pages from held-out books '
                            f'({", ".join(sorted(book_hits)) or "none"})')

    report = dict(held_out_books=sorted(held_books),
                  evaluation_fingerprints=len(eval_prints),
                  training_files=summary, findings=findings,
                  verdict='CLEAN — no training corpus contains held-out material'
                          if not findings else 'CONTAMINATION FOUND',
                  note='curriculum_clean_v1 is the SOURCE the evaluation sets were cut from, so it '
                       'necessarily contains them. It must never be used directly as a training corpus; '
                       'a training corpus has to be built with the held-out books excluded.',
                  limitations='Fingerprint is the first 400 characters after whitespace normalisation. '
                              'It catches identical and near-identical pages, not paraphrase or partial '
                              'overlap, and not a document that merely shares a topic.')
    (args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if findings and args.strict:
        print('\nCONTAMINATION: refusing to proceed.', file=sys.stderr)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
