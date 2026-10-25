"""Check that no held-out evaluation document appears in any training corpus."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

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


def fingerprint(text):
    """Normalised whitespace, first 400 characters: catches near-identical pages."""
    return hashlib.sha256(' '.join(text.split())[:400].encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training', nargs='*', type=Path, default=TRAINING)
    p.add_argument('--limit', type=int, default=0, help='Rows to read per training file')
    p.add_argument('--out', type=Path, default=ROOT / 'docs/contamination_audit.json')
    args = p.parse_args()

    manifest = json.loads((EVAL / 'manifest.json').read_text(encoding='utf-8'))
    held_books = set(manifest['provenance']['held_out_books'])
    eval_prints, eval_sources = {}, {}
    for name in ('dari_heldout', 'science_heldout', 'pashto_heldout'):
        for row in rows(EVAL / f'{name}.jsonl'):
            text = row.get('text', '')
            if text:
                eval_prints[fingerprint(text)] = name
                eval_sources.setdefault(name, set()).add(row.get('book') or row.get('title'))

    findings, summary = [], {}
    for path in list(args.training) + SOURCES:
        if not path.exists():
            summary[str(path.relative_to(ROOT))] = dict(status='absent')
            continue
        data = rows(path, args.limit)
        overlap = Counter()
        book_hits = Counter()
        for row in data:
            text = row.get('text', '')
            book = row.get('book') or ''
            if book in held_books:
                book_hits[book] += 1
            if text and fingerprint(text) in eval_prints:
                overlap[eval_prints[fingerprint(text)]] += 1
        is_source = path in SOURCES
        status = ('SOURCE — contains evaluation pages by design, not a training corpus' if is_source
                  else 'CLEAN' if not overlap and not book_hits else 'CONTAMINATED')
        summary[str(path.relative_to(ROOT))] = dict(
            rows=len(data), status=status,
            exact_overlap_with_eval=dict(overlap),
            held_out_books_present=dict(book_hits))
        if status == 'CONTAMINATED':
            findings.append(f'{path.relative_to(ROOT)}: '
                            f'{sum(overlap.values())} pages identical to evaluation rows, '
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


if __name__ == '__main__':
    main()
