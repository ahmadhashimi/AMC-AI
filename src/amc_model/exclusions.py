"""Single enforcement point for sealed evaluation material.

Every corpus builder routes through this module. A builder that cannot read the
sealed manifest refuses to run rather than guessing its own split: two
disagreeing holdout rules already put 21 pages of a sealed book into training.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'data/eval/manifest.json'
EVAL_SETS = ('dari_heldout', 'science_heldout', 'pashto_heldout')
# Unfiltered extractions: these contain every book, sealed ones included.
# Boilerplate is measured against one fixed corpus so the filter and the audit
# agree. Measuring it on a filtered output makes it a moving target: removing a
# row changes the statistics and a different row then looks contaminated.
BOILERPLATE_REFERENCE = ROOT / 'data/processed/curriculum_clean_v1/pages.jsonl'
FORBIDDEN_AS_TRAINING_INPUT = (
    'data/extracted/afghan_curriculum_corpus_v1/pages.jsonl',
    'data/processed/curriculum_clean_v1/pages.jsonl',
)


class ContaminationError(RuntimeError):
    """Raised when evaluation material reaches a training corpus."""


def load_manifest(path=None):
    path = Path(path) if path else MANIFEST
    if not Path(path).exists():
        raise ContaminationError(
            f'Sealed evaluation manifest missing: {path}. Run scripts/build_eval_sets.py first; '
            'a training corpus may not be built without a definition of what is held out.')
    return json.loads(Path(path).read_text(encoding='utf-8'))


def held_out_books(path=None):
    return set(load_manifest(path)['provenance']['held_out_books'])


def fingerprint(text, characters=400):
    """Whitespace-normalised prefix hash; catches identical and near-identical text."""
    return hashlib.sha256(' '.join(text.split())[:characters].encode()).hexdigest()


SHINGLE_WORDS = 12          # a 12-word run is long enough to be evidence, short enough to catch fragments
FRONT_MATTER_PAGES = 6      # cover, imprint, minister's foreword, anthem, contents


def shingles(text, size=SHINGLE_WORDS):
    """Overlapping word n-grams: catches a fragment of evaluation text reused
    inside a larger document, which exact paragraph hashing misses."""
    words = text.split()
    return {hashlib.sha256(' '.join(words[i:i + size]).encode()).hexdigest()[:16]
            for i in range(max(len(words) - size + 1, 0))}


def evaluation_shingles(eval_dir=None, max_document_frequency=1):
    """Shingles that identify evaluation text, with shared boilerplate removed.

    Every Ministry textbook carries the same imprint page, so its word runs occur
    in many evaluation documents and in every other book. Such a shingle is not
    evidence of copying: keeping it flagged 43 legitimate training rows. Only
    shingles confined to a single evaluation document are retained.
    """
    folder = Path(eval_dir) if eval_dir else MANIFEST.parent
    occurrences, owner = {}, {}
    for name in EVAL_SETS:
        path = folder / f'{name}.jsonl'
        if not path.exists():
            continue
        for line in path.read_text(encoding='utf-8').split('\n'):
            if not line.strip():
                continue
            row = json.loads(line)
            # Front matter is reprinted in every Ministry textbook, so it can never
            # evidence that an evaluation page was copied. The sealed files are not
            # modified; these pages are simply not used as evidence.
            if (row.get('page') or 99) <= FRONT_MATTER_PAGES:
                continue
            for digest in shingles(row.get('text', '')):
                occurrences[digest] = occurrences.get(digest, 0) + 1
                owner[digest] = name
    return {digest: owner[digest] for digest, count in occurrences.items()
            if count <= max_document_frequency}


def evaluation_fingerprints(eval_dir=None, characters=400):
    """Document-level and paragraph-level fingerprints of every sealed row."""
    folder = Path(eval_dir) if eval_dir else MANIFEST.parent
    documents, paragraphs = {}, {}
    for name in EVAL_SETS:
        path = folder / f'{name}.jsonl'
        if not path.exists():
            continue
        for line in path.read_text(encoding='utf-8').split('\n'):
            if not line.strip():
                continue
            text = json.loads(line).get('text', '')
            if not text:
                continue
            documents[fingerprint(text, characters)] = name
            for block in text.split('\n'):
                block = block.strip()
                if len(block) >= 120:          # short lines recur legitimately
                    paragraphs[fingerprint(block, characters)] = name
    return documents, paragraphs


def reference_boilerplate(path=None, min_books=2):
    """Boilerplate from the fixed reference corpus, so every caller agrees."""
    path = Path(path) if path else BOILERPLATE_REFERENCE
    if not path.exists():
        return set()
    rows = [json.loads(l) for l in path.read_text(encoding='utf-8').split('\n') if l.strip()]
    return corpus_boilerplate(rows, min_books=min_books)


def corpus_boilerplate(rows, text_key='text', book_key='book', min_books=2):
    """Word runs shared by several distinct books in the candidate corpus itself.

    Every Ministry textbook reprints the same imprint and licence page. Such a run
    is publisher boilerplate, not evidence that evaluation text was copied, and
    flagging it rejected 42 legitimate rows. Detected from the corpus side because
    the sealed evaluation sets must never be edited to resolve a conflict.
    """
    occurrences = {}
    for row in rows:
        book = row.get(book_key) or ''
        for digest in shingles(row.get(text_key) or ''):
            occurrences.setdefault(digest, set()).add(book)
    return {digest for digest, books in occurrences.items() if len(books) >= min_books}


def assert_source_allowed(path):
    """Refuse an unfiltered extraction as a direct training input."""
    try:
        relative = str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return
    if relative in FORBIDDEN_AS_TRAINING_INPUT:
        raise ContaminationError(
            f'{relative} is an unfiltered extraction containing sealed evaluation books and may not be '
            'used as training input. Build an approved corpus with the held-out books excluded.')


def filter_rows(rows, book_key='book', text_key='text', eval_dir=None):
    """Drop rows from sealed books or matching sealed text. Returns (kept, report).

    eval_dir selects which sealed directory defines the exclusions, so a test can
    point at a fixture without touching the real sealed sets.
    """
    excluded = held_out_books(Path(eval_dir) / 'manifest.json' if eval_dir else None)
    documents, paragraphs = evaluation_fingerprints(eval_dir)
    grams = evaluation_shingles(eval_dir)
    boilerplate = reference_boilerplate() | corpus_boilerplate(rows, text_key, book_key)
    grams = {g: name for g, name in grams.items() if g not in boilerplate}
    kept, by_book, by_document, by_paragraph, by_shingle = [], {}, {}, {}, {}
    for row in rows:
        book = row.get(book_key) or ''
        text = row.get(text_key) or ''
        if book in excluded:
            by_book[book] = by_book.get(book, 0) + 1
            continue
        hit = documents.get(fingerprint(text))
        if hit:
            by_document[hit] = by_document.get(hit, 0) + 1
            continue
        blocks = [b.strip() for b in text.split('\n') if len(b.strip()) >= 120]
        matched = next((paragraphs[fingerprint(b)] for b in blocks if fingerprint(b) in paragraphs), None)
        if matched:
            by_paragraph[matched] = by_paragraph.get(matched, 0) + 1
            continue
        overlap = next((grams[g] for g in shingles(text) if g in grams), None)
        if overlap:
            by_shingle[overlap] = by_shingle.get(overlap, 0) + 1
            continue
        kept.append(row)
    return kept, dict(rows_in=len(rows), rows_kept=len(kept),
                      dropped_by_book=by_book, dropped_by_document=by_document,
                      dropped_by_paragraph=by_paragraph, dropped_by_shingle=by_shingle,
                      shingle_words=SHINGLE_WORDS, boilerplate_shingles_ignored=len(boilerplate),
                      held_out_books=sorted(excluded))
