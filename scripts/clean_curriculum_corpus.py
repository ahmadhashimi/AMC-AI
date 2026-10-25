"""Clean extracted curriculum pages: drop damage, tag language, repair line edges."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/extracted/afghan_curriculum_corpus_v1/pages.jsonl'
OUT = ROOT / 'data/processed/curriculum_clean_v1'

PASHTO_ONLY = set('ټځڅډړږښګڼۍ')
ARABIC_DIACRITICS = set('ًٌٍَُِّْ')
SENTENCE_START = '.:؛،؟!'
WORD = re.compile(r'[؀-ۿ]{2,}')
LEADER = re.compile(r'\.{4,}|…')   # dotted leaders mark contents and index pages


def language_of(text):
    """Script-level tag only; it does not distinguish Afghan Dari from Iranian Persian."""
    if sum(1 for c in text if c in PASHTO_ONLY) > len(text) * 0.004:
        return 'ps'
    if sum(1 for c in text if c in ARABIC_DIACRITICS) > len(text) * 0.03:
        return 'ar'
    return 'fa'


def damage(text):
    arabic = sum(1 for c in text if '؀' <= c <= 'ۿ')
    noise = sum(1 for c in text if 'Ā' <= c <= '˿')
    return noise / (arabic + noise) if arabic + noise else 1.0


def strip_decoration(text):
    """Remove kashida: it stretches glyphs for justification and carries no meaning,
    but it makes every word it touches a different string from the real word."""
    return text.replace('\u0640', '')


def repair_line_edges(text):
    """Move sentence punctuation stranded at a line start back to that line's end.

    Right-to-left lines put neutral characters on the wrong visual side, so a
    full stop that belongs after the sentence is emitted before it. Only the
    documented punctuation set moves; no word is reordered or invented.
    """
    out, moved = [], 0
    for line in text.split('\n'):
        stripped = line.strip()
        if len(stripped) > 1 and stripped[0] in SENTENCE_START and stripped[-1] not in SENTENCE_START:
            out.append(stripped[1:].rstrip() + stripped[0])
            moved += 1
        else:
            out.append(stripped)
    return '\n'.join(out), moved


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--out-dir', type=Path, default=OUT)
    p.add_argument('--max-damage', type=float, default=0.30, help='Drop books above this noise share')
    p.add_argument('--min-words', type=int, default=25, help='Drop pages with fewer real words')
    args = p.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f'Preserving existing output: {args.out_dir}')

    rows = [json.loads(l) for l in args.source.read_text(encoding='utf-8').split('\n') if l.strip()]
    by_book = defaultdict(list)
    for row in rows:
        by_book[row['source_url'].rsplit('/', 1)[-1]].append(row)

    dropped_books = {n: round(damage(''.join(r['text'] for r in pages)), 3)
                     for n, pages in by_book.items()
                     if damage(''.join(r['text'] for r in pages)) > args.max_damage}

    kept, seen, counts = [], set(), Counter()
    for name, pages in sorted(by_book.items()):
        if name in dropped_books:
            counts['pages_in_dropped_books'] += len(pages)
            continue
        for row in sorted(pages, key=lambda r: r['page']):
            text, moved = repair_line_edges(strip_decoration(row['text']))
            counts['lines_repunctuated'] += moved
            if '�' in text:
                counts['dropped_replacement_chars'] += 1
                continue
            if len(WORD.findall(text)) < args.min_words:
                counts['dropped_too_few_words'] += 1
                continue
            if sum(bool(LEADER.search(x)) for x in text.split('\n')) >= 4:
                counts['dropped_contents_page'] += 1
                continue
            digest = hashlib.sha256(' '.join(text.split()).encode()).hexdigest()
            if digest in seen:          # identical pages recur across books
                counts['dropped_duplicate'] += 1
                continue
            seen.add(digest)
            kept.append(dict(text=text, language=language_of(text), book=name,
                             grade=row['grade'], page=row['page'],
                             source_url=row['source_url'], approved_for_training=False))
            counts['kept'] += 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'pages.jsonl').write_text(
        ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in kept), encoding='utf-8')
    languages = Counter(r['language'] for r in kept)
    characters = Counter()
    for r in kept:
        characters[r['language']] += len(r['text'])
    report = dict(source=str(args.source.relative_to(ROOT)), pages_in=len(rows),
                  pages_kept=len(kept), books_in=len(by_book),
                  books_kept=len(by_book) - len(dropped_books),
                  books_dropped_for_font_damage=dropped_books,
                  actions=dict(counts), pages_by_language=dict(languages),
                  characters_by_language=dict(characters),
                  approved_for_training=False,
                  limitations='Script-level language tags only; Afghan Dari is not distinguished from '
                              'Iranian Persian. Line-edge punctuation is moved, nothing else is rewritten. '
                              'Tables, equations and figure captions remain unreconstructed. Words split '
                              'across line breaks by zero-width non-joiners are left as extracted.')
    (args.out_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                                              encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
