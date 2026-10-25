"""Contamination must be detected and must stop a build. These tests inject it."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from amc_model.exclusions import (ContaminationError, assert_source_allowed, evaluation_fingerprints,
                                  filter_rows, fingerprint, held_out_books)

EVAL_PAGE = ('شاگردان متن درس را بخوانند و بنویسند و اهمیت فصل بهار را بدانند. '
             'در تصویر چه می‌بینید؟ دهقان کدام کارها را در فصل بهار انجام می‌دهد؟ ' * 3)
CLEAN_PAGE = ('میکروب‌ها با فعالیت‌های حیاتی انسان‌ها و دیگر موجودات زنده سر و کار دارند. '
              'استخوان‌های اطراف بدن شامل استخوان‌های دست‌ها و پاهاست. ' * 3)


def sealed_dir(folder):
    """A miniature sealed evaluation directory, so tests never touch the real one."""
    folder = Path(folder)
    (folder / 'dari_heldout.jsonl').write_text(
        json.dumps({'text': EVAL_PAGE, 'book': 'HELD-OUT-BOOK.pdf'}, ensure_ascii=False) + '\n',
        encoding='utf-8')
    (folder / 'manifest.json').write_text(json.dumps(
        {'provenance': {'held_out_books': ['HELD-OUT-BOOK.pdf']}, 'sealed': True}), encoding='utf-8')
    return folder


class SourceGuard(unittest.TestCase):
    def test_unfiltered_extractions_are_refused_as_training_input(self):
        for bad in ('data/extracted/afghan_curriculum_corpus_v1/pages.jsonl',
                    'data/processed/curriculum_clean_v1/pages.jsonl'):
            with self.assertRaises(ContaminationError, msg=bad):
                assert_source_allowed(ROOT / bad)

    def test_approved_corpus_is_accepted(self):
        assert_source_allowed(ROOT / 'data/processed/dari_finetune_v1/train.jsonl')

    def test_missing_manifest_refuses_rather_than_guessing(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ContaminationError):
                held_out_books(Path(folder) / 'manifest.json')


class InjectedContamination(unittest.TestCase):
    """Each test deliberately plants evaluation material and demands rejection."""

    def test_page_from_a_held_out_book_is_dropped(self):
        rows = [{'book': 'HELD-OUT-BOOK.pdf', 'text': 'anything at all'},
                {'book': 'SAFE-BOOK.pdf', 'text': CLEAN_PAGE}]
        with tempfile.TemporaryDirectory() as folder:
            sealed_dir(folder)
            kept, report = filter_rows(rows, eval_dir=folder)
        self.assertEqual(len(kept), 1)
        self.assertEqual(report['dropped_by_book'], {'HELD-OUT-BOOK.pdf': 1})

    def test_verbatim_evaluation_text_is_dropped_even_under_a_new_book_name(self):
        rows = [{'book': 'INNOCENT-NAME.pdf', 'text': EVAL_PAGE},
                {'book': 'SAFE-BOOK.pdf', 'text': CLEAN_PAGE}]
        with tempfile.TemporaryDirectory() as folder:
            sealed_dir(folder)
            kept, report = filter_rows(rows, eval_dir=folder)
        self.assertEqual([r['book'] for r in kept], ['SAFE-BOOK.pdf'])
        self.assertTrue(report['dropped_by_document'], 'renamed evaluation page slipped through')

    def test_embedded_evaluation_paragraph_is_dropped(self):
        buried = CLEAN_PAGE + '\n' + EVAL_PAGE.split('. ')[0] + '. ' + EVAL_PAGE.split('. ')[1] + '.'
        rows = [{'book': 'MIXED.pdf', 'text': buried}, {'book': 'SAFE.pdf', 'text': CLEAN_PAGE}]
        with tempfile.TemporaryDirectory() as folder:
            sealed_dir(folder)
            kept, report = filter_rows(rows, eval_dir=folder)
        self.assertNotIn('MIXED.pdf', [r['book'] for r in kept],
                         'evaluation paragraph buried in a larger page was not detected')

    def test_clean_corpus_passes_untouched(self):
        rows = [{'book': 'SAFE-A.pdf', 'text': CLEAN_PAGE}, {'book': 'SAFE-B.pdf', 'text': CLEAN_PAGE[::-1]}]
        with tempfile.TemporaryDirectory() as folder:
            sealed_dir(folder)
            kept, report = filter_rows(rows, eval_dir=folder)
        self.assertEqual(len(kept), 2)
        self.assertFalse(report['dropped_by_book'] or report['dropped_by_document'])


class RealSealedSets(unittest.TestCase):
    def test_real_evaluation_fingerprints_load(self):
        documents, paragraphs = evaluation_fingerprints()
        self.assertGreater(len(documents), 100, 'sealed evaluation rows not found')
        self.assertGreater(len(paragraphs), len(documents))

    def test_fingerprint_ignores_whitespace_differences(self):
        self.assertEqual(fingerprint('یک  دو\n\nسه'), fingerprint('یک دو سه'))


if __name__ == '__main__':
    unittest.main()
