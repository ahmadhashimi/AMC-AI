"""Regression tests for every PDF extraction defect found in this project.

Each test names the defect, the symptom seen in real output, and the fix.
They use synthetic character streams so no PDF or licensed text is required.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from extract_curriculum_text import SEPARATORS, order_blocks, repair_line
from clean_curriculum_corpus import LEADER, language_of, repair_line_edges, strip_decoration

ALEF, LAM = 'ا', 'ل'


class LigatureRepair(unittest.TestCase):
    """Defect: a lam-alef ligature emits a ZERO-WIDTH ALEF before its LAM,
    so بالای extracted as باالی. The book contained zero لا sequences."""

    def test_zero_width_alef_before_lam_is_swapped(self):
        # ب ا ل ا ی  as the PDF emits it: the ligature ALEF has width 0.
        chars = [('ب', 4.2), (ALEF, 5.2), (ALEF, 0.0), (LAM, 10.5), ('ی', 13.6)]
        text, repairs = repair_line(chars)
        self.assertEqual(text, 'بالای')
        self.assertEqual(repairs, 1)
        self.assertIn(LAM + ALEF, text)

    def test_real_width_alef_before_lam_is_left_alone(self):
        # نااميد genuinely contains ALEF ALEF; both have real width.
        chars = [(ALEF, 5.0), (ALEF, 5.1), (LAM, 10.0)]
        text, repairs = repair_line(chars)
        self.assertEqual(repairs, 0)
        self.assertEqual(text, ALEF + ALEF + LAM)

    def test_trailing_zero_width_alef_is_not_consumed(self):
        chars = [('ب', 4.0), (ALEF, 0.0)]
        text, repairs = repair_line(chars)
        self.assertEqual(repairs, 0)
        self.assertEqual(len(text), 2)


class SeparatorCorruption(unittest.TestCase):
    """Defect: PDF text carries \\x0c and U+2028. json.dumps leaves them
    unescaped, so splitlines() split JSONL rows mid-string: 222 rows read as 226."""

    def test_separators_become_newlines(self):
        raw = 'یک\x0cدو سه\x0bچهار'
        self.assertEqual(raw.translate(SEPARATORS), 'یک\nدو\nسه\nچهار')

    def test_jsonl_row_survives_a_round_trip(self):
        line = json.dumps({'text': 'یک\x0cدو'.translate(SEPARATORS)}, ensure_ascii=False)
        self.assertEqual(len(line.split('\n')), 1)          # one row on the wire
        self.assertEqual(json.loads(line)['text'].count('\n'), 1)


class KashidaStripping(unittest.TestCase):
    """Defect: kashida (U+0640) stretches glyphs for justification. It made
    کیلوگـرام a different string from کیلوگرام, and my own quality metric
    deleted 264 good pages before this was caught."""

    def test_kashida_removed_and_word_restored(self):
        self.assertEqual(strip_decoration('کیلوگـرام'), 'کیلوگرام')
        self.assertEqual(strip_decoration('بشـیر خریـد'), 'بشیر خرید')

    def test_text_without_kashida_is_unchanged(self):
        self.assertEqual(strip_decoration('کتاب مکتب'), 'کتاب مکتب')


class LineEdgePunctuation(unittest.TestCase):
    """Defect: right-to-left lines place sentence-final punctuation at the line
    START. 17,089 lines were affected across 54 books."""

    def test_leading_full_stop_moves_to_the_end(self):
        text, moved = repair_line_edges('.این یک جمله است')
        self.assertEqual(text, 'این یک جمله است.')
        self.assertEqual(moved, 1)

    def test_correctly_punctuated_line_is_untouched(self):
        text, moved = repair_line_edges('این یک جمله است.')
        self.assertEqual(moved, 0)
        self.assertEqual(text, 'این یک جمله است.')

    def test_single_character_line_is_not_mangled(self):
        text, moved = repair_line_edges('.')
        self.assertEqual(moved, 0)
        self.assertEqual(text, '.')


class ReadingOrder(unittest.TestCase):
    """Defect: get_text(sort=True) reverses word order on RTL lines, and plain
    block order put a two-column poem's left hemistich before the right."""

    def test_blocks_in_one_band_read_right_to_left(self):
        right = dict(bbox=(400, 100.0, 560, 130), text='يك جفت كبوتران ابلق', repairs=0)
        left = dict(bbox=(100, 100.4, 260, 130), text='استند جدا جدا معلق', repairs=0)
        # The left block sits a fraction higher; banding must not let that win.
        self.assertEqual([b['text'] for b in order_blocks([left, right], band=12.0)],
                         [right['text'], left['text']])

    def test_lower_band_follows_upper_band(self):
        top = dict(bbox=(100, 100, 300, 120), text='first line', repairs=0)
        bottom = dict(bbox=(400, 140, 560, 160), text='second line', repairs=0)
        self.assertEqual([b['text'] for b in order_blocks([bottom, top], band=12.0)],
                         ['first line', 'second line'])


class ContentsAndLanguage(unittest.TestCase):
    def test_dotted_leader_marks_a_contents_page(self):
        self.assertTrue(LEADER.search('ترکیب حرف «الف».........2'))
        self.assertIsNone(LEADER.search('این یک جملهٔ عادی است.'))

    def test_pashto_letters_select_pashto(self):
        self.assertEqual(language_of('ښه راغلاست زما ګرانو زده کوونکو ' * 5), 'ps')

    def test_plain_dari_is_tagged_fa(self):
        self.assertEqual(language_of('شاگردان متن درس را بخوانند و بنویسند ' * 5), 'fa')


if __name__ == '__main__':
    unittest.main()
