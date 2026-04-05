"""
Unit tests for app.services.cleaner.clean_text

These are pure-function tests — no external dependencies.
"""

import pytest

from app.services.cleaner import clean_text


class TestCleanTextBasic:
    def test_empty_string_returns_empty(self):
        assert clean_text("") == ""

    def test_whitespace_only_returns_empty(self):
        assert clean_text("   \t  \n  ") == ""

    def test_no_op_on_clean_text(self):
        text = "Hello world."
        assert clean_text(text) == "Hello world."

    def test_strips_leading_trailing_whitespace(self):
        assert clean_text("  hello  ") == "hello"

    def test_strips_leading_trailing_newlines(self):
        assert clean_text("\n\nhello\n\n") == "hello"


class TestHorizontalWhitespaceCollapse:
    def test_multiple_spaces_collapsed_to_one(self):
        assert clean_text("hello   world") == "hello world"

    def test_tab_collapsed_to_space(self):
        assert clean_text("hello\tworld") == "hello world"

    def test_mixed_spaces_and_tabs_collapsed(self):
        assert clean_text("hello \t  \t world") == "hello world"

    def test_non_breaking_space_collapsed(self):
        # U+00A0 non-breaking space
        assert clean_text("hello\u00a0world") == "hello world"

    def test_zero_width_space_collapsed(self):
        # U+200B zero-width space
        assert clean_text("hello\u200bworld") == "hello world"

    def test_ideographic_space_collapsed(self):
        # U+3000 ideographic (CJK) space
        assert clean_text("hello\u3000world") == "hello world"

    def test_leading_spaces_on_line_stripped(self):
        text = "   first line\n   second line"
        assert clean_text(text) == "first line\nsecond line"

    def test_trailing_spaces_on_line_stripped(self):
        text = "first line   \nsecond line   "
        assert clean_text(text) == "first line\nsecond line"


class TestLineEndings:
    def test_crlf_normalised_to_lf(self):
        assert clean_text("hello\r\nworld") == "hello\nworld"

    def test_cr_only_normalised_to_lf(self):
        assert clean_text("hello\rworld") == "hello\nworld"

    def test_mixed_line_endings(self):
        result = clean_text("line1\r\nline2\rline3\nline4")
        assert result == "line1\nline2\nline3\nline4"


class TestBlankLineCollapse:
    def test_single_blank_line_preserved(self):
        result = clean_text("para1\n\npara2")
        assert result == "para1\n\npara2"

    def test_three_blank_lines_collapsed_to_two(self):
        result = clean_text("para1\n\n\n\npara2")
        # After stripping blank lines in step 4/5, consecutive blanks
        # are collapsed; the result should not have more than 2 newlines
        assert "\n\n\n" not in result

    def test_empty_lines_after_stripping_removed(self):
        # A line that contains only spaces becomes empty after strip → removed
        text = "line1\n   \nline2"
        result = clean_text(text)
        assert "   " not in result


class TestUnicodeNormalisation:
    def test_nfc_normalisation(self):
        # "é" can be represented as precomposed (NFC) or decomposed (NFD)
        nfd = "e\u0301"  # decomposed
        nfc = "\xe9"     # precomposed
        assert clean_text(nfd) == nfc

    def test_mixed_unicode_preserved(self):
        text = "Xin chào, thế giới!"
        result = clean_text(text)
        assert "Xin chào" in result
        assert "thế giới" in result


class TestRealWorldArtefacts:
    def test_ocr_artefact_multiple_spaces(self):
        """Typical OCR output: words separated by many spaces."""
        ocr_line = "Revenue     growth     Q3     2024"
        assert clean_text(ocr_line) == "Revenue growth Q3 2024"

    def test_copy_paste_artefact_nbsp(self):
        text = "Section\u00a01.2\u00a0–\u00a0Overview"
        result = clean_text(text)
        assert "\u00a0" not in result
        assert "Section 1.2" in result

    def test_multiline_ocr_with_blank_lines(self):
        text = "Header\n\n\n\nParagraph   one.\n\n\n\nParagraph   two."
        result = clean_text(text)
        assert "Paragraph one." in result
        assert "Paragraph two." in result
        assert "\n\n\n" not in result

    def test_only_punctuation_line_removed(self):
        """Lines that are whitespace after stripping should be gone."""
        text = "Real content.\n   \nMore content."
        result = clean_text(text)
        assert "Real content." in result
        assert "More content." in result
        # The blank-only line should not produce triple newlines
        assert "\n\n\n" not in result
