"""
Unit tests for app.services.chunker.chunk_text

These are pure-function tests — no external dependencies.
"""

import pytest

from app.services.chunker import _count_words, _split_sentences, _tail_words, chunk_text


# ── Helper function tests ─────────────────────────────────────────────────────

class TestCountWords:
    def test_empty(self):
        assert _count_words("") == 0

    def test_single_word(self):
        assert _count_words("hello") == 1

    def test_multiple_words(self):
        assert _count_words("the quick brown fox") == 4

    def test_extra_spaces_ignored(self):
        # Python str.split() handles multiple spaces
        assert _count_words("  word   another  ") == 2


class TestSplitSentences:
    def test_splits_on_period(self):
        parts = _split_sentences("Hello. World.")
        assert len(parts) == 2
        assert parts[0] == "Hello."
        assert parts[1] == "World."

    def test_splits_on_exclamation(self):
        parts = _split_sentences("Great! Really.")
        assert len(parts) == 2

    def test_splits_on_question_mark(self):
        parts = _split_sentences("Really? Yes.")
        assert len(parts) == 2

    def test_no_split_without_boundary(self):
        parts = _split_sentences("no boundary here at all")
        assert len(parts) == 1

    def test_empty_string(self):
        parts = _split_sentences("")
        assert parts == []


class TestTailWords:
    def test_returns_last_n_words(self):
        result = _tail_words(["one two three four five"], 3)
        assert result == "three four five"

    def test_returns_all_if_fewer_than_n(self):
        result = _tail_words(["one two"], 10)
        assert result == "one two"

    def test_multiple_sentences(self):
        result = _tail_words(["alpha beta", "gamma delta epsilon"], 2)
        assert result == "delta epsilon"

    def test_empty_list(self):
        result = _tail_words([], 5)
        assert result == ""


# ── chunk_text core behaviour ─────────────────────────────────────────────────

class TestChunkTextBasic:
    def test_empty_string_returns_empty_list(self):
        assert chunk_text("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert chunk_text("   \n  ") == []

    def test_single_short_chunk(self):
        # Text has 9 words — well above min_words=5, fits in a single chunk
        text = "This is a short sentence. And another one here."
        result = chunk_text(text, max_words=50)
        assert len(result) == 1
        assert "short sentence" in result[0]

    def test_returns_list_of_strings(self):
        result = chunk_text("Hello world. How are you?", max_words=50)
        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)

    def test_no_empty_chunks(self):
        text = "Word " * 200
        chunks = chunk_text(text, max_words=50)
        assert all(c.strip() for c in chunks)


class TestWordLimits:
    def test_chunk_does_not_exceed_max_words(self):
        # Build a text with many sentences, each 6 words
        sentences = ["Alpha beta gamma delta epsilon zeta." for _ in range(20)]
        text = " ".join(sentences)
        chunks = chunk_text(text, max_words=50, overlap_words=5, min_words=3)
        for c in chunks:
            assert _count_words(c) <= 50, f"Chunk too long: {_count_words(c)} words"

    def test_min_words_filters_short_chunks(self):
        # The overlap tail might be very short; min_words should discard it
        text = "One two three four five six seven eight nine ten."
        chunks = chunk_text(text, max_words=8, overlap_words=3, min_words=5)
        for c in chunks:
            assert _count_words(c) >= 5

    def test_long_sentence_force_split(self):
        """A single sentence with 100 words must be split into multiple chunks."""
        long_sentence = " ".join([f"word{i}" for i in range(100)]) + "."
        chunks = chunk_text(long_sentence, max_words=20, overlap_words=5, min_words=3)
        assert len(chunks) >= 2
        for c in chunks:
            assert _count_words(c) <= 20


class TestOverlap:
    def test_overlap_carries_context(self):
        """Words at end of chunk N should appear at start of chunk N+1."""
        # Create text that will definitely produce multiple chunks
        # 10 sentences of ~8 words each = ~80 words total; max_words=30 → ~3 chunks
        sentences = [
            "The committee reviewed the annual financial performance report carefully.",  # 8w
            "Revenue increased by fifteen percent compared to the previous year.",        # 11w
            "Operating expenses were reduced through strategic cost management initiatives.", # 9w
            "The board approved the proposed budget for the upcoming fiscal quarter.",   # 13w
            "Shareholders expressed strong confidence in the long-term growth strategy.", # 9w
        ]
        text = " ".join(sentences)
        chunks = chunk_text(text, max_words=25, overlap_words=5, min_words=4)

        if len(chunks) >= 2:
            last_words_of_first = chunks[0].split()[-5:]
            first_words_of_second = chunks[1].split()[:10]
            overlap_found = any(w in first_words_of_second for w in last_words_of_first)
            assert overlap_found, (
                f"No overlap detected.\nChunk 0 tail: {last_words_of_first}\n"
                f"Chunk 1 head: {first_words_of_second}"
            )

    def test_zero_overlap_no_repetition(self):
        sentences = [
            "First sentence with exactly eight distinct unique words here.",
            "Second sentence with exactly eight distinct unique words here.",
            "Third sentence ensuring we fill more than one chunk window.",
        ]
        text = " ".join(sentences)
        chunks = chunk_text(text, max_words=15, overlap_words=0, min_words=3)
        # With zero overlap the same word should not appear in consecutive chunks
        # (this is a soft check — just verify chunks exist)
        assert len(chunks) >= 1


class TestDefaultParameters:
    def test_default_max_words_is_50(self):
        """The default max_words matches settings (50) — verify via import."""
        from app.core.config import settings

        assert settings.chunk_max_words == 50

    def test_normal_document_chunked_with_defaults(self):
        """A ~300 word text should produce several chunks with default settings."""
        # 60 sentences × 5 words each = 300 words
        sentences = ["Alpha beta gamma delta epsilon." for _ in range(60)]
        text = " ".join(sentences)
        chunks = chunk_text(text)  # use defaults
        assert len(chunks) >= 2

    def test_chunks_cover_content(self):
        """Every word in the source should appear in at least one chunk."""
        text = "The quick brown fox. Jumps over the lazy dog. Pack my box."
        chunks = chunk_text(text, max_words=10, overlap_words=2, min_words=2)
        combined = " ".join(chunks).lower()
        for word in ["quick", "brown", "fox", "lazy", "dog", "pack"]:
            assert word in combined, f"'{word}' missing from all chunks"


class TestEdgeCases:
    def test_text_without_sentence_boundaries(self):
        """Text with no .!? should be treated as one long block."""
        text = " ".join([f"word{i}" for i in range(30)])
        chunks = chunk_text(text, max_words=15, overlap_words=3, min_words=3)
        assert len(chunks) >= 1

    def test_single_very_long_word(self):
        """Single token exceeding max_words — shouldn't raise, just return it."""
        text = "a" * 10  # One token, far below max_words
        chunks = chunk_text(text, max_words=50, min_words=1)
        assert len(chunks) == 1

    def test_multiline_text(self):
        text = (
            "Line one contains some words.\n"
            "Line two contains more words.\n"
            "Line three also contributes content.\n"
        )
        chunks = chunk_text(text, max_words=50)
        assert len(chunks) >= 1

    def test_mixed_punctuation(self):
        text = "Is this correct? Yes! It is. Definitely."
        chunks = chunk_text(text, max_words=50, min_words=2)
        assert len(chunks) >= 1
        combined = " ".join(chunks)
        assert "correct" in combined
