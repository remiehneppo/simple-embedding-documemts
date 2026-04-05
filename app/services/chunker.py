"""
Sentence-aware sliding-window chunker.

Strategy
--------
1. Split the cleaned text into sentences using punctuation boundaries.
2. Accumulate sentences until the running word count would exceed *max_words*.
3. When a flush happens, retain the last *overlap_words* words as the start
   of the next chunk (overlap keeps context across chunk boundaries).
4. If a single sentence is longer than *max_words*, it is force-split at word
   boundaries while still respecting overlap.
5. Chunks with fewer than *min_words* words are discarded — they are usually
   stray punctuation or header fragments.
"""

import re

# Match sentence-ending punctuation followed by whitespace or end-of-string.
# Supports ".", "!", "?", and their Vietnamese/CJK equivalents.
_SENTENCE_END = re.compile(r"(?<=[.!?。！？])\s+")


def _count_words(text: str) -> int:
    return len(text.split())


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_END.split(text)
    return [p.strip() for p in parts if p.strip()]


def _tail_words(sentences: list[str], n: int) -> str:
    """Return the last *n* words from the concatenated sentence list."""
    full = " ".join(sentences)
    words = full.split()
    return " ".join(words[-n:]) if len(words) > n else full


def chunk_text(
    text: str,
    max_words: int = 50,
    overlap_words: int = 10,
    min_words: int = 5,
) -> list[str]:
    """
    Split *text* into overlapping chunks of at most *max_words* words.

    Parameters
    ----------
    text:          Cleaned input text.
    max_words:     Hard upper bound on words per chunk.
    overlap_words: Words carried over from the previous chunk.
    min_words:     Chunks with fewer words are discarded.

    Returns
    -------
    List of non-empty chunk strings.
    """
    if not text.strip():
        return []

    sentences = _split_sentences(text)
    if not sentences:
        # No sentence boundaries found — treat whole text as one sentence block
        sentences = [text.strip()]

    chunks: list[str] = []
    current: list[str] = []  # sentences in the current window
    current_wc: int = 0

    for sentence in sentences:
        s_wc = _count_words(sentence)

        # ---- Sentence longer than the window → force-split by words --------
        if s_wc > max_words:
            # Flush current accumulator first
            if current:
                candidate = " ".join(current)
                if _count_words(candidate) >= min_words:
                    chunks.append(candidate)
                overlap_text = _tail_words(current, overlap_words)
                current = [overlap_text] if overlap_text else []
                current_wc = _count_words(" ".join(current))

            # Now split the long sentence into sub-chunks
            words = sentence.split()
            stride = max(1, max_words - overlap_words)
            for i in range(0, len(words), stride):
                sub = " ".join(words[i : i + max_words])
                if _count_words(sub) >= min_words:
                    chunks.append(sub)

            # Carry overlap from the end of the long sentence into next window
            tail = " ".join(words[-overlap_words:]) if len(words) >= overlap_words else sentence
            current = [tail]
            current_wc = _count_words(tail)
            continue

        # ---- Normal sentence: would it overflow the current window? --------
        if current_wc + s_wc > max_words and current:
            candidate = " ".join(current)
            if _count_words(candidate) >= min_words:
                chunks.append(candidate)
            overlap_text = _tail_words(current, overlap_words)
            current = [overlap_text] if overlap_text else []
            current_wc = _count_words(" ".join(current))

        current.append(sentence)
        current_wc += s_wc

    # ---- Flush whatever remains ------------------------------------------
    if current:
        candidate = " ".join(current)
        if _count_words(candidate) >= min_words:
            chunks.append(candidate)

    return chunks
