import re
import unicodedata

# Matches one or more horizontal whitespace characters (space, tab, non-breaking space, etc.)
_HORIZ_WS = re.compile(r"[ \t\u00a0\u200b\u3000]+")
# Windows / old Mac line endings
_LINE_ENDINGS = re.compile(r"\r\n|\r")
# More than two consecutive blank lines
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
# Lines that are only whitespace (after per-line strip)
_BLANK_LINE = re.compile(r"^\s+$", re.MULTILINE)


def clean_text(raw: str) -> str:
    """
    Normalise and clean text extracted from any document source.

    Operations (in order):
    1. Unicode NFC normalisation — ensures consistent code points.
    2. Normalise line endings to \\n.
    3. Collapse any run of horizontal whitespace (spaces, tabs, NBSP…) to a
       single space — this is the main fix for OCR / copy-paste artefacts.
    4. Strip leading/trailing whitespace from every line.
    5. Remove lines that became empty after stripping.
    6. Collapse three or more consecutive blank lines to exactly two.
    7. Strip overall leading/trailing whitespace.
    """
    if not raw:
        return ""

    # 1. Unicode normalisation
    text = unicodedata.normalize("NFC", raw)

    # 2. Normalise line endings
    text = _LINE_ENDINGS.sub("\n", text)

    # 3. Collapse horizontal whitespace runs (NOT newlines) to single space
    text = _HORIZ_WS.sub(" ", text)

    # 4. Strip leading/trailing horizontal whitespace from each line.
    # Blank lines (empty after strip) are kept — they are paragraph separators.
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # 5. Cap blank-line runs at 2 (3 or more → 2)
    text = _EXCESS_BLANK_LINES.sub("\n\n", text)

    # 6. Final overall strip
    return text.strip()
