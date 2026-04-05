from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF

from app.core.logging import get_logger
from app.services.extractor.base import BaseExtractor, ExtractedPage

log = get_logger(__name__)

# Minimum characters on a page to consider it "has real text"
# 50 chars avoids false positives from PDF metadata / stray glyphs in scanned files
_TEXT_THRESHOLD = 50
# Need at least 30% of pages with real text to treat as a text-layer PDF.
# Lower ratio means: even partially-scanned docs with some text pages are OCR'd
# (the scanned pages would otherwise yield nothing from PdfTextExtractor).
_TEXT_LAYER_PAGE_RATIO = 0.3


def has_text_layer(file_path: Path) -> bool:
    """
    Return True only if enough PDF pages contain a substantial text layer.

    A 'scanned' PDF typically has 0 extractable characters per page.
    A 'text' PDF typically has hundreds or thousands of characters per page.
    We require both the per-page threshold AND the ratio to be met to avoid
    treating a scanned PDF with a few stray glyphs as a text document.
    """
    try:
        doc = fitz.open(str(file_path))
        total = len(doc)
        if total == 0:
            doc.close()
            return False
        text_pages = sum(
            1 for page in doc if len(page.get_text().strip()) >= _TEXT_THRESHOLD
        )
        doc.close()
        result = (text_pages / total) >= _TEXT_LAYER_PAGE_RATIO
        log.debug(
            "pdf.has_text_layer",
            file=file_path.name,
            total_pages=total,
            text_pages=text_pages,
            ratio=round(text_pages / total, 2),
            result=result,
        )
        return result
    except Exception as exc:
        log.warning("pdf.has_text_layer.error", file=str(file_path), error=str(exc), exc_info=True)
        return False


class PdfTextExtractor(BaseExtractor):
    """Extract text from PDFs that have a native text layer (not scanned)."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def extract(self, file_path: Path) -> Iterator[ExtractedPage]:
        log.debug("pdf.text.start", file=str(file_path))
        try:
            doc = fitz.open(str(file_path))
        except Exception as exc:
            log.error("pdf.text.open_failed", file=str(file_path), error=str(exc))
            raise

        try:
            for page_num, page in enumerate(doc, start=1):
                try:
                    text = page.get_text()
                    if text.strip():
                        yield ExtractedPage(page_number=page_num, text=text)
                    else:
                        log.debug(
                            "pdf.text.empty_page",
                            file=str(file_path),
                            page=page_num,
                        )
                except Exception as exc:
                    log.warning(
                        "pdf.text.page_failed",
                        file=str(file_path),
                        page=page_num,
                        error=str(exc),
                    )
                    continue
        finally:
            doc.close()

        log.debug("pdf.text.done", file=str(file_path))
