from pathlib import Path

from app.core.exceptions import UnsupportedFileTypeError
from app.core.logging import get_logger
from app.services.extractor.base import BaseExtractor
from app.services.extractor.docx import DocxExtractor
from app.services.extractor.pdf_ocr import PdfOcrExtractor
from app.services.extractor.pdf_tesseract import PdfTesseractExtractor
from app.services.extractor.pdf_text import PdfTextExtractor, has_text_layer
from app.services.extractor.plaintext import PlainTextExtractor

log = get_logger(__name__)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".pdf", ".docx", ".txt", ".md", ".csv"}
)

# Valid OCR engine identifiers
OCR_ENGINE_PADDLE = "paddle"
OCR_ENGINE_TESSERACT = "tesseract"


def get_extractor(
    file_path: Path,
    ocr_langs: list[str] | None = None,
    ocr_engine: str | None = None,
) -> BaseExtractor:
    """Return the appropriate extractor for a given file.

    Args:
        file_path: Path to the file.
        ocr_langs: Language codes for OCR (e.g. ["vie", "rus"]). Falls back to
            settings.ocr_langs when None.
        ocr_engine: ``"tesseract"`` or ``"paddle"``. Falls back to
            settings.ocr_engine when None.
    """
    suffix = file_path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(suffix)

    if suffix == ".pdf":
        if has_text_layer(file_path):
            log.debug("extractor.selected", type="pdf_text", file=file_path.name)
            return PdfTextExtractor()
        else:
            from app.core.config import settings

            langs = ocr_langs or [l.strip() for l in settings.ocr_langs.split(",") if l.strip()]
            engine = (ocr_engine or settings.ocr_engine).lower()

            if engine == OCR_ENGINE_TESSERACT:
                log.debug(
                    "extractor.selected",
                    type="pdf_tesseract",
                    file=file_path.name,
                    langs=langs,
                )
                return PdfTesseractExtractor(langs=langs)
            else:
                log.debug(
                    "extractor.selected",
                    type="pdf_ocr",
                    file=file_path.name,
                    langs=langs,
                )
                return PdfOcrExtractor(langs=langs)

    if suffix == ".docx":
        log.debug("extractor.selected", type="docx", file=file_path.name)
        return DocxExtractor()

    # .txt | .md | .csv
    log.debug("extractor.selected", type="plaintext", file=file_path.name)
    return PlainTextExtractor()
