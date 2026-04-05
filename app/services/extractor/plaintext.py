from pathlib import Path
from typing import Iterator

from app.core.logging import get_logger
from app.services.extractor.base import BaseExtractor, ExtractedPage

log = get_logger(__name__)


class PlainTextExtractor(BaseExtractor):
    """Extract text from plain-text files: .txt, .md, .csv."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".txt", ".md", ".csv"]

    def extract(self, file_path: Path) -> Iterator[ExtractedPage]:
        log.debug("plaintext.start", file=str(file_path))
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            log.error("plaintext.read_failed", file=str(file_path), error=str(exc))
            raise

        if text.strip():
            yield ExtractedPage(page_number=0, text=text)
        else:
            log.warning("plaintext.empty_file", file=str(file_path))

        log.debug("plaintext.done", file=str(file_path), chars=len(text))
