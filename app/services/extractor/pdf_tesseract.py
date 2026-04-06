import io
import re
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import fitz  # PyMuPDF

from app.core.logging import get_logger
from app.services.extractor.base import BaseExtractor, ExtractedPage

if TYPE_CHECKING:
    from PIL import Image as PILImage

log = get_logger(__name__)

# Tesseract language code mapping: internal name → tesseract lang code
_LANG_MAP: dict[str, str] = {
    "vi": "vie",
    "vie": "vie",
    "ru": "rus",
    "rus": "rus",
    "en": "eng",
    "eng": "eng",
}

# OSD minimum confidence (degrees) to trust orientation detection
_OSD_MIN_CONFIDENCE = 1.5


def _to_tesseract_langs(langs: list[str]) -> str:
    """Convert internal lang codes to a tesseract-compatible '+' joined string."""
    seen: list[str] = []
    for lang in langs:
        tess_code = _LANG_MAP.get(lang.lower())
        if tess_code and tess_code not in seen:
            seen.append(tess_code)
    return "+".join(seen) if seen else "vie"


def _auto_rotate(img: "PILImage.Image") -> "PILImage.Image":
    """
    Detect page orientation using Tesseract OSD and rotate accordingly.
    Returns the (possibly rotated) image.
    Falls back to the original image if OSD fails or confidence is too low.
    """
    try:
        import pytesseract

        osd_raw = pytesseract.image_to_osd(
            img,
            config="--psm 0 --oem 3",
            nice=0,
        )
        # Parse "Rotate: 90" from the OSD output
        rotate_match = re.search(r"Rotate:\s*(\d+)", osd_raw)
        conf_match = re.search(r"Orientation confidence:\s*([0-9.]+)", osd_raw)

        if rotate_match:
            degrees = int(rotate_match.group(1))
            confidence = float(conf_match.group(1)) if conf_match else 0.0
            log.debug(
                "ocr.osd",
                rotate=degrees,
                confidence=confidence,
            )
            if degrees != 0 and confidence >= _OSD_MIN_CONFIDENCE:
                # PIL rotate is counter-clockwise; OSD "Rotate" means
                # "rotate this many degrees counter-clockwise to fix orientation"
                img = img.rotate(degrees, expand=True)
                log.debug("ocr.rotated", degrees=degrees)
    except Exception as exc:
        log.debug("ocr.osd_failed", error=str(exc))

    return img


class PdfTesseractExtractor(BaseExtractor):
    """Extract text from scanned PDF pages using Tesseract OCR (LSTM engine, full models)."""

    def __init__(self, langs: list[str] | None = None, dpi: int = 300) -> None:
        self._langs = langs or ["vie"]
        self._dpi = dpi
        self._tess_langs = _to_tesseract_langs(self._langs)
        # --oem 1  → LSTM neural network only (no legacy, full accuracy models)
        # --psm 6  → Assume a uniform block of text
        self._tess_config = "--oem 1 --psm 6"

    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def extract(self, file_path: Path) -> Iterator[ExtractedPage]:
        import pytesseract
        from PIL import Image

        log.debug(
            "pdf.tesseract.start",
            file=str(file_path),
            langs=self._tess_langs,
            dpi=self._dpi,
        )

        try:
            doc = fitz.open(str(file_path))
        except Exception as exc:
            log.error("pdf.tesseract.open_failed", file=str(file_path), error=str(exc))
            raise

        try:
            for page_num, page in enumerate(doc, start=1):
                try:
                    pix = page.get_pixmap(dpi=self._dpi)
                    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

                    log.debug(
                        "pdf.tesseract.page_raster",
                        file=file_path.name,
                        page=page_num,
                        width=img.width,
                        height=img.height,
                    )

                    # Auto-detect and correct orientation
                    img = _auto_rotate(img)

                    text = pytesseract.image_to_string(
                        img,
                        lang=self._tess_langs,
                        config=self._tess_config,
                    )

                    if text.strip():
                        yield ExtractedPage(page_number=page_num, text=text)
                    else:
                        log.warning(
                            "pdf.tesseract.empty_page",
                            file=str(file_path),
                            page=page_num,
                        )
                except Exception as exc:
                    log.warning(
                        "pdf.tesseract.page_failed",
                        file=str(file_path),
                        page=page_num,
                        error=str(exc),
                    )
                    continue
        finally:
            doc.close()

        log.debug("pdf.tesseract.done", file=str(file_path))
