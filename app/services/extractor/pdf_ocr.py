import io
import os
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

# Must be set before PaddlePaddle is imported to disable the oneDNN backend
# which causes a NotImplementedError on certain CPUs with Paddle 3.x.
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import fitz  # PyMuPDF

from app.core.logging import get_logger

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image
from app.services.extractor.base import BaseExtractor, ExtractedPage

log = get_logger(__name__)


class PdfOcrExtractor(BaseExtractor):
    """Extract text from scanned PDF pages using PaddleOCR."""

    def __init__(self, langs: list[str] | None = None) -> None:
        self._langs = langs or ["vi", "ru"]
        self._ocr_cache: dict[str, object] = {}

    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def _get_ocr(self, lang: str):
        if lang not in self._ocr_cache:
            from paddleocr import PaddleOCR  # lazy import — heavy

            log.info("ocr.init", lang=lang)
            self._ocr_cache[lang] = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang=lang,
            )
        return self._ocr_cache[lang]

    def _run_ocr(self, lang: str, img_array) -> list[str]:
        """Run OCR and return a list of recognised text strings."""
        ocr = self._get_ocr(lang)
        texts: list[str] = []
        try:
            results = ocr.predict(img_array)
            log.debug(
                "ocr.raw_results",
                lang=lang,
                results_type=type(results).__name__,
                results_len=len(results) if results is not None else -1,
            )
            for i, result in enumerate(results or []):
                log.debug(
                    "ocr.result_item",
                    lang=lang,
                    index=i,
                    item_type=type(result).__name__,
                    item_repr=repr(result)[:300],
                )
                # PaddleOCR 3.x / PaddleX: OCRResult is a dict subclass.
                # Access rec_texts directly — do NOT call .json() on it.
                if isinstance(result, dict):
                    rec_texts = result.get("rec_texts", [])
                    log.debug("ocr.rec_texts", lang=lang, count=len(rec_texts), sample=rec_texts[:3])
                    texts.extend(t for t in rec_texts if t and t.strip())
                else:
                    # Fallback: legacy API returns list of [bbox, (text, score)]
                    try:
                        json_data = result.json()
                        if callable(json_data):
                            json_data = json_data()
                        rec_texts = json_data.get("rec_texts", []) if isinstance(json_data, dict) else []
                        log.debug("ocr.rec_texts_via_json", lang=lang, count=len(rec_texts))
                        texts.extend(t for t in rec_texts if t and t.strip())
                    except Exception:
                        if isinstance(result, list):
                            for line in result:
                                if line and len(line) >= 2:
                                    text_info = line[1]
                                    if isinstance(text_info, (list, tuple)) and text_info:
                                        texts.append(str(text_info[0]))
        except Exception as exc:
            log.error("ocr.predict_failed", lang=lang, error=str(exc), exc_info=True)
        log.debug("ocr.extracted_texts", lang=lang, count=len(texts), sample=texts[:3])
        return texts

    def extract(self, file_path: Path) -> Iterator[ExtractedPage]:
        log.debug("pdf.ocr.start", file=str(file_path), langs=self._langs)
        try:
            doc = fitz.open(str(file_path))
        except Exception as exc:
            log.error("pdf.ocr.open_failed", file=str(file_path), error=str(exc))
            raise

        try:
            for page_num, page in enumerate(doc, start=1):
                try:
                    import numpy as np  # noqa: PLC0415
                    from PIL import Image  # noqa: PLC0415

                    pix = page.get_pixmap(dpi=200)
                    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                    img_array = np.array(img)[:, :, ::-1]
                    log.debug(
                        "pdf.ocr.page_raster",
                        file=file_path.name,
                        page=page_num,
                        width=img.width,
                        height=img.height,
                        array_shape=list(img_array.shape),
                    )

                    all_texts: list[str] = []
                    for lang in self._langs:
                        all_texts.extend(self._run_ocr(lang, img_array))

                    # Deduplicate consecutive identical lines that multi-lang OCR can produce
                    seen: set[str] = set()
                    unique_texts = []
                    for t in all_texts:
                        if t not in seen:
                            seen.add(t)
                            unique_texts.append(t)

                    full_text = "\n".join(unique_texts)
                    if full_text.strip():
                        yield ExtractedPage(page_number=page_num, text=full_text)
                    else:
                        log.warning(
                            "pdf.ocr.empty_page",
                            file=str(file_path),
                            page=page_num,
                            langs=self._langs,
                        )
                except Exception as exc:
                    log.warning(
                        "pdf.ocr.page_failed",
                        file=str(file_path),
                        page=page_num,
                        error=str(exc),
                    )
                    continue
        finally:
            doc.close()

        log.debug("pdf.ocr.done", file=str(file_path))
