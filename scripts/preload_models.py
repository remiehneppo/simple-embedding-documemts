import os
import sys

# Must be set BEFORE PaddlePaddle is imported.
# Disables oneDNN backend which crashes on some CPUs with Paddle 3.x.
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
# Skip the HuggingFace connectivity pre-check; models are still downloaded
# if absent. At runtime (offline) this prevents network calls entirely.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def _langs_from_env() -> list[str]:
    raw = os.environ.get("OCR_LANGS", "vi,ru,en")
    langs = [item.strip() for item in raw.split(",") if item.strip()]
    return langs or ["vi", "ru", "en"]


def _preload_paddle_models() -> None:
    """
    Initialise PaddleOCR for every requested language and run a dummy
    prediction.  Running predict() is essential — it triggers the full
    download of both the text-detection model (PP-OCRv5_server_det/mobile_det)
    and the language-specific text-recognition model so all weights are baked
    into the image and available offline.
    """
    import numpy as np
    from paddleocr import PaddleOCR

    # Minimal white image — just large enough that the detector finds nothing
    # but still exercises the full pipeline end-to-end.
    dummy_img = np.full((64, 256, 3), 255, dtype=np.uint8)

    langs = _langs_from_env()
    print(f"[preload] langs: {', '.join(langs)}", flush=True)

    for lang in langs:
        print(f"[preload] ── lang={lang}: initialising ...", flush=True)
        ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang=lang,
        )
        print(f"[preload] ── lang={lang}: running dummy predict ...", flush=True)
        try:
            _ = ocr.predict(dummy_img)
            print(f"[preload] ── lang={lang}: OK", flush=True)
        except Exception as exc:
            # A failed predict on a blank image is non-fatal (no text found),
            # but a crash here means the model weights are bad/missing.
            print(f"[preload] ── lang={lang}: predict error — {exc}", flush=True)
            raise


if __name__ == "__main__":
    try:
        _preload_paddle_models()
        print("[preload] all models ready", flush=True)
    except Exception as exc:
        print(f"[preload] FAILED: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
