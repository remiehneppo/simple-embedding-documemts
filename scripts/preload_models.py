import os
import sys

# Must be set BEFORE PaddlePaddle is imported.
# Disables oneDNN backend which crashes on some CPUs with Paddle 3.x.
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
# Skip the HuggingFace connectivity pre-check; models are still downloaded
# if absent. At runtime (offline) this prevents network calls entirely.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# Tesseract lang code mapping (internal → tessdata name)
_TESS_LANG_MAP = {
    "vi":  "vie",
    "vie": "vie",
    "ru":  "rus",
    "rus": "rus",
    "en":  "eng",
    "eng": "eng",
}


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
    print(f"[preload/paddle] langs: {', '.join(langs)}", flush=True)

    for lang in langs:
        print(f"[preload/paddle] ── lang={lang}: initialising ...", flush=True)
        ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang=lang,
        )
        print(f"[preload/paddle] ── lang={lang}: running dummy predict ...", flush=True)
        try:
            _ = ocr.predict(dummy_img)
            print(f"[preload/paddle] ── lang={lang}: OK", flush=True)
        except Exception as exc:
            # A failed predict on a blank image is non-fatal (no text found),
            # but a crash here means the model weights are bad/missing.
            print(f"[preload/paddle] ── lang={lang}: predict error — {exc}", flush=True)
            raise


def _verify_tesseract_models() -> None:
    """
    Verify that Tesseract and the expected language data files are installed.
    Runs a quick image_to_string on a blank image for each language so that
    any missing tessdata is caught at build time, not at runtime.
    """
    import shutil
    import subprocess

    import numpy as np
    import pytesseract
    from PIL import Image

    # Check the binary is present
    if not shutil.which("tesseract"):
        raise RuntimeError(
            "tesseract binary not found — install tesseract-ocr in the Dockerfile"
        )

    version_out = subprocess.check_output(
        ["tesseract", "--version"], stderr=subprocess.STDOUT, text=True
    ).splitlines()[0]
    print(f"[preload/tesseract] {version_out}", flush=True)

    # Verify installed language packs
    available_raw = subprocess.check_output(
        ["tesseract", "--list-langs"], stderr=subprocess.STDOUT, text=True
    )
    available = set(available_raw.splitlines())
    print(f"[preload/tesseract] available langs: {', '.join(sorted(available))}", flush=True)

    langs = _langs_from_env()
    tess_langs = []
    for lang in langs:
        code = _TESS_LANG_MAP.get(lang.lower())
        if code and code not in tess_langs:
            tess_langs.append(code)

    for code in tess_langs:
        if code not in available:
            raise RuntimeError(
                f"Tesseract language pack '{code}' is not installed. "
                "Add tesseract-ocr-{code} to the apt-get install list."
            )

    # Run a dummy OCR call for each language to exercise the LSTM model
    dummy_img = Image.fromarray(np.full((64, 256, 3), 255, dtype=np.uint8))
    lang_str = "+".join(tess_langs) if tess_langs else "vie"
    print(f"[preload/tesseract] running dummy OCR for langs: {lang_str} ...", flush=True)
    try:
        pytesseract.image_to_string(dummy_img, lang=lang_str, config="--oem 1 --psm 6")
        print("[preload/tesseract] OK", flush=True)
    except Exception as exc:
        print(f"[preload/tesseract] error — {exc}", flush=True)
        raise


if __name__ == "__main__":
    try:
        _preload_paddle_models()
        print("[preload/paddle] all models ready", flush=True)
    except Exception as exc:
        print(f"[preload/paddle] FAILED: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)

    try:
        _verify_tesseract_models()
        print("[preload/tesseract] all language packs verified", flush=True)
    except Exception as exc:
        print(f"[preload/tesseract] FAILED: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
