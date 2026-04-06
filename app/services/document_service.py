"""
Full processing pipeline: upload → extract → clean → chunk → embed → store.
"""

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path

import structlog

from app.core.config import settings
from app.core.exceptions import ExtractionError, UnsupportedFileTypeError
from app.core.logging import get_logger
from app.db.chroma import get_collection
from app.services.chunker import chunk_text
from app.services.cleaner import clean_text
from app.services.embedder import embed_texts
from app.services.extractor_factory import get_extractor
from app.services.extractor.pdf_text import PdfTextExtractor
from app.services.extractor.pdf_ocr import PdfOcrExtractor
from app.services.extractor.pdf_tesseract import PdfTesseractExtractor

log = get_logger(__name__)


def _compute_sha256(file_path: Path) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as fh:
        while chunk := fh.read(65_536):
            sha.update(chunk)
    return sha.hexdigest()


async def process_document(
    file_path: Path,
    original_filename: str,
    ocr_langs: list[str] | None = None,
    ocr_engine: str | None = None,
) -> dict:
    """
    Orchestrate the full ingestion pipeline for one document.

    Returns a dict compatible with DocumentUploadResponse.
    """
    doc_id = str(uuid.uuid4())

    # Attach doc_id to every log line emitted during this pipeline run
    structlog.contextvars.bind_contextvars(doc_id=doc_id)
    log.info("document.start", filename=original_filename, path=str(file_path))

    try:
        return await _run_pipeline(doc_id, file_path, original_filename, ocr_langs, ocr_engine)
    except Exception:
        raise
    finally:
        structlog.contextvars.clear_contextvars()


async def _run_pipeline(
    doc_id: str,
    file_path: Path,
    original_filename: str,
    ocr_langs: list[str] | None = None,
    ocr_engine: str | None = None,
) -> dict:
    # ── 1. Deduplication check (SHA-256) ────────────────────────────────────
    file_hash = _compute_sha256(file_path)
    collection = get_collection()

    existing = collection.get(
        where={"file_hash": file_hash},
        limit=1,
        include=[],
    )
    if existing["ids"]:
        log.info("document.duplicate", filename=original_filename, hash=file_hash[:12])
        return {
            "doc_id": doc_id,
            "file_name": original_filename,
            "chunks": 0,
            "pages": 0,
            "status": "duplicate",
        }

    # ── 2. Select extractor ─────────────────────────────────────────────────
    try:
        extractor = get_extractor(file_path, ocr_langs=ocr_langs, ocr_engine=ocr_engine)
    except UnsupportedFileTypeError:
        log.error("document.unsupported_type", filename=original_filename, exc_info=True)
        raise

    # ── 3. Extract pages ────────────────────────────────────────────────────
    pages = []
    try:
        for page in extractor.extract(file_path):
            pages.append(page)
    except Exception as exc:
        log.error(
            "document.extraction_failed",
            filename=original_filename,
            error=str(exc),
            exc_info=True,
        )
        raise ExtractionError(str(file_path), str(exc)) from exc

    # ── 3b. Fallback: if PdfTextExtractor yielded nothing, retry with OCR ───
    # This handles scanned PDFs that slipped through the has_text_layer check.
    if not pages and isinstance(extractor, PdfTextExtractor):
        langs = ocr_langs or [lang.strip() for lang in settings.ocr_langs.split(",") if lang.strip()]
        engine = (ocr_engine or settings.ocr_engine).lower()
        log.warning(
            "document.text_extraction_empty_fallback_ocr",
            filename=original_filename,
            langs=langs,
            engine=engine,
        )
        ocr_extractor: PdfTesseractExtractor | PdfOcrExtractor
        if engine == "tesseract":
            ocr_extractor = PdfTesseractExtractor(langs=langs)
        else:
            ocr_extractor = PdfOcrExtractor(langs=langs)
        try:
            for page in ocr_extractor.extract(file_path):
                pages.append(page)
        except Exception as exc:
            log.error(
                "document.ocr_fallback_failed",
                filename=original_filename,
                error=str(exc),
                exc_info=True,
            )
            raise ExtractionError(str(file_path), str(exc)) from exc

    log.info("document.extracted", filename=original_filename, pages=len(pages))

    # ── 4. Clean + chunk ────────────────────────────────────────────────────
    raw_chunks: list[dict] = []
    for page in pages:
        cleaned = clean_text(page.text)
        if not cleaned:
            log.debug("document.page_empty_after_clean", page=page.page_number)
            continue
        page_chunks = chunk_text(
            cleaned,
            max_words=settings.chunk_max_words,
            overlap_words=settings.chunk_overlap_words,
            min_words=settings.chunk_min_words,
        )
        for c in page_chunks:
            raw_chunks.append({"text": c, "page_number": page.page_number})

    if not raw_chunks:
        log.warning("document.no_chunks", filename=original_filename)
        return {
            "doc_id": doc_id,
            "file_name": original_filename,
            "chunks": 0,
            "pages": len(pages),
            "status": "empty",
        }

    log.info(
        "document.chunked",
        filename=original_filename,
        pages=len(pages),
        chunks=len(raw_chunks),
    )

    # ── 5. Embed ─────────────────────────────────────────────────────────────
    texts = [c["text"] for c in raw_chunks]
    embeddings = await embed_texts(texts)
    log.info("document.embedded", filename=original_filename, vectors=len(embeddings))

    # ── 6. Store in ChromaDB ─────────────────────────────────────────────────
    now = datetime.now(timezone.utc).isoformat()
    total_chunks = len(raw_chunks)
    ids = [f"{doc_id}_chunk_{i}" for i in range(total_chunks)]

    metadatas = [
        {
            "doc_id": doc_id,
            "file_name": original_filename,
            "file_path": str(file_path),
            "file_type": file_path.suffix.lower().lstrip("."),
            "file_hash": file_hash,
            "page_number": c["page_number"],
            "chunk_index": i,
            "total_chunks": total_chunks,
            "page_count": len(pages),
            "upload_ts": now,
        }
        for i, c in enumerate(raw_chunks)
    ]

    try:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
    except Exception as exc:
        log.error(
            "document.store_failed",
            filename=original_filename,
            error=str(exc),
            exc_info=True,
        )
        raise

    log.info(
        "document.done",
        filename=original_filename,
        doc_id=doc_id,
        chunks=total_chunks,
        pages=len(pages),
    )

    return {
        "doc_id": doc_id,
        "file_name": original_filename,
        "chunks": total_chunks,
        "pages": len(pages),
        "status": "processed",
    }


def list_documents(limit: int = 200) -> list[dict]:
    """Return one metadata record per unique document (deduplicated by doc_id)."""
    collection = get_collection()
    result = collection.get(limit=limit, include=["metadatas"])

    seen: dict[str, dict] = {}
    for meta in result["metadatas"]:
        doc_id = meta.get("doc_id", "")
        if doc_id and doc_id not in seen:
            seen[doc_id] = meta
    return list(seen.values())


async def dry_run_document(
    file_path: Path,
    original_filename: str,
    ocr_langs: list[str] | None = None,
    ocr_engine: str | None = None,
    max_pages: int = 0,
) -> dict:
    """
    Extract, clean, and chunk a document without embedding or storing anything.

    Args:
        max_pages: Limit extraction to the first N pages. 0 means all pages.

    Returns a dict compatible with DryRunResponse.
    """
    resolved_langs = ocr_langs or [l.strip() for l in settings.ocr_langs.split(",") if l.strip()]
    resolved_engine = (ocr_engine or settings.ocr_engine).lower()

    # Safety guard: this function must NEVER write to the database.
    # If called from within a context that has already bound doc_id (i.e. from
    # _run_pipeline), something is wrong — bail out immediately.
    structlog.contextvars.bind_contextvars(dry_run=True)
    log.info("document.dry_run.start", filename=original_filename)

    try:
        # ── Select extractor ────────────────────────────────────────────────
        extractor = get_extractor(file_path, ocr_langs=ocr_langs, ocr_engine=ocr_engine)

        # ── Extract pages ───────────────────────────────────────────────────
        pages = []
        try:
            for page in extractor.extract(file_path):
                pages.append(page)
        except Exception as exc:
            raise ExtractionError(str(file_path), str(exc)) from exc

        # Fallback OCR (same logic as main pipeline)
        if not pages and isinstance(extractor, PdfTextExtractor):
            log.warning("document.dry_run.fallback_ocr", filename=original_filename)
            ocr_extractor: PdfTesseractExtractor | PdfOcrExtractor
            if resolved_engine == "tesseract":
                ocr_extractor = PdfTesseractExtractor(langs=resolved_langs)
            else:
                ocr_extractor = PdfOcrExtractor(langs=resolved_langs)
            try:
                for page in ocr_extractor.extract(file_path):
                    pages.append(page)
            except Exception as exc:
                raise ExtractionError(str(file_path), str(exc)) from exc

        # Apply page limit
        if max_pages and max_pages > 0:
            pages = pages[:max_pages]
            log.debug("document.dry_run.page_limit", max_pages=max_pages, actual=len(pages))

        # ── Clean + chunk (no embed, no store) ──────────────────────────────
        from app.models.schemas import DryRunPage

        preview: list[DryRunPage] = []
        total_chunks = 0
        for page in pages:
            cleaned = clean_text(page.text)
            if not cleaned:
                preview.append(DryRunPage(
                    page_number=page.page_number,
                    raw_text=page.text,
                    chunks=[],
                ))
                continue
            page_chunks = chunk_text(
                cleaned,
                max_words=settings.chunk_max_words,
                overlap_words=settings.chunk_overlap_words,
                min_words=settings.chunk_min_words,
            )
            total_chunks += len(page_chunks)
            preview.append(DryRunPage(
                page_number=page.page_number,
                raw_text=page.text,
                chunks=page_chunks,
            ))

        log.info(
            "document.dry_run.done",
            filename=original_filename,
            pages=len(pages),
            chunks=total_chunks,
        )

        return {
            "file_name": original_filename,
            "ocr_engine": resolved_engine,
            "ocr_langs": resolved_langs,
            "pages": len(pages),
            "total_chunks": total_chunks,
            "preview": [p.model_dump() for p in preview],
        }
    finally:
        structlog.contextvars.clear_contextvars()


def delete_document(doc_id: str) -> int:
    """Delete all chunks for a document. Returns the number of chunks removed."""
    from app.core.exceptions import DocumentNotFoundError

    collection = get_collection()
    existing = collection.get(where={"doc_id": doc_id}, include=[])
    if not existing["ids"]:
        raise DocumentNotFoundError(doc_id)

    collection.delete(where={"doc_id": doc_id})
    removed = len(existing["ids"])
    log.info("document.deleted", doc_id=doc_id, chunks_removed=removed)
    return removed
