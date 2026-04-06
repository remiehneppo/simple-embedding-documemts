import aiofiles
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from typing import Annotated

from app.core.config import settings
from app.core.exceptions import DocumentNotFoundError
from app.core.logging import get_logger
from app.models.schemas import DeleteResponse, DocumentUploadResponse, DryRunResponse
from app.services.document_service import delete_document, dry_run_document, list_documents, process_document

router = APIRouter(prefix="/documents", tags=["documents"])
log = get_logger(__name__)

_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}
_VALID_OCR_LANGS = {"vie", "rus", "eng"}
_VALID_OCR_ENGINES = {"tesseract", "paddle"}


def _resolve_ocr_params(
    ocr_langs: list[str] | None,
    ocr_engine: str | None,
) -> tuple[list[str] | None, str | None]:
    """Validate and normalise OCR form params. Returns (resolved_langs, resolved_engine)."""
    resolved_langs: list[str] | None = None
    if ocr_langs:
        flat = [lang.strip().lower() for entry in ocr_langs for lang in entry.split(",") if lang.strip()]
        invalid = [l for l in flat if l not in _VALID_OCR_LANGS]
        if invalid:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown OCR language(s): {invalid}. Valid: {sorted(_VALID_OCR_LANGS)}",
            )
        resolved_langs = flat if flat else None

    resolved_engine: str | None = None
    if ocr_engine:
        ocr_engine = ocr_engine.strip().lower()
        if ocr_engine not in _VALID_OCR_ENGINES:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown OCR engine '{ocr_engine}'. Valid: {sorted(_VALID_OCR_ENGINES)}",
            )
        resolved_engine = ocr_engine

    return resolved_langs, resolved_engine


@router.post("/upload", response_model=DocumentUploadResponse | DryRunResponse, status_code=200)
async def upload_document(
    file: Annotated[UploadFile, File(...)],
    ocr_langs: Annotated[list[str] | None, Form()] = None,
    ocr_engine: Annotated[str | None, Form()] = None,
    dry_run: Annotated[str, Form()] = "false",
    dry_run_pages: Annotated[str, Form()] = "0",
):
    """Upload and ingest a document into the embedding store.

    Optional form fields:
    - **dry_run**: ``true`` / ``1`` — extract and chunk the document but do **not**
      embed or store it. Returns a ``DryRunResponse`` with per-page text and chunks.
    - **dry_run_pages**: limit OCR to the first N pages (0 = all pages). Only
      used when ``dry_run`` is ``true``.
    - **ocr_langs**: one or more language codes (``vie``, ``rus``, ``eng``).
      Can be repeated or comma-separated. Defaults to server config. (PDF only)
    - **ocr_engine**: ``tesseract`` (default) or ``paddle``. (PDF only)
    """
    # Explicitly coerce form strings — Pydantic v2 can silently keep bool=False
    # when the form sends the string "true".
    is_dry_run: bool = dry_run.strip().lower() in ("true", "1", "yes", "on")
    dry_run_pages_int: int = max(0, int(dry_run_pages.strip() or "0"))
    log.debug("api.upload.params", dry_run_raw=dry_run, is_dry_run=is_dry_run,
              dry_run_pages_raw=dry_run_pages, dry_run_pages_int=dry_run_pages_int)
    suffix = "." + (file.filename or "").rsplit(".", 1)[-1].lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"File type '{suffix}' is not supported. "
            f"Allowed: {sorted(_ALLOWED_EXTENSIONS)}",
        )

    resolved_langs, resolved_engine = _resolve_ocr_params(ocr_langs, ocr_engine)

    # For dry-run we use a temp file; for real ingestion we persist to documents_path.
    content = await file.read()

    if is_dry_run:
        import tempfile, os  # noqa: E401

        suffix_tmp = file.filename.rsplit(".", 1)[-1] if "." in (file.filename or "") else "bin"
        with tempfile.NamedTemporaryFile(suffix=f".{suffix_tmp}", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            log.info("api.dry_run", filename=file.filename, bytes=len(content),
                     ocr_langs=resolved_langs, ocr_engine=resolved_engine,
                     dry_run_pages=dry_run_pages_int)
            result = await dry_run_document(
                file_path=__import__("pathlib").Path(tmp_path),
                original_filename=file.filename,
                ocr_langs=resolved_langs,
                ocr_engine=resolved_engine,
                max_pages=dry_run_pages_int,
            )
        except Exception:
            log.error("api.dry_run_failed", filename=file.filename, exc_info=True)
            raise
        finally:
            os.unlink(tmp_path)
        return result

    settings.documents_path.mkdir(parents=True, exist_ok=True)
    dest = settings.documents_path / file.filename

    async with aiofiles.open(dest, "wb") as fh:
        await fh.write(content)

    log.info("api.upload", filename=file.filename, bytes=len(content),
             ocr_langs=resolved_langs, ocr_engine=resolved_engine)

    try:
        result = await process_document(
            dest,
            file.filename,
            ocr_langs=resolved_langs,
            ocr_engine=resolved_engine,
        )
    except Exception:
        log.error("api.upload_pipeline_failed", filename=file.filename, exc_info=True)
        raise
    return result


@router.get("/", response_model=list[dict])
async def get_documents(limit: int = 200):
    """List all indexed documents (one entry per document, not per chunk)."""
    return list_documents(limit=limit)


@router.delete("/{doc_id}", response_model=DeleteResponse)
async def remove_document(doc_id: str):
    """Delete a document and all its chunks from the store."""
    try:
        removed = delete_document(doc_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return DeleteResponse(deleted=True, doc_id=doc_id, chunks_removed=removed)


@router.get("/{doc_id}/file")
async def serve_document_file(doc_id: str):
    """Download the original file for a given document id."""
    from app.db.chroma import get_collection
    from pathlib import Path

    collection = get_collection()
    result = collection.get(where={"doc_id": doc_id}, limit=1, include=["metadatas"])
    if not result["ids"]:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")

    file_path = Path(result["metadatas"][0]["file_path"])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Original file not found on disk")

    return FileResponse(
        str(file_path),
        filename=result["metadatas"][0].get("file_name", file_path.name),
    )
