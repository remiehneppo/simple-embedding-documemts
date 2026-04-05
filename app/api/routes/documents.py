import aiofiles
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.exceptions import DocumentNotFoundError
from app.core.logging import get_logger
from app.models.schemas import DeleteResponse, DocumentUploadResponse
from app.services.document_service import delete_document, list_documents, process_document

router = APIRouter(prefix="/documents", tags=["documents"])
log = get_logger(__name__)

_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}


@router.post("/upload", response_model=DocumentUploadResponse, status_code=200)
async def upload_document(file: UploadFile = File(...)):
    """Upload and ingest a document into the embedding store."""
    suffix = "." + (file.filename or "").rsplit(".", 1)[-1].lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"File type '{suffix}' is not supported. "
            f"Allowed: {sorted(_ALLOWED_EXTENSIONS)}",
        )

    settings.documents_path.mkdir(parents=True, exist_ok=True)
    dest = settings.documents_path / file.filename

    content = await file.read()
    async with aiofiles.open(dest, "wb") as fh:
        await fh.write(content)

    log.info("api.upload", filename=file.filename, bytes=len(content))

    try:
        result = await process_document(dest, file.filename)
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
