from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.logging import get_logger

log = get_logger(__name__)


class UnsupportedFileTypeError(Exception):
    def __init__(self, file_type: str) -> None:
        self.file_type = file_type
        super().__init__(f"Unsupported file type: '{file_type}'")


class EmbeddingServiceError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(f"Embedding service error: {message}")


class DocumentNotFoundError(Exception):
    def __init__(self, doc_id: str) -> None:
        self.doc_id = doc_id
        super().__init__(f"Document not found: '{doc_id}'")


class ExtractionError(Exception):
    def __init__(self, file_path: str, reason: str) -> None:
        self.file_path = file_path
        self.reason = reason
        super().__init__(f"Extraction failed for '{file_path}': {reason}")


# ── FastAPI exception handlers ──────────────────────────────────────────────

async def unsupported_file_type_handler(
    request: Request, exc: UnsupportedFileTypeError
) -> JSONResponse:
    log.warning(
        "exception.unsupported_file_type",
        path=str(request.url),
        file_type=exc.file_type,
        exc_info=True,
    )
    return JSONResponse(
        status_code=422,
        content={"error": "unsupported_file_type", "detail": str(exc)},
    )


async def extraction_error_handler(
    request: Request, exc: ExtractionError
) -> JSONResponse:
    log.error(
        "exception.extraction_failed",
        path=str(request.url),
        file_path=exc.file_path,
        reason=exc.reason,
        exc_info=True,
    )
    return JSONResponse(
        status_code=422,
        content={"error": "extraction_failed", "detail": str(exc)},
    )


async def embedding_service_error_handler(
    request: Request, exc: EmbeddingServiceError
) -> JSONResponse:
    log.error(
        "exception.embedding_service_error",
        path=str(request.url),
        exc_info=True,
    )
    return JSONResponse(
        status_code=503,
        content={"error": "embedding_service_unavailable", "detail": str(exc)},
    )


async def document_not_found_handler(
    request: Request, exc: DocumentNotFoundError
) -> JSONResponse:
    log.warning(
        "exception.document_not_found",
        path=str(request.url),
        doc_id=exc.doc_id,
        exc_info=True,
    )
    return JSONResponse(
        status_code=404,
        content={"error": "document_not_found", "detail": str(exc)},
    )
