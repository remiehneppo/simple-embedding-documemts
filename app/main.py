from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import documents, search
from app.core.config import settings
from app.core.exceptions import (
    DocumentNotFoundError,
    EmbeddingServiceError,
    ExtractionError,
    UnsupportedFileTypeError,
    document_not_found_handler,
    embedding_service_error_handler,
    extraction_error_handler,
    unsupported_file_type_handler,
)
from app.core.logging import get_logger, setup_logging

setup_logging(log_level=settings.log_level, log_path=settings.log_path)
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    settings.documents_path.mkdir(parents=True, exist_ok=True)
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    settings.log_path.mkdir(parents=True, exist_ok=True)
    log.info("app.started", host=settings.host, port=settings.port)
    yield


app = FastAPI(
    title=settings.app_name,
    description="Semantic document search powered by Ollama embeddings + ChromaDB",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (allow all origins — restrict in production) ────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Custom exception handlers ────────────────────────────────────────────────
app.add_exception_handler(UnsupportedFileTypeError, unsupported_file_type_handler)
app.add_exception_handler(ExtractionError, extraction_error_handler)
app.add_exception_handler(EmbeddingServiceError, embedding_service_error_handler)
app.add_exception_handler(DocumentNotFoundError, document_not_found_handler)


# ── Global catch-all handler (unhandled 500 errors) ──────────────────────────
async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.error(
        "unhandled_exception",
        method=request.method,
        path=str(request.url.path),
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "detail": "An unexpected error occurred"},
    )


app.add_exception_handler(Exception, _global_exception_handler)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(documents.router)
app.include_router(search.router)


# ── Health endpoint ───────────────────────────────────────────────────────────
@app.get("/health", tags=["health"])
async def health():
    ollama_ok = False
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{settings.ollama_url}/api/tags", timeout=3.0)
            ollama_ok = r.status_code == 200
    except Exception:
        log.warning("health.ollama_unreachable", url=settings.ollama_url, exc_info=True)

    chroma_ok = False
    try:
        from app.db.chroma import get_collection

        get_collection()
        chroma_ok = True
    except Exception:
        log.warning("health.chroma_unreachable", exc_info=True)

    return {
        "status": "ok" if (ollama_ok and chroma_ok) else "degraded",
        "ollama_reachable": ollama_ok,
        "chroma_reachable": chroma_ok,
        "version": "1.0.0",
    }


# ── Static frontend (served last so API routes take priority) ─────────────────
_frontend = Path(__file__).parent.parent / "frontend"
if _frontend.exists():
    app.mount("/", StaticFiles(directory=str(_frontend), html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
