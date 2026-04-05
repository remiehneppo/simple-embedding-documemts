"""
Ollama embedding client with batching and exponential-backoff retries.
"""

import asyncio

import httpx

from app.core.config import settings
from app.core.exceptions import EmbeddingServiceError
from app.core.logging import get_logger

log = get_logger(__name__)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts via the Ollama /api/embed endpoint.

    Texts are processed in batches of *embed_batch_size* to avoid timeouts.
    Each batch is retried up to *ollama_max_retries* times with exponential
    back-off on connection / timeout errors.
    """
    if not texts:
        return []

    all_embeddings: list[list[float]] = []
    batch_size = settings.embed_batch_size

    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start : batch_start + batch_size]
        log.debug(
            "embedder.batch",
            start=batch_start,
            size=len(batch),
            total=len(texts),
        )
        embeddings = await _embed_batch_with_retry(batch)
        all_embeddings.extend(embeddings)

    return all_embeddings


async def _embed_batch_with_retry(
    texts: list[str],
    attempt: int = 0,
) -> list[list[float]]:
    url = f"{settings.ollama_url}/api/embed"
    payload = {"model": settings.ollama_model, "input": texts}

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, timeout=settings.ollama_timeout)
            resp.raise_for_status()
            data = resp.json()

        embeddings = data.get("embeddings")
        if not embeddings or len(embeddings) != len(texts):
            raise EmbeddingServiceError(
                f"Expected {len(texts)} embeddings, got {len(embeddings or [])}"
            )
        return embeddings

    except httpx.HTTPStatusError as exc:
        log.error(
            "ollama.http_error",
            status=exc.response.status_code,
            body=exc.response.text[:200],
            exc_info=True,
        )
        raise EmbeddingServiceError(
            f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"
        ) from exc

    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        if attempt < settings.ollama_max_retries:
            wait = 2**attempt
            log.warning(
                "ollama.retry",
                attempt=attempt + 1,
                max=settings.ollama_max_retries,
                wait_s=wait,
                error=type(exc).__name__,
            )
            await asyncio.sleep(wait)
            return await _embed_batch_with_retry(texts, attempt + 1)

        log.error("ollama.max_retries_exceeded", url=url, error=str(exc), exc_info=True)
        raise EmbeddingServiceError(
            f"Cannot reach Ollama at {settings.ollama_url} after "
            f"{settings.ollama_max_retries} retries: {exc}"
        ) from exc

    except Exception as exc:
        log.error("ollama.unexpected_error", error=str(exc), exc_info=True)
        raise EmbeddingServiceError(f"Unexpected error: {exc}") from exc
