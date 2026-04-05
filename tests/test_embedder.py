"""
Unit tests for app.services.embedder

The Ollama HTTP API is mocked with unittest.mock so no real server is needed.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.core.exceptions import EmbeddingServiceError
from app.services.embedder import _embed_batch_with_retry, embed_texts


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_response(embeddings: list[list[float]], status: int = 200) -> MagicMock:
    """Build a fake httpx.Response-like object."""
    mock_resp = MagicMock()
    mock_resp.status_code = status
    mock_resp.json.return_value = {"embeddings": embeddings}
    mock_resp.raise_for_status = MagicMock()
    if status >= 400:
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=mock_resp
        )
        mock_resp.text = f"HTTP {status} error"
    return mock_resp


def _make_context_manager(response):
    """Return an async context manager that yields a client whose post() returns response."""
    client_mock = AsyncMock()
    client_mock.post = AsyncMock(return_value=response)

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=client_mock)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


# ── embed_texts (public API) ─────────────────────────────────────────────────

class TestEmbedTexts:
    @pytest.mark.asyncio
    async def test_empty_list_returns_empty(self):
        result = await embed_texts([])
        assert result == []

    @pytest.mark.asyncio
    async def test_single_text_returns_one_vector(self):
        fake_vec = [0.1, 0.2, 0.3]
        resp = _make_response([fake_vec])

        with patch("httpx.AsyncClient", return_value=_make_context_manager(resp)):
            result = await embed_texts(["hello"])

        assert result == [fake_vec]

    @pytest.mark.asyncio
    async def test_multiple_texts_returns_correct_count(self):
        texts = ["one", "two", "three"]
        fake_vecs = [[float(i)] * 4 for i in range(3)]
        resp = _make_response(fake_vecs)

        with patch("httpx.AsyncClient", return_value=_make_context_manager(resp)):
            result = await embed_texts(texts)

        assert len(result) == 3
        assert result == fake_vecs

    @pytest.mark.asyncio
    async def test_batching_makes_multiple_requests(self, monkeypatch):
        """Texts > batch_size should trigger multiple HTTP requests."""
        from app.core import config

        monkeypatch.setattr(config.settings, "embed_batch_size", 2)

        texts = ["a", "b", "c", "d", "e"]
        # Each call returns 2 or 1 embeddings depending on batch size
        call_count = [0]

        async def post_side_effect(*args, **kwargs):
            batch = kwargs.get("json", {}).get("input", [])
            vecs = [[float(i + call_count[0])] * 4 for i in range(len(batch))]
            call_count[0] += len(batch)
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {"embeddings": vecs}
            return resp

        client_mock = AsyncMock()
        client_mock.post = post_side_effect

        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=client_mock)
        cm.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=cm):
            result = await embed_texts(texts)

        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_http_error_raises_embedding_service_error(self):
        resp = _make_response([], status=503)

        with patch("httpx.AsyncClient", return_value=_make_context_manager(resp)):
            with pytest.raises(EmbeddingServiceError) as exc_info:
                await embed_texts(["hello"])

        assert "503" in str(exc_info.value) or "Embedding service error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_wrong_embedding_count_raises(self):
        """If Ollama returns fewer embeddings than texts, raise."""
        # Send 2 texts but get back only 1 embedding
        resp = _make_response([[0.1, 0.2]])  # only 1 for 2 texts

        with patch("httpx.AsyncClient", return_value=_make_context_manager(resp)):
            with pytest.raises(EmbeddingServiceError) as exc_info:
                await embed_texts(["text1", "text2"])

        assert "Expected 2" in str(exc_info.value)


# ── Retry logic ───────────────────────────────────────────────────────────────

class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_retries_on_connect_error(self, monkeypatch):
        """Succeeds on the second attempt after a ConnectError."""
        from app.core import config

        monkeypatch.setattr(config.settings, "ollama_max_retries", 3)
        fake_vec = [1.0, 0.0]

        attempt = [0]

        async def post_side_effect(*args, **kwargs):
            attempt[0] += 1
            if attempt[0] < 2:
                raise httpx.ConnectError("refused")
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {"embeddings": [fake_vec]}
            return resp

        client_mock = AsyncMock()
        client_mock.post = post_side_effect

        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=client_mock)
        cm.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=cm):
            with patch("asyncio.sleep", new_callable=AsyncMock):  # skip real sleep
                result = await embed_texts(["hello"])

        assert result == [fake_vec]
        assert attempt[0] == 2

    @pytest.mark.asyncio
    async def test_max_retries_exceeded_raises(self, monkeypatch):
        """After exceeding max_retries, EmbeddingServiceError is raised."""
        from app.core import config

        monkeypatch.setattr(config.settings, "ollama_max_retries", 2)

        async def always_fails(*args, **kwargs):
            raise httpx.ConnectError("refused")

        client_mock = AsyncMock()
        client_mock.post = always_fails

        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=client_mock)
        cm.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=cm):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(EmbeddingServiceError) as exc_info:
                    await embed_texts(["hello"])

        assert "retries" in str(exc_info.value).lower() or "Embedding service error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_timeout_triggers_retry(self, monkeypatch):
        from app.core import config

        monkeypatch.setattr(config.settings, "ollama_max_retries", 2)

        attempt = [0]
        fake_vec = [0.5, 0.5]

        async def post_side_effect(*args, **kwargs):
            attempt[0] += 1
            if attempt[0] == 1:
                raise httpx.TimeoutException("timeout")
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {"embeddings": [fake_vec]}
            return resp

        client_mock = AsyncMock()
        client_mock.post = post_side_effect

        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=client_mock)
        cm.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=cm):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await embed_texts(["hello"])

        assert result == [fake_vec]

    @pytest.mark.asyncio
    async def test_unexpected_exception_raises(self):
        async def boom(*args, **kwargs):
            raise ValueError("unexpected")

        client_mock = AsyncMock()
        client_mock.post = boom

        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=client_mock)
        cm.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=cm):
            with pytest.raises(EmbeddingServiceError) as exc_info:
                await embed_texts(["hello"])

        assert "Unexpected" in str(exc_info.value)
