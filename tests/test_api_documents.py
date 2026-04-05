"""
API-level tests for /documents endpoints.

Uses the async HTTPX client fixture (api_client from conftest) which:
  - mounts the real FastAPI app via ASGITransport
  - isolates ChromaDB to a tmp directory
  - mocks the Ollama embedder
"""

from __future__ import annotations

import io

import pytest


# ── Upload endpoint ───────────────────────────────────────────────────────────

class TestDocumentUpload:
    @pytest.mark.asyncio
    async def test_upload_txt_returns_200(self, api_client, tmp_path):
        p = tmp_path / "report.txt"
        p.write_text("This is a simple test document with enough words to chunk.")

        with open(p, "rb") as fh:
            resp = await api_client.post(
                "/documents/upload",
                files={"file": ("report.txt", fh, "text/plain")},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("processed", "duplicate")
        assert body["file_name"] == "report.txt"
        assert "doc_id" in body

    @pytest.mark.asyncio
    async def test_upload_md_returns_200(self, api_client, sample_md):
        with open(sample_md, "rb") as fh:
            resp = await api_client.post(
                "/documents/upload",
                files={"file": (sample_md.name, fh, "text/markdown")},
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "processed"

    @pytest.mark.asyncio
    async def test_upload_docx_returns_200(self, api_client, sample_docx):
        with open(sample_docx, "rb") as fh:
            resp = await api_client.post(
                "/documents/upload",
                files={"file": (sample_docx.name, fh, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "processed"

    @pytest.mark.asyncio
    async def test_upload_pdf_returns_200(self, api_client, sample_pdf_text):
        with open(sample_pdf_text, "rb") as fh:
            resp = await api_client.post(
                "/documents/upload",
                files={"file": (sample_pdf_text.name, fh, "application/pdf")},
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "processed"

    @pytest.mark.asyncio
    async def test_upload_unsupported_type_returns_422(self, api_client, tmp_path):
        p = tmp_path / "data.json"
        p.write_text('{"key": "value"}')
        with open(p, "rb") as fh:
            resp = await api_client.post(
                "/documents/upload",
                files={"file": ("data.json", fh, "application/json")},
            )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_upload_same_file_twice_is_duplicate(self, api_client, tmp_path):
        p = tmp_path / "dup.txt"
        p.write_text("Duplicate detection test with enough words here.")

        with open(p, "rb") as fh:
            resp1 = await api_client.post(
                "/documents/upload",
                files={"file": ("dup.txt", fh, "text/plain")},
            )
        with open(p, "rb") as fh:
            resp2 = await api_client.post(
                "/documents/upload",
                files={"file": ("dup.txt", fh, "text/plain")},
            )

        assert resp1.json()["status"] == "processed"
        assert resp2.json()["status"] == "duplicate"

    @pytest.mark.asyncio
    async def test_upload_response_schema(self, api_client, tmp_path):
        p = tmp_path / "schema_test.txt"
        p.write_text("Testing the response schema fields thoroughly.")
        with open(p, "rb") as fh:
            resp = await api_client.post(
                "/documents/upload",
                files={"file": ("schema_test.txt", fh, "text/plain")},
            )
        body = resp.json()
        assert set(body.keys()) >= {"doc_id", "file_name", "chunks", "pages", "status"}
        assert isinstance(body["chunks"], int)
        assert isinstance(body["pages"], int)


# ── List documents endpoint ───────────────────────────────────────────────────

class TestListDocuments:
    @pytest.mark.asyncio
    async def test_empty_list(self, api_client):
        resp = await api_client.get("/documents/")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_uploaded_document_appears_in_list(self, api_client, tmp_path):
        p = tmp_path / "listed.txt"
        p.write_text("This document should appear in the list endpoint output.")
        with open(p, "rb") as fh:
            await api_client.post(
                "/documents/upload",
                files={"file": ("listed.txt", fh, "text/plain")},
            )

        resp = await api_client.get("/documents/")
        assert resp.status_code == 200
        names = [d["file_name"] for d in resp.json()]
        assert "listed.txt" in names

    @pytest.mark.asyncio
    async def test_two_documents_both_listed(self, api_client, tmp_path):
        for name in ("alpha.txt", "beta.txt"):
            p = tmp_path / name
            p.write_text(f"Content of {name} document with enough words here.")
            with open(p, "rb") as fh:
                await api_client.post(
                    "/documents/upload",
                    files={"file": (name, fh, "text/plain")},
                )

        resp = await api_client.get("/documents/")
        names = {d["file_name"] for d in resp.json()}
        assert "alpha.txt" in names
        assert "beta.txt" in names

    @pytest.mark.asyncio
    async def test_limit_query_param(self, api_client, tmp_path):
        for i in range(3):
            p = tmp_path / f"doc{i}.txt"
            p.write_text(f"Document {i} unique content with enough words present.")
            with open(p, "rb") as fh:
                await api_client.post(
                    "/documents/upload",
                    files={"file": (p.name, fh, "text/plain")},
                )

        resp = await api_client.get("/documents/?limit=2")
        assert resp.status_code == 200
        # limit is applied at the chunk level, not doc level; just check no error
        assert isinstance(resp.json(), list)


# ── Delete endpoint ───────────────────────────────────────────────────────────

class TestDeleteDocument:
    @pytest.mark.asyncio
    async def test_delete_existing_document(self, api_client, tmp_path):
        p = tmp_path / "to_delete.txt"
        p.write_text("This document will be deleted via the API endpoint.")
        with open(p, "rb") as fh:
            upload_resp = await api_client.post(
                "/documents/upload",
                files={"file": ("to_delete.txt", fh, "text/plain")},
            )
        doc_id = upload_resp.json()["doc_id"]

        del_resp = await api_client.delete(f"/documents/{doc_id}")
        assert del_resp.status_code == 200
        body = del_resp.json()
        assert body["deleted"] is True
        assert body["doc_id"] == doc_id
        assert body["chunks_removed"] > 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_404(self, api_client):
        resp = await api_client.delete("/documents/nonexistent-doc-id-xyz")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_deleted_document_gone_from_list(self, api_client, tmp_path):
        p = tmp_path / "gone.txt"
        p.write_text("This document should disappear from the listing after deletion.")
        with open(p, "rb") as fh:
            upload = await api_client.post(
                "/documents/upload",
                files={"file": ("gone.txt", fh, "text/plain")},
            )
        doc_id = upload.json()["doc_id"]
        await api_client.delete(f"/documents/{doc_id}")

        list_resp = await api_client.get("/documents/")
        names = [d["file_name"] for d in list_resp.json()]
        assert "gone.txt" not in names

    @pytest.mark.asyncio
    async def test_delete_response_schema(self, api_client, tmp_path):
        p = tmp_path / "schema.txt"
        p.write_text("Schema validation test document with enough words.")
        with open(p, "rb") as fh:
            upload = await api_client.post(
                "/documents/upload",
                files={"file": ("schema.txt", fh, "text/plain")},
            )
        doc_id = upload.json()["doc_id"]
        resp = await api_client.delete(f"/documents/{doc_id}")
        body = resp.json()
        assert set(body.keys()) >= {"deleted", "doc_id", "chunks_removed"}
        assert isinstance(body["chunks_removed"], int)


# ── Serve file endpoint ───────────────────────────────────────────────────────

class TestServeFile:
    @pytest.mark.asyncio
    async def test_serve_file_returns_200(self, api_client, tmp_path, monkeypatch):
        """File must exist on disk at the path stored in metadata."""
        from app.core import config

        # Override documents_path so the uploaded file lands in tmp_path/documents
        monkeypatch.setattr(config.settings, "documents_path", tmp_path / "documents")
        (tmp_path / "documents").mkdir(exist_ok=True)

        p = tmp_path / "documents" / "serveable.txt"
        p.write_text("Serveable file content for download testing here.")

        with open(p, "rb") as fh:
            upload = await api_client.post(
                "/documents/upload",
                files={"file": ("serveable.txt", fh, "text/plain")},
            )
        doc_id = upload.json()["doc_id"]

        resp = await api_client.get(f"/documents/{doc_id}/file")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_serve_nonexistent_doc_returns_404(self, api_client):
        resp = await api_client.get("/documents/nonexistent-xyz/file")
        assert resp.status_code == 404
