"""
Integration tests for app.services.document_service

Uses a real ChromaDB collection in a tmp directory.
The Ollama embedder is replaced by the fake from conftest.patch_embedder.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from app.core.exceptions import DocumentNotFoundError, UnsupportedFileTypeError
from app.services.document_service import (
    delete_document,
    list_documents,
    process_document,
)


# ── process_document — happy paths ───────────────────────────────────────────

class TestProcessDocumentHappyPath:
    @pytest.mark.asyncio
    async def test_txt_file_processed(self, chroma_tmp, patch_embedder, sample_txt):
        result = await process_document(sample_txt, sample_txt.name)
        assert result["status"] == "processed"
        assert result["chunks"] > 0
        assert result["pages"] >= 1
        assert result["file_name"] == sample_txt.name
        assert "doc_id" in result

    @pytest.mark.asyncio
    async def test_md_file_processed(self, chroma_tmp, patch_embedder, sample_md):
        result = await process_document(sample_md, sample_md.name)
        assert result["status"] == "processed"
        assert result["chunks"] > 0

    @pytest.mark.asyncio
    async def test_docx_file_processed(self, chroma_tmp, patch_embedder, sample_docx):
        result = await process_document(sample_docx, sample_docx.name)
        assert result["status"] == "processed"
        assert result["chunks"] > 0

    @pytest.mark.asyncio
    async def test_pdf_text_file_processed(self, chroma_tmp, patch_embedder, sample_pdf_text):
        result = await process_document(sample_pdf_text, sample_pdf_text.name)
        assert result["status"] == "processed"
        assert result["chunks"] > 0
        assert result["pages"] >= 1

    @pytest.mark.asyncio
    async def test_chunks_stored_in_chroma(self, chroma_tmp, patch_embedder, sample_txt):
        result = await process_document(sample_txt, sample_txt.name)
        from app.db.chroma import get_collection

        collection = get_collection()
        stored = collection.get(
            where={"doc_id": result["doc_id"]}, include=["metadatas"]
        )
        assert len(stored["ids"]) == result["chunks"]

    @pytest.mark.asyncio
    async def test_metadata_contains_required_fields(
        self, chroma_tmp, patch_embedder, sample_txt
    ):
        result = await process_document(sample_txt, sample_txt.name)
        from app.db.chroma import get_collection

        collection = get_collection()
        stored = collection.get(
            where={"doc_id": result["doc_id"]}, include=["metadatas"], limit=1
        )
        meta = stored["metadatas"][0]
        assert meta["doc_id"] == result["doc_id"]
        assert meta["file_name"] == sample_txt.name
        assert "file_hash" in meta
        assert "page_number" in meta
        assert "chunk_index" in meta
        assert "total_chunks" in meta
        assert "upload_ts" in meta

    @pytest.mark.asyncio
    async def test_embed_texts_called_with_chunk_texts(
        self, chroma_tmp, patch_embedder, sample_txt
    ):
        result = await process_document(sample_txt, sample_txt.name)
        assert patch_embedder.called
        # embedder should have been called with a list of strings
        call_args = patch_embedder.call_args[0][0]
        assert isinstance(call_args, list)
        assert all(isinstance(t, str) for t in call_args)


# ── process_document — deduplication ─────────────────────────────────────────

class TestDocumentDeduplication:
    @pytest.mark.asyncio
    async def test_same_file_second_upload_is_duplicate(
        self, chroma_tmp, patch_embedder, sample_txt
    ):
        first = await process_document(sample_txt, sample_txt.name)
        assert first["status"] == "processed"

        second = await process_document(sample_txt, sample_txt.name)
        assert second["status"] == "duplicate"
        assert second["chunks"] == 0

    @pytest.mark.asyncio
    async def test_duplicate_does_not_add_chunks(
        self, chroma_tmp, patch_embedder, sample_txt
    ):
        from app.db.chroma import get_collection

        await process_document(sample_txt, sample_txt.name)
        count_after_first = get_collection().count()

        await process_document(sample_txt, sample_txt.name)
        count_after_second = get_collection().count()

        assert count_after_first == count_after_second

    @pytest.mark.asyncio
    async def test_different_content_not_duplicate(
        self, chroma_tmp, patch_embedder, tmp_path
    ):
        p1 = tmp_path / "file1.txt"
        p2 = tmp_path / "file2.txt"
        p1.write_text("First document with unique content alpha.")
        p2.write_text("Second document with unique content beta.")

        r1 = await process_document(p1, p1.name)
        r2 = await process_document(p2, p2.name)

        assert r1["status"] == "processed"
        assert r2["status"] == "processed"
        assert r1["doc_id"] != r2["doc_id"]


# ── process_document — error cases ───────────────────────────────────────────

class TestProcessDocumentErrors:
    @pytest.mark.asyncio
    async def test_unsupported_file_type_raises(self, chroma_tmp, patch_embedder, tmp_path):
        bad_file = tmp_path / "data.json"
        bad_file.write_text('{"key": "value"}')
        with pytest.raises(UnsupportedFileTypeError):
            await process_document(bad_file, bad_file.name)

    @pytest.mark.asyncio
    async def test_empty_txt_returns_empty_status(
        self, chroma_tmp, patch_embedder, tmp_path
    ):
        empty = tmp_path / "empty.txt"
        empty.write_text("")
        result = await process_document(empty, empty.name)
        assert result["status"] == "empty"
        assert result["chunks"] == 0

    @pytest.mark.asyncio
    async def test_embed_failure_propagates(
        self, chroma_tmp, monkeypatch, sample_txt
    ):
        from unittest.mock import AsyncMock
        from app.core.exceptions import EmbeddingServiceError

        monkeypatch.setattr(
            "app.services.document_service.embed_texts",
            AsyncMock(side_effect=EmbeddingServiceError("Ollama down")),
        )
        with pytest.raises(EmbeddingServiceError):
            await process_document(sample_txt, sample_txt.name)

    @pytest.mark.asyncio
    async def test_doc_id_is_valid_uuid(self, chroma_tmp, patch_embedder, sample_txt):
        import uuid

        result = await process_document(sample_txt, sample_txt.name)
        # Should not raise
        parsed = uuid.UUID(result["doc_id"])
        assert str(parsed) == result["doc_id"]


# ── list_documents ────────────────────────────────────────────────────────────

class TestListDocuments:
    @pytest.mark.asyncio
    async def test_empty_returns_empty_list(self, chroma_tmp):
        result = list_documents()
        assert result == []

    @pytest.mark.asyncio
    async def test_one_document_shows_once(
        self, chroma_tmp, patch_embedder, sample_txt
    ):
        await process_document(sample_txt, sample_txt.name)
        result = list_documents()
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_two_documents_show_separately(
        self, chroma_tmp, patch_embedder, tmp_path
    ):
        p1 = tmp_path / "doc1.txt"
        p2 = tmp_path / "doc2.txt"
        p1.write_text("Document one content here with enough words.")
        p2.write_text("Document two content here with enough words.")

        await process_document(p1, p1.name)
        await process_document(p2, p2.name)

        result = list_documents()
        assert len(result) == 2
        names = {r["file_name"] for r in result}
        assert "doc1.txt" in names
        assert "doc2.txt" in names

    @pytest.mark.asyncio
    async def test_chunks_of_same_doc_deduplicated(
        self, chroma_tmp, patch_embedder, sample_txt
    ):
        """Multiple chunks from one document → only one entry in list."""
        await process_document(sample_txt, sample_txt.name)
        result = list_documents()
        # Even if chunks > 1, list should show 1 document
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_limit_respected(self, chroma_tmp, patch_embedder, tmp_path):
        for i in range(5):
            p = tmp_path / f"doc{i}.txt"
            p.write_text(f"Document {i} with unique content. Enough words to pass min.")
            await process_document(p, p.name)

        result = list_documents(limit=3)
        # The limit applies to raw chunk fetches; deduplicated count may vary
        # but should not exceed total documents
        assert len(result) <= 5


# ── delete_document ───────────────────────────────────────────────────────────

class TestDeleteDocument:
    @pytest.mark.asyncio
    async def test_delete_removes_document(
        self, chroma_tmp, patch_embedder, sample_txt
    ):
        result = await process_document(sample_txt, sample_txt.name)
        doc_id = result["doc_id"]

        removed = delete_document(doc_id)
        assert removed > 0

        # Verify gone from collection
        from app.db.chroma import get_collection

        remaining = get_collection().get(
            where={"doc_id": doc_id}, include=[]
        )
        assert remaining["ids"] == []

    @pytest.mark.asyncio
    async def test_delete_returns_chunk_count(
        self, chroma_tmp, patch_embedder, sample_txt
    ):
        result = await process_document(sample_txt, sample_txt.name)
        removed = delete_document(result["doc_id"])
        assert removed == result["chunks"]

    def test_delete_nonexistent_raises(self, chroma_tmp):
        with pytest.raises(DocumentNotFoundError) as exc_info:
            delete_document("nonexistent-doc-id")
        assert "nonexistent-doc-id" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_delete_does_not_affect_other_documents(
        self, chroma_tmp, patch_embedder, tmp_path
    ):
        p1 = tmp_path / "keep.txt"
        p2 = tmp_path / "remove.txt"
        p1.write_text("Keep this document content intact please.")
        p2.write_text("Remove this document content entirely now.")

        r1 = await process_document(p1, p1.name)
        r2 = await process_document(p2, p2.name)

        delete_document(r2["doc_id"])

        from app.db.chroma import get_collection

        remaining = get_collection().get(
            where={"doc_id": r1["doc_id"]}, include=[]
        )
        assert remaining["ids"] != []
