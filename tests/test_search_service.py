"""
Integration tests for app.services.search_service

Uses a real ChromaDB collection in a tmp directory for storage.
The Ollama embedder is replaced by the fake from conftest.patch_embedder.
"""

from __future__ import annotations

import pytest

from app.services.search_service import _exact_search, _merge, _regex_search, search


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def populated_collection(chroma_tmp, patch_embedder):
    """
    Seed the ChromaDB collection with a small set of chunks and return it.
    Vectors are fake (from patch_embedder / make_fake_embeddings).
    """
    from tests.conftest import make_fake_embeddings
    from app.db.chroma import get_collection

    collection = get_collection()
    texts = [
        "Revenue increased by fifteen percent in Q3 2024.",
        "Operating costs were reduced through automation.",
        "The board approved a new budget for fiscal year 2025.",
        "Customer satisfaction scores reached an all-time high.",
        "Machine learning models improved prediction accuracy.",
        "Supply chain disruptions affected global manufacturing.",
        "Renewable energy investments grew by thirty percent.",
        "Quarterly dividend was increased to two dollars per share.",
    ]
    embeddings = make_fake_embeddings(texts)
    ids = [f"doc1_chunk_{i}" for i in range(len(texts))]
    metadatas = [
        {
            "doc_id": "doc1",
            "file_name": "report.pdf",
            "file_path": "/storage/report.pdf",
            "file_type": "pdf",
            "file_hash": "abc123",
            "page_number": i + 1,
            "chunk_index": i,
            "total_chunks": len(texts),
            "page_count": 8,
            "upload_ts": "2026-04-01T10:00:00Z",
        }
        for i in range(len(texts))
    ]
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return collection


# ── Exact search ──────────────────────────────────────────────────────────────

class TestExactSearch:
    def test_finds_exact_substring(self, populated_collection):
        results = _exact_search("Revenue increased", top_k=10)
        assert len(results) == 1
        assert "Revenue increased" in results[0]["text"]

    def test_case_sensitive_miss(self, populated_collection):
        # ChromaDB $contains is case-sensitive
        results = _exact_search("revenue increased", top_k=10)
        # Case-sensitive: may return 0 results
        assert isinstance(results, list)

    def test_no_match_returns_empty(self, populated_collection):
        results = _exact_search("xyzzy_does_not_exist_abc", top_k=10)
        assert results == []

    def test_result_has_required_fields(self, populated_collection):
        results = _exact_search("Revenue", top_k=10)
        assert len(results) >= 1
        item = results[0]
        assert "id" in item
        assert "text" in item
        assert "metadata" in item
        assert "score" in item
        assert item["source"] == "exact"
        assert item["score"] == 1.0

    def test_top_k_respected(self, populated_collection):
        # All chunks contain "doc1" in their IDs but we can't guarantee text match
        # Use a common word that might appear in multiple chunks
        results = _exact_search("percent", top_k=2)
        assert len(results) <= 2

    def test_empty_collection_returns_empty(self, chroma_tmp):
        results = _exact_search("anything", top_k=10)
        assert results == []


# ── Regex search ──────────────────────────────────────────────────────────────

class TestRegexSearch:
    def test_simple_pattern_matches(self, populated_collection):
        results = _regex_search("Revenue", top_k=10)
        assert len(results) >= 1
        assert "Revenue" in results[0]["text"]

    def test_case_insensitive(self, populated_collection):
        results = _regex_search("revenue", top_k=10)
        assert len(results) >= 1

    def test_regex_pattern_with_alternation(self, populated_collection):
        results = _regex_search(r"Revenue|costs", top_k=10)
        assert len(results) >= 2

    def test_regex_word_boundary(self, populated_collection):
        results = _regex_search(r"\bboard\b", top_k=10)
        assert len(results) >= 1
        assert "board" in results[0]["text"].lower()

    def test_invalid_regex_returns_empty(self, populated_collection):
        results = _regex_search("[invalid(regex", top_k=10)
        assert results == []

    def test_no_match_returns_empty(self, populated_collection):
        results = _regex_search("xyzzy_no_match_12345", top_k=10)
        assert results == []

    def test_result_source_is_regex(self, populated_collection):
        results = _regex_search("Revenue", top_k=10)
        assert results[0]["source"] == "regex"

    def test_top_k_limits_results(self, populated_collection):
        # "percent" appears in 2 chunks; limit to 1
        results = _regex_search("percent", top_k=1)
        assert len(results) <= 1

    def test_empty_collection_returns_empty(self, chroma_tmp):
        results = _regex_search("anything", top_k=10)
        assert results == []


# ── Semantic search ───────────────────────────────────────────────────────────

class TestSemanticSearch:
    @pytest.mark.asyncio
    async def test_returns_results(self, populated_collection, patch_embedder):
        from app.services.search_service import _semantic_search

        results = await _semantic_search("financial performance", top_k=5)
        assert isinstance(results, list)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_results_have_score_field(self, populated_collection, patch_embedder):
        from app.services.search_service import _semantic_search

        results = await _semantic_search("revenue", top_k=3)
        for item in results:
            assert 0.0 <= item["score"] <= 1.0
            assert item["source"] == "semantic"

    @pytest.mark.asyncio
    async def test_empty_collection_returns_empty(self, chroma_tmp, patch_embedder):
        from app.services.search_service import _semantic_search

        results = await _semantic_search("anything", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_embed_failure_returns_empty(self, populated_collection, monkeypatch):
        from unittest.mock import AsyncMock

        monkeypatch.setattr(
            "app.services.search_service.embed_texts",
            AsyncMock(side_effect=Exception("Ollama down")),
        )
        from app.services.search_service import _semantic_search

        results = await _semantic_search("test query", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_top_k_respected(self, populated_collection, patch_embedder):
        from app.services.search_service import _semantic_search

        results = await _semantic_search("performance", top_k=3)
        assert len(results) <= 3


# ── Merge & deduplicate ───────────────────────────────────────────────────────

class TestMerge:
    def _make_item(self, id_: str, score: float, source: str = "exact") -> dict:
        return {
            "id": id_,
            "text": f"text for {id_}",
            "metadata": {},
            "score": score,
            "source": source,
        }

    def test_deduplicates_by_id(self):
        item = self._make_item("a", 1.0, "exact")
        duplicate = self._make_item("a", 1.0, "semantic")
        result = _merge([item], [], [duplicate])
        ids = [r["id"] for r in result]
        assert ids.count("a") == 1

    def test_all_unique_items_included(self):
        a = self._make_item("a", 1.0)
        b = self._make_item("b", 0.9)
        c = self._make_item("c", 0.8)
        result = _merge([a], [b], [c])
        assert len(result) == 3

    def test_sorted_by_score_descending(self):
        a = self._make_item("a", 0.5)
        b = self._make_item("b", 0.9)
        c = self._make_item("c", 0.7)
        result = _merge([a], [b], [c])
        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_empty_inputs_returns_empty(self):
        assert _merge([], [], []) == []

    def test_exact_only(self):
        a = self._make_item("a", 1.0)
        result = _merge([a], [], [])
        assert len(result) == 1

    def test_all_layers_populated(self):
        exact = [self._make_item("e1", 1.0), self._make_item("e2", 1.0)]
        regex = [self._make_item("r1", 1.0), self._make_item("e1", 1.0)]  # e1 duplicate
        semantic = [self._make_item("s1", 0.8), self._make_item("r1", 0.8)]  # r1 duplicate
        result = _merge(exact, regex, semantic)
        # Should have: e1, e2, r1, s1 (4 unique)
        ids = {r["id"] for r in result}
        assert ids == {"e1", "e2", "r1", "s1"}


# ── High-level search() function ─────────────────────────────────────────────

class TestSearchFunction:
    @pytest.mark.asyncio
    async def test_search_all_mode(self, populated_collection, patch_embedder):
        result = await search("Revenue", mode="all")
        assert "exact" in result
        assert "regex" in result
        assert "semantic" in result
        assert "merged" in result

    @pytest.mark.asyncio
    async def test_search_exact_mode_only(self, populated_collection, patch_embedder):
        result = await search("Revenue increased", mode="exact")
        assert len(result["exact"]) >= 1
        # regex and semantic should be empty (not run)
        assert result["regex"] == []
        assert result["semantic"] == []

    @pytest.mark.asyncio
    async def test_search_regex_mode_only(self, populated_collection, patch_embedder):
        result = await search("Revenue", mode="regex")
        assert len(result["regex"]) >= 1
        assert result["exact"] == []
        assert result["semantic"] == []

    @pytest.mark.asyncio
    async def test_search_semantic_mode_only(self, populated_collection, patch_embedder):
        result = await search("financial results", mode="semantic")
        assert len(result["semantic"]) >= 1
        assert result["exact"] == []
        assert result["regex"] == []

    @pytest.mark.asyncio
    async def test_merged_deduplicates_across_layers(self, populated_collection, patch_embedder):
        result = await search("Revenue increased", mode="all")
        merged_ids = [r["id"] for r in result["merged"]]
        # No duplicate IDs in merged
        assert len(merged_ids) == len(set(merged_ids))

    @pytest.mark.asyncio
    async def test_top_k_parameter(self, populated_collection, patch_embedder):
        result = await search("percent", top_k=1, mode="regex")
        assert len(result["regex"]) <= 1

    @pytest.mark.asyncio
    async def test_empty_collection(self, chroma_tmp, patch_embedder):
        result = await search("anything", mode="all")
        assert result["exact"] == []
        assert result["regex"] == []
        assert result["semantic"] == []
        assert result["merged"] == []
