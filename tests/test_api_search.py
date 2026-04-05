"""
API-level tests for the /search endpoint.

Uses the async HTTPX client fixture (api_client from conftest).
"""

from __future__ import annotations

import pytest


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
async def seeded_api(api_client, tmp_path):
    """Upload a couple of documents so search has data to work with."""
    docs = {
        "finance.txt": (
            "Revenue increased by fifteen percent in Q3 2024. "
            "Operating costs were reduced through automation initiatives. "
            "The board approved a new budget for fiscal year 2025."
        ),
        "tech.txt": (
            "Machine learning models improved prediction accuracy significantly. "
            "Neural networks enable complex pattern recognition tasks. "
            "Deep learning frameworks continue to evolve rapidly."
        ),
    }
    for name, content in docs.items():
        p = tmp_path / name
        p.write_text(content)
        with open(p, "rb") as fh:
            await api_client.post(
                "/documents/upload",
                files={"file": (name, fh, "text/plain")},
            )
    return api_client


# ── Basic search ─────────────────────────────────────────────────────────────

class TestSearchEndpointBasic:
    @pytest.mark.asyncio
    async def test_search_returns_200(self, seeded_api):
        resp = await seeded_api.get("/search/", params={"query": "revenue"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_search_response_schema(self, seeded_api):
        resp = await seeded_api.get("/search/", params={"query": "revenue"})
        body = resp.json()
        assert "query" in body
        assert "mode" in body
        assert "exact" in body
        assert "regex" in body
        assert "semantic" in body
        assert "merged" in body
        assert "total" in body
        assert isinstance(body["exact"], list)
        assert isinstance(body["regex"], list)
        assert isinstance(body["semantic"], list)
        assert isinstance(body["merged"], list)
        assert isinstance(body["total"], int)

    @pytest.mark.asyncio
    async def test_search_query_echoed_in_response(self, seeded_api):
        resp = await seeded_api.get("/search/", params={"query": "machine learning"})
        assert resp.json()["query"] == "machine learning"

    @pytest.mark.asyncio
    async def test_search_result_item_schema(self, seeded_api):
        resp = await seeded_api.get("/search/", params={"query": "Revenue"})
        body = resp.json()
        # At least one layer should have results
        all_items = body["exact"] + body["regex"] + body["semantic"] + body["merged"]
        if all_items:
            item = all_items[0]
            assert "id" in item
            assert "text" in item
            assert "metadata" in item
            assert "score" in item
            assert "source" in item
            assert isinstance(item["score"], float)

    @pytest.mark.asyncio
    async def test_search_empty_collection(self, api_client):
        resp = await api_client.get("/search/", params={"query": "anything"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0


# ── Query validation ──────────────────────────────────────────────────────────

class TestSearchQueryValidation:
    @pytest.mark.asyncio
    async def test_missing_query_returns_422(self, api_client):
        resp = await api_client.get("/search/")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_empty_query_returns_422(self, api_client):
        resp = await api_client.get("/search/", params={"query": ""})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_query_too_long_returns_422(self, api_client):
        long_query = "x" * 501
        resp = await api_client.get("/search/", params={"query": long_query})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_mode_returns_422(self, api_client):
        resp = await api_client.get(
            "/search/", params={"query": "test", "mode": "invalid_mode"}
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_top_k_zero_returns_422(self, api_client):
        resp = await api_client.get(
            "/search/", params={"query": "test", "top_k": 0}
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_top_k_above_100_returns_422(self, api_client):
        resp = await api_client.get(
            "/search/", params={"query": "test", "top_k": 101}
        )
        assert resp.status_code == 422


# ── Search modes ──────────────────────────────────────────────────────────────

class TestSearchModes:
    @pytest.mark.asyncio
    async def test_mode_all_runs_all_layers(self, seeded_api):
        resp = await seeded_api.get(
            "/search/", params={"query": "Revenue", "mode": "all"}
        )
        body = resp.json()
        assert body["mode"] == "all"
        # With real data, at least regex should find it
        total_results = len(body["exact"]) + len(body["regex"]) + len(body["semantic"])
        assert total_results >= 1

    @pytest.mark.asyncio
    async def test_mode_exact(self, seeded_api):
        resp = await seeded_api.get(
            "/search/", params={"query": "Revenue increased", "mode": "exact"}
        )
        body = resp.json()
        assert body["mode"] == "exact"
        assert body["regex"] == []
        assert body["semantic"] == []

    @pytest.mark.asyncio
    async def test_mode_regex(self, seeded_api):
        resp = await seeded_api.get(
            "/search/", params={"query": "Revenue", "mode": "regex"}
        )
        body = resp.json()
        assert body["mode"] == "regex"
        assert body["exact"] == []
        assert body["semantic"] == []
        assert len(body["regex"]) >= 1

    @pytest.mark.asyncio
    async def test_mode_semantic(self, seeded_api):
        resp = await seeded_api.get(
            "/search/", params={"query": "financial results", "mode": "semantic"}
        )
        body = resp.json()
        assert body["mode"] == "semantic"
        assert body["exact"] == []
        assert body["regex"] == []
        assert len(body["semantic"]) >= 1

    @pytest.mark.asyncio
    async def test_top_k_parameter_respected(self, seeded_api):
        resp = await seeded_api.get(
            "/search/", params={"query": "percent", "mode": "regex", "top_k": 1}
        )
        assert len(resp.json()["regex"]) <= 1

    @pytest.mark.asyncio
    async def test_merged_has_no_duplicate_ids(self, seeded_api):
        resp = await seeded_api.get(
            "/search/", params={"query": "Revenue", "mode": "all"}
        )
        merged = resp.json()["merged"]
        ids = [item["id"] for item in merged]
        assert len(ids) == len(set(ids)), "Duplicate IDs found in merged results"

    @pytest.mark.asyncio
    async def test_total_matches_merged_count(self, seeded_api):
        resp = await seeded_api.get("/search/", params={"query": "machine learning"})
        body = resp.json()
        assert body["total"] == len(body["merged"])


# ── Health endpoint ───────────────────────────────────────────────────────────

class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, api_client):
        resp = await api_client.get("/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_response_has_status(self, api_client):
        body = resp = await api_client.get("/health")
        body = resp.json()
        assert "status" in body
        assert "chroma_reachable" in body
        assert "ollama_reachable" in body
