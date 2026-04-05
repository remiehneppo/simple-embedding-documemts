"""
Three-layer search: exact substring → regex → semantic (vector).

Results from all layers are merged and deduplicated, with exact matches
ranked first, then regex, then semantic — each group sorted by score.
"""

import re

from app.core.config import settings
from app.core.logging import get_logger
from app.db.chroma import get_collection
from app.services.embedder import embed_texts

log = get_logger(__name__)


# ── Public API ───────────────────────────────────────────────────────────────

async def search(
    query: str,
    top_k: int | None = None,
    mode: str = "all",
) -> dict:
    """
    Run document search and return results grouped by strategy.

    mode: "all" | "exact" | "regex" | "semantic"
    """
    k = top_k or settings.search_top_k
    log.info("search.start", query=query[:80], top_k=k, mode=mode)

    exact: list[dict] = []
    regex: list[dict] = []
    semantic: list[dict] = []

    if mode in ("all", "exact"):
        exact = _exact_search(query, k)

    if mode in ("all", "regex"):
        regex = _regex_search(query, k)

    if mode in ("all", "semantic"):
        semantic = await _semantic_search(query, k)

    merged = _merge(exact, regex, semantic)

    log.info(
        "search.done",
        query=query[:80],
        exact=len(exact),
        regex=len(regex),
        semantic=len(semantic),
        merged=len(merged),
    )

    return {
        "exact": exact,
        "regex": regex,
        "semantic": semantic,
        "merged": merged,
    }


# ── Layer implementations ────────────────────────────────────────────────────

def _exact_search(query: str, top_k: int) -> list[dict]:
    """Substring match via ChromaDB where_document filter."""
    collection = get_collection()
    try:
        result = collection.get(
            where_document={"$contains": query},
            include=["documents", "metadatas"],
            limit=top_k,
        )
        items = []
        for id_, text, meta in zip(
            result["ids"], result["documents"], result["metadatas"]
        ):
            items.append(
                {"id": id_, "text": text, "metadata": meta, "score": 1.0, "source": "exact"}
            )
        log.debug("search.exact_hits", count=len(items))
        return items
    except Exception as exc:
        log.error("search.exact_failed", error=str(exc), exc_info=True)
        return []


def _regex_search(query: str, top_k: int) -> list[dict]:
    """
    Regex match against a local pool of chunks.

    We fetch at most *search_regex_pool_limit* stored chunks and filter them
    locally. This is intentionally limited to keep the operation fast on large
    collections.
    """
    try:
        pattern = re.compile(query, re.IGNORECASE)
    except re.error as exc:
        log.warning("search.invalid_regex", pattern=query, error=str(exc))
        return []

    collection = get_collection()
    try:
        pool = collection.get(
            limit=settings.search_regex_pool_limit,
            include=["documents", "metadatas"],
        )
    except Exception as exc:
        log.error("search.regex_pool_failed", error=str(exc), exc_info=True)
        return []

    matches: list[dict] = []
    for id_, text, meta in zip(pool["ids"], pool["documents"], pool["metadatas"]):
        if pattern.search(text):
            matches.append(
                {"id": id_, "text": text, "metadata": meta, "score": 1.0, "source": "regex"}
            )
            if len(matches) >= top_k:
                break

    log.debug("search.regex_hits", count=len(matches))
    return matches


async def _semantic_search(query: str, top_k: int) -> list[dict]:
    """Vector similarity search via Ollama embeddings + ChromaDB cosine distance."""
    try:
        query_vecs = await embed_texts([query])
        query_vec = query_vecs[0]
    except Exception as exc:
        log.error("search.semantic_embed_failed", error=str(exc), exc_info=True)
        return []

    collection = get_collection()
    try:
        # n_results cannot exceed the number of items in the collection
        count = collection.count()
        n = min(top_k, count) if count > 0 else 0
        if n == 0:
            return []

        result = collection.query(
            query_embeddings=[query_vec],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        log.error("search.semantic_query_failed", error=str(exc), exc_info=True)
        return []

    items: list[dict] = []
    if result["ids"] and result["ids"][0]:
        for id_, text, meta, dist in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            # ChromaDB cosine distance ∈ [0, 2]; similarity = 1 - distance
            score = max(0.0, 1.0 - float(dist))
            items.append(
                {"id": id_, "text": text, "metadata": meta, "score": score, "source": "semantic"}
            )

    log.debug("search.semantic_hits", count=len(items))
    return items


# ── Merge & rank ─────────────────────────────────────────────────────────────

def _merge(
    exact: list[dict],
    regex: list[dict],
    semantic: list[dict],
) -> list[dict]:
    """
    Combine results from all layers without duplicates.

    Priority order: exact → regex → semantic.
    Within each group items keep their original order / score.
    """
    seen: set[str] = set()
    merged: list[dict] = []

    for item in exact + regex + semantic:
        if item["id"] not in seen:
            seen.add(item["id"])
            merged.append(item)

    # Sort by score descending as final ranking signal
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged
