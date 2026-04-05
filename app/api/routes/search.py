from fastapi import APIRouter, Query

from app.core.logging import get_logger
from app.models.schemas import SearchResponse, SearchResultItem
from app.services.search_service import search

router = APIRouter(prefix="/search", tags=["search"])
log = get_logger(__name__)

_VALID_MODES = {"all", "exact", "regex", "semantic"}


@router.get("/", response_model=SearchResponse)
async def search_documents(
    query: str = Query(..., min_length=1, max_length=500, description="Search query"),
    top_k: int = Query(default=10, ge=1, le=100, description="Max results per layer"),
    mode: str = Query(
        default="all",
        description="Search mode: all | exact | regex | semantic",
    ),
):
    """
    Search indexed documents.

    - **exact**: substring match (case-sensitive, fast)
    - **regex**: regular-expression match (case-insensitive)
    - **semantic**: vector similarity via embedding model
    - **all**: run all three layers and merge results
    """
    if mode not in _VALID_MODES:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=422,
            detail=f"Invalid mode '{mode}'. Choose from: {sorted(_VALID_MODES)}",
        )

    log.info("api.search", query=query[:80], top_k=top_k, mode=mode)
    results = await search(query, top_k=top_k, mode=mode)

    return SearchResponse(
        query=query,
        mode=mode,
        exact=[SearchResultItem(**i) for i in results["exact"]],
        regex=[SearchResultItem(**i) for i in results["regex"]],
        semantic=[SearchResultItem(**i) for i in results["semantic"]],
        merged=[SearchResultItem(**i) for i in results["merged"]],
        total=len(results["merged"]),
    )
