"""
ChromaDB persistent client — thin wrapper with lazy initialisation.

A single PersistentClient and collection are reused across the app lifetime.
All callers use *get_collection()* which initialises on first call.
"""

from typing import Optional

import chromadb
import chromadb.api
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

_client: Optional[chromadb.api.ClientAPI] = None
_collection: Optional[chromadb.Collection] = None


def get_chroma_client() -> chromadb.api.ClientAPI:
    global _client
    if _client is None:
        db_path = settings.chroma_path
        db_path.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(db_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        log.info("chroma.client_ready", path=str(db_path))
    return _client


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = get_chroma_client()
        _collection = client.get_or_create_collection(
            name=settings.chroma_collection,
            # cosine distance → similarity = 1 - distance
            metadata={"hnsw:space": "cosine"},
        )
        log.info(
            "chroma.collection_ready",
            name=settings.chroma_collection,
            count=_collection.count(),
        )
    return _collection


def reset_collection_cache() -> None:
    """Force re-initialisation on next call (useful for testing)."""
    global _client, _collection
    _client = None
    _collection = None
