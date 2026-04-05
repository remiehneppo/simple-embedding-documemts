from typing import Any

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    doc_id: str
    file_name: str
    chunks: int
    pages: int
    status: str  # "processed" | "duplicate"


class DocumentMeta(BaseModel):
    doc_id: str
    file_name: str
    file_path: str
    file_type: str
    file_hash: str
    page_count: int
    chunk_count: int
    upload_ts: str


class SearchResultItem(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any]
    score: float
    source: str  # "exact" | "regex" | "semantic"


class SearchResponse(BaseModel):
    query: str
    mode: str
    exact: list[SearchResultItem] = Field(default_factory=list)
    regex: list[SearchResultItem] = Field(default_factory=list)
    semantic: list[SearchResultItem] = Field(default_factory=list)
    merged: list[SearchResultItem] = Field(default_factory=list)
    total: int = 0


class DeleteResponse(BaseModel):
    deleted: bool
    doc_id: str
    chunks_removed: int


class HealthResponse(BaseModel):
    status: str
    ollama_reachable: bool
    chroma_reachable: bool
    version: str = "1.0.0"
