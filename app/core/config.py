from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_name: str = "simple-embed"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # Paths (resolved relative to project root at runtime)
    storage_path: Path = Path("storage")
    documents_path: Path = Path("storage/documents")
    chroma_path: Path = Path("storage/chroma_db")
    log_path: Path = Path("logs")

    # Ollama
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "embeddinggemma:300m-qat-q8_0"
    embed_batch_size: int = 32
    ollama_timeout: float = 60.0
    ollama_max_retries: int = 3

    # Chunking
    chunk_max_words: int = 50
    chunk_overlap_words: int = 10
    chunk_min_words: int = 5  # Discard chunks shorter than this

    # ChromaDB
    chroma_collection: str = "documents"

    # Search
    search_top_k: int = 10
    search_regex_pool_limit: int = 5000  # Max docs scanned for regex search

    # OCR
    ocr_langs: str = "en,vi,ru"


settings = Settings()
