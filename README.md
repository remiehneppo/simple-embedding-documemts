# simple-embed

Semantic document search engine powered by **Ollama embeddings** + **ChromaDB** + **PaddleOCR**.

Upload PDF, DOCX, TXT, MD, or CSV files → they are automatically extracted (with OCR fallback for scanned PDFs), chunked, embedded, and indexed. Search across documents using exact substring, regex, or vector similarity.

---

## Features

- **Multi-format ingestion**: PDF (text layer + scanned OCR), DOCX, TXT, Markdown, CSV
- **OCR**: PaddleOCR with Vietnamese, Russian, English support; automatic fallback when no text layer is detected
- **Three-layer search**: exact substring · regular expression · semantic (vector similarity)
- **Embedding**: Ollama-backed embedding model (default: `embeddinggemma:300m-qat-q8_0`)
- **Vector store**: ChromaDB with SHA-256 deduplication
- **Structured logging**: structlog with source location and full tracebacks

---

## Architecture

```
Upload → Extract (PyMuPDF / PaddleOCR) → Clean → Chunk → Embed (Ollama) → Store (ChromaDB)
Search query → Embed → ChromaDB cosine similarity + exact/regex match → Merged results
```

---

## Requirements

| Dependency | Version |
|---|---|
| Python | 3.11+ |
| Ollama | any recent |
| PaddlePaddle | 3.2.0 (CPU) |

---

## Quick start — local (no Docker)

### 1. Clone

```bash
git clone https://github.com/remiehneppo/simple-embedding-documemts.git
cd simple-embedding-documemts
```

### 2. Install Ollama and pull the embedding model

```bash
# Install Ollama: https://ollama.com/download
ollama pull embeddinggemma:300m-qat-q8_0
```

### 3. Run

```bash
chmod +x dev.sh
./dev.sh
```

`dev.sh` will automatically:
- Create a Python virtualenv at `.venv/`
- Install all Python dependencies from `requirements.txt`
- Create `storage/` and `logs/` directories
- Start `ollama serve` if it is not already running
- Pull the configured embedding model if absent
- Start the FastAPI app at **http://localhost:8000** with hot-reload

**Options:**
```bash
./dev.sh              # start with auto-reload + DEBUG log (default)
./dev.sh --no-reload  # start without hot-reload
./dev.sh --setup      # only setup environment, do not start app
./dev.sh --help
```

### 4. Open the UI

- Web UI: http://localhost:8000
- API docs: http://localhost:8000/docs

---

## Configuration

Copy `.env.example` to `.env` and adjust as needed:

```bash
cp .env.example .env
```

Key variables:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `embeddinggemma:300m-qat-q8_0` | Embedding model name |
| `OCR_LANGS` | `vi,ru,en` | Languages for PaddleOCR (comma-separated) |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `PORT` | `8000` | API server port |
| `CHUNK_MAX_WORDS` | `50` | Max words per chunk |
| `CHUNK_OVERLAP_WORDS` | `10` | Overlap words between chunks |
| `SEARCH_TOP_K` | `10` | Max results per search layer |

---

## Running with Docker

### Prerequisites

- Docker with Compose plugin

### Build and run

```bash
# Step 1 — build the base image (downloads all OCR model weights, ~1-2 GB)
# Only needed once; re-run when requirements.txt or OCR_LANGS changes.
docker compose --profile base build base

# Step 2 — build the app image (fast, no internet needed)
docker compose build app

# Step 3 — start everything (Ollama + app)
docker compose up
```

The `base` image bakes all PaddleOCR model weights into the layer so the
`app` container can run **completely offline** after the base is built.

### Environment overrides

Pass overrides via a `.env` file or shell environment before running:

```bash
LOG_LEVEL=DEBUG PORT=9000 docker compose up
```

---

## API reference

### Upload a document

```bash
curl -X POST http://localhost:8000/documents/upload \
     -F "file=@/path/to/document.pdf"
```

Response:
```json
{
  "doc_id": "uuid",
  "file_name": "document.pdf",
  "chunks": 42,
  "pages": 5,
  "status": "processed"
}
```

### Search

```bash
curl "http://localhost:8000/search/?query=machine+learning&top_k=5&mode=all"
```

`mode`: `all` (default) · `exact` · `regex` · `semantic`

### List documents

```bash
curl http://localhost:8000/documents/
```

### Delete a document

```bash
curl -X DELETE http://localhost:8000/documents/{doc_id}
```

### Health check

```bash
curl http://localhost:8000/health
```

---

## Supported file types

| Extension | Extraction method |
|---|---|
| `.pdf` | PyMuPDF (text layer) → PaddleOCR fallback for scanned pages |
| `.docx` | python-docx |
| `.txt` / `.md` | Plain text |
| `.csv` | Plain text |

---

## Project structure

```
app/
  api/routes/       # FastAPI routers (documents, search)
  core/             # Config, logging, exceptions
  db/               # ChromaDB client
  models/           # Pydantic schemas
  services/
    extractor/      # PDF text, PDF OCR, DOCX, plaintext extractors
    chunker.py      # Word-overlap chunking
    cleaner.py      # Text normalisation
    embedder.py     # Ollama embedding client with retry
    document_service.py  # Full ingestion pipeline
    search_service.py    # Three-layer search
cli/                # Click CLI client
docker/             # Dockerfile.base, Dockerfile.app, docker-compose.yml
scripts/
  preload_models.py # Downloads all OCR weights at Docker build time
frontend/           # Static HTML/JS UI
tests/              # pytest test suite
dev.sh              # Local development runner
```

---

## Development

### Run tests

```bash
source .venv/bin/activate
pytest
```

### Logs

- Console: structured, coloured in TTY; JSON in non-TTY
- File: `logs/app.log` (rotating, 10 MB × 5 backups)
