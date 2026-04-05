#!/usr/bin/env bash
# run.sh — Start the simple-embed application
#
# Usage:
#   ./run.sh                  # local mode (requires Ollama running on localhost)
#   ./run.sh --docker         # full Docker Compose stack
#   ./run.sh --docker --build # rebuild Docker images before starting
#   ./run.sh --help
#
# Environment variables (local mode):
#   HOST              API host          (default: 0.0.0.0)
#   PORT              API port          (default: 8000)
#   OLLAMA_URL        Ollama base URL   (default: http://localhost:11434)
#   OLLAMA_MODEL      Embedding model   (default: embeddinggemma:300m-qat-q8_0)
#   LOG_LEVEL         Logging level     (default: INFO)
#   RELOAD            Enable hot-reload (default: false)

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Defaults ──────────────────────────────────────────────────────────────────
MODE="local"
DOCKER_BUILD=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Argument parsing ──────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --docker) MODE="docker" ;;
        --build)  DOCKER_BUILD=true ;;
        --help|-h)
            sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) error "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ─────────────────────────────────────────────────────────────────────────────
# DOCKER mode
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$MODE" == "docker" ]]; then
    info "Starting simple-embed via Docker Compose..."

    if ! command -v docker &>/dev/null; then
        error "Docker is not installed or not in PATH."
        exit 1
    fi

    COMPOSE_FILE="$SCRIPT_DIR/docker/docker-compose.yml"

    if [[ "$DOCKER_BUILD" == true ]]; then
        info "Building base image (simple-embed-base)..."
        docker compose -f "$COMPOSE_FILE" --profile base build base
        info "Building app and ollama images..."
        docker compose -f "$COMPOSE_FILE" build
    fi

    info "Starting services (ollama + app)..."
    docker compose -f "$COMPOSE_FILE" up

    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# LOCAL mode
# ─────────────────────────────────────────────────────────────────────────────
info "Starting simple-embed in local mode..."

# ── Resolve virtualenv ────────────────────────────────────────────────────────
VENV_PYTHON=""
for candidate in \
    "$SCRIPT_DIR/.venv/bin/python" \
    "$SCRIPT_DIR/venv/bin/python" \
    "$(command -v python3 2>/dev/null || true)" \
    "$(command -v python 2>/dev/null || true)"
do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
        VENV_PYTHON="$candidate"
        break
    fi
done

if [[ -z "$VENV_PYTHON" ]]; then
    error "No Python interpreter found. Create a virtualenv first:"
    error "  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

success "Python: $VENV_PYTHON"

# ── Check required packages ───────────────────────────────────────────────────
if ! "$VENV_PYTHON" -c "import fastapi, uvicorn, chromadb, httpx" 2>/dev/null; then
    warn "Some dependencies are missing. Installing from requirements.txt..."
    "$VENV_PYTHON" -m pip install -q -r "$SCRIPT_DIR/requirements.txt"
fi

# ── Create required directories ───────────────────────────────────────────────
mkdir -p \
    "$SCRIPT_DIR/storage/documents" \
    "$SCRIPT_DIR/storage/chroma_db" \
    "$SCRIPT_DIR/logs"
success "Storage directories ready."

# ── Check Ollama ──────────────────────────────────────────────────────────────
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
info "Checking Ollama at $OLLAMA_URL ..."
if curl -sf "$OLLAMA_URL/api/tags" -o /dev/null 2>/dev/null; then
    success "Ollama is reachable."
else
    warn "Ollama is NOT reachable at $OLLAMA_URL."
    warn "Start Ollama with: ollama serve"
    warn "Then pull the model: ollama pull \${OLLAMA_MODEL:-embeddinggemma:300m-qat-q8_0}"
    warn "The app will start anyway, but embedding calls will fail until Ollama is up."
fi

# ── Launch uvicorn ────────────────────────────────────────────────────────────
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
LOG_LEVEL="${LOG_LEVEL:-info}"
RELOAD_FLAG=""
if [[ "${RELOAD:-false}" == "true" ]]; then
    RELOAD_FLAG="--reload"
    warn "Hot-reload enabled (--reload). Do not use in production."
fi

info "Starting uvicorn on http://${HOST}:${PORT} ..."
echo ""

cd "$SCRIPT_DIR"
exec "$VENV_PYTHON" -m uvicorn app.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --log-level "$(echo "$LOG_LEVEL" | tr '[:upper:]' '[:lower:]')" \
    $RELOAD_FLAG
