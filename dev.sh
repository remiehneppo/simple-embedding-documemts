#!/usr/bin/env bash
# dev.sh — Local development runner (no Docker required)
#
# Usage:
#   ./dev.sh              # start app with auto-reload + DEBUG logging
#   ./dev.sh --setup      # only setup env/deps, do not start the app
#   ./dev.sh --no-reload  # start without hot-reload (production-like locally)
#   ./dev.sh --help
#
# Prerequisites:
#   - Python 3.11+ installed
#   - Ollama installed and the embedding model pulled
#       ollama pull embeddinggemma:300m-qat-q8_0
#
# Environment overrides (or set in .env):
#   OLLAMA_MODEL   embedding model name   (default from .env or config default)
#   OLLAMA_URL     Ollama base URL        (default: http://localhost:11434)
#   PORT           API port               (default: 8000)
#   LOG_LEVEL      DEBUG | INFO | WARNING (default: DEBUG in dev mode)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[ OK ]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERR ]${NC}  $*" >&2; }
header()  { echo -e "\n${BOLD}$*${NC}"; }

# ── Argument parsing ──────────────────────────────────────────────────────────
SETUP_ONLY=false
RELOAD=true

for arg in "$@"; do
    case "$arg" in
        --setup)     SETUP_ONLY=true ;;
        --no-reload) RELOAD=false ;;
        --help|-h)
            sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) error "Unknown argument: $arg"; exit 1 ;;
    esac
done

cd "$SCRIPT_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# 1. .env file
# ─────────────────────────────────────────────────────────────────────────────
header "── Environment ──────────────────────────────────────────────"

if [[ ! -f "$SCRIPT_DIR/.env" ]]; then
    if [[ -f "$SCRIPT_DIR/.env.example" ]]; then
        cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
        success "Created .env from .env.example — review it before starting."
    else
        warn ".env not found and no .env.example to copy from. Using defaults."
    fi
else
    success ".env found."
fi

# Load .env into shell so we can read OLLAMA_MODEL etc. below
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    set -o allexport
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.env" || true
    set +o allexport
fi

# Apply defaults for variables not set in .env
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-embeddinggemma:300m-qat-q8_0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
LOG_LEVEL="${LOG_LEVEL:-DEBUG}"

# ─────────────────────────────────────────────────────────────────────────────
# 2. Python virtualenv
# ─────────────────────────────────────────────────────────────────────────────
header "── Python environment ───────────────────────────────────────"

VENV_DIR="$SCRIPT_DIR/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating virtualenv at .venv ..."
    python3 -m venv "$VENV_DIR"
    success "Virtualenv created."
fi

PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

if [[ ! -x "$PYTHON" ]]; then
    error "Could not find Python in .venv/bin/python"
    exit 1
fi

PYTHON_VER="$("$PYTHON" --version 2>&1)"
success "Python: $PYTHON_VER"

# ─────────────────────────────────────────────────────────────────────────────
# 3. Install / sync dependencies
# ─────────────────────────────────────────────────────────────────────────────
header "── Dependencies ─────────────────────────────────────────────"

# Detect if a full install is needed (check a few key packages)
if ! "$PYTHON" -c "import fastapi, uvicorn, chromadb, structlog, httpx" 2>/dev/null; then
    info "Installing dependencies from requirements.txt ..."
    "$PIP" install --quiet --upgrade pip
    "$PIP" install --quiet -r "$SCRIPT_DIR/requirements.txt"
    success "Dependencies installed."
else
    success "All core dependencies already installed."
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4. Storage directories
# ─────────────────────────────────────────────────────────────────────────────
header "── Storage ──────────────────────────────────────────────────"

mkdir -p \
    "$SCRIPT_DIR/storage/documents" \
    "$SCRIPT_DIR/storage/chroma_db" \
    "$SCRIPT_DIR/logs"
success "storage/documents, storage/chroma_db, logs — ready."

# ─────────────────────────────────────────────────────────────────────────────
# 5. Ollama checks
# ─────────────────────────────────────────────────────────────────────────────
header "── Ollama ───────────────────────────────────────────────────"

if ! command -v ollama &>/dev/null; then
    error "ollama binary not found in PATH."
    error "Install from: https://ollama.com/download"
    exit 1
fi

OLLAMA_VERSION="$(ollama --version 2>&1 | head -1)"
success "Ollama binary: $OLLAMA_VERSION"

# Check if Ollama server is running
if curl -sf "$OLLAMA_URL/api/tags" -o /dev/null 2>/dev/null; then
    success "Ollama server is running at $OLLAMA_URL"
else
    warn "Ollama server is not running. Starting it in the background ..."
    ollama serve &>/dev/null &
    OLLAMA_PID=$!
    # Wait up to 10 seconds for it to become ready
    for i in $(seq 1 10); do
        sleep 1
        if curl -sf "$OLLAMA_URL/api/tags" -o /dev/null 2>/dev/null; then
            success "Ollama server started (pid $OLLAMA_PID)."
            break
        fi
        if [[ $i -eq 10 ]]; then
            error "Ollama did not become ready after 10 seconds."
            error "Run manually: ollama serve"
            exit 1
        fi
    done
fi

# Check if the embedding model is available; pull if missing
info "Checking model '${OLLAMA_MODEL}' ..."
if ollama list 2>/dev/null | grep -q "^${OLLAMA_MODEL}"; then
    success "Model '${OLLAMA_MODEL}' is available."
else
    warn "Model '${OLLAMA_MODEL}' not found locally. Pulling ..."
    ollama pull "${OLLAMA_MODEL}"
    success "Model '${OLLAMA_MODEL}' pulled."
fi

# ─────────────────────────────────────────────────────────────────────────────
# 6. Done — either exit (--setup) or launch uvicorn
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$SETUP_ONLY" == true ]]; then
    echo ""
    success "Setup complete. Run './dev.sh' to start the application."
    exit 0
fi

header "── Starting app ─────────────────────────────────────────────"

RELOAD_FLAGS=()
if [[ "$RELOAD" == true ]]; then
    RELOAD_FLAGS=(--reload --reload-dir "$SCRIPT_DIR/app")
    warn "Hot-reload enabled (watches app/). Press Ctrl+C to stop."
fi

echo ""
info "API  → http://${HOST}:${PORT}"
info "Docs → http://${HOST}:${PORT}/docs"
info "Logs → ${SCRIPT_DIR}/logs/app.log"
info "Log level: ${LOG_LEVEL}"
echo ""

# Export so pydantic-settings picks them up
export HOST PORT LOG_LEVEL OLLAMA_URL OLLAMA_MODEL

exec "$PYTHON" -m uvicorn app.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --log-level "$(echo "$LOG_LEVEL" | tr '[:upper:]' '[:lower:]')" \
    "${RELOAD_FLAGS[@]+"${RELOAD_FLAGS[@]}"}"
