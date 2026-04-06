#!/bin/sh
set -e

MODEL="${OLLAMA_MODEL:-embeddinggemma:300m-qat-q8_0}"

# Start ollama server in background
ollama serve &
OLLAMA_PID=$!

# Wait until server is ready (wget not available in alpine/ollama; use ollama list)
echo "[entrypoint] Waiting for Ollama server to be ready..."
until ollama list >/dev/null 2>&1; do
    sleep 1
done
echo "[entrypoint] Ollama server is ready."

# Model must already be baked into the image at build time.
# No internet pull is attempted — fail fast if missing.
if ollama show "${MODEL}" >/dev/null 2>&1; then
    echo "[entrypoint] Model ${MODEL} is present."
else
    echo "[entrypoint] ERROR: model ${MODEL} not found in image." >&2
    echo "[entrypoint] Rebuild the image with: docker build --build-arg OLLAMA_MODEL=${MODEL} ..." >&2
    kill $OLLAMA_PID
    exit 1
fi

# Hand off to ollama server process (keep container alive)
wait $OLLAMA_PID
