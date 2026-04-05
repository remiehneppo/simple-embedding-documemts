#!/bin/sh
set -e

MODEL="${OLLAMA_MODEL:-embeddinggemma:300m-qat-q8_0}"

# Start ollama server in background
ollama serve &
OLLAMA_PID=$!

# Wait until server is ready
echo "[entrypoint] Waiting for Ollama server to be ready..."
until wget -qO- http://localhost:11434/api/tags >/dev/null 2>&1; do
    sleep 1
done
echo "[entrypoint] Ollama server is ready."

# Pull model if not already present
if ollama list | grep -q "^${MODEL}"; then
    echo "[entrypoint] Model ${MODEL} already present, skipping pull."
else
    echo "[entrypoint] Pulling model ${MODEL}..."
    ollama pull "${MODEL}"
    echo "[entrypoint] Model ${MODEL} pulled successfully."
fi

# Hand off to ollama server process (keep container alive)
wait $OLLAMA_PID
