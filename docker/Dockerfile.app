# ── App image ────────────────────────────────────────────────────────────────
# Built on top of simple-embed-base which already contains:
#   - all system packages and Python dependencies
#   - preloaded PaddleOCR model weights (vi, ru, en) at /root/.paddlex/
#
# Build base first:
#   docker build -f docker/Dockerfile.base -t simple-embed-base .
# Or via docker-compose:
#   docker compose --profile base build base
FROM simple-embed-base

# All PaddlePaddle runtime flags are inherited from the base image ENV.
# Confirm PYTHONPATH is set for the app package.
ENV PYTHONPATH=/app

# Copy application source
COPY app/      /app/app/
COPY frontend/ /app/frontend/
COPY cli/      /app/cli/

# Persistent data directories (overridden by named volumes in compose)
RUN mkdir -p /app/storage/documents /app/storage/chroma_db /app/logs

EXPOSE 8000

# Use shell form so ${LOG_LEVEL} env var is expanded at container start.
# Uvicorn log-level must be lowercase; pydantic-settings reads LOG_LEVEL for structlog.
CMD uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level "${LOG_LEVEL:-INFO}"
