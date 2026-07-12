# syntax=docker/dockerfile:1
FROM python:3.10-slim

# Build tools needed by some ML packages (sentence-transformers/torch deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Fast, reproducible dependency installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first — this layer only rebuilds when deps change,
# not on every code edit, so rebuilds during development stay fast.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Now copy the actual application code
COPY . .

ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["uv", "run", "streamlit", "run", "app.py"]