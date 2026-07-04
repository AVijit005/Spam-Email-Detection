# =============================================================================
# Stage 1 — Build dependencies (wheels, no runtime bloat)
# =============================================================================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .

# CPU-only torch from PyTorch index (~200MB vs ~2GB CUDA)
RUN pip wheel --wheel-dir /wheels \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip wheel --wheel-dir /wheels --find-links /wheels \
    -r requirements.txt

# =============================================================================
# Stage 2 — Runtime (minimal, no build tools)
# =============================================================================
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd --create-home --shell /bin/bash appuser

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --find-links /wheels -r requirements.txt \
    && rm -rf /wheels

COPY . /app
RUN mkdir -p /app/data /app/model/hf_model /app/model/checkpoints && \
    chown -R appuser:appuser /app/data /app/model

USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import os; from urllib.request import urlopen; urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/v1/health')" || exit 1

CMD ["sh", "-c", "exec gunicorn app.main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-8000} --timeout 120"]
