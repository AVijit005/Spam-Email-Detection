# =============================================================================
# Stage 1 — Build all wheels (fast, runs once)
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

# CPU-only torch first
RUN pip wheel --wheel-dir /wheels \
    torch --index-url https://download.pytorch.org/whl/cpu

# All runtime deps
RUN pip wheel --wheel-dir /wheels --find-links /wheels \
    fastapi "uvicorn>=0.27.0" gunicorn \
    "scikit-learn>=1.4.0" "pandas>=2.1.0" "numpy>=1.26.0" "scipy>=1.12.0" \
    joblib nltk psutil tqdm httpx PyMySQL \
    pydantic-settings slowapi safetensors \
    huggingface-hub transformers xgboost

COPY scripts/dedup_wheels.py /tmp/dedup_wheels.py
RUN python3 /tmp/dedup_wheels.py

# =============================================================================
# Stage 2 — Install from pre-built wheels only (instant, no network)
# =============================================================================
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-deps /wheels/*.whl && rm -rf /wheels

COPY . /app
RUN mkdir -p /app/data /app/model/hf_model /app/model/checkpoints && \
    chown -R appuser:appuser /app/data /app/model

USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
    CMD python -c "import os; from urllib.request import urlopen; urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/v1/health')" || exit 1

CMD ["sh", "-c", "exec gunicorn app.main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-8000} --timeout 120"]
