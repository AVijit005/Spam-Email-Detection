FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN useradd --create-home --shell /bin/bash appuser

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first (~200MB vs ~2GB CUDA)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Runtime deps only (no datasets/lightgbm/optuna/accelerate — Kaggle-only)
RUN pip install --no-cache-dir \
    fastapi>=0.110.0 \
    "uvicorn[standard]>=0.27.0" \
    gunicorn>=21.0.0 \
    scikit-learn>=1.4.0 \
    pandas>=2.1.0 \
    numpy>=1.26.0 \
    scipy>=1.12.0 \
    joblib>=1.3.0 \
    nltk>=3.8.0 \
    psutil>=5.9.0 \
    tqdm>=4.66.0 \
    httpx>=0.27.0 \
    PyMySQL>=1.1.0 \
    pydantic-settings>=2.1.0 \
    slowapi>=0.1.9 \
    safetensors>=0.4.0 \
    "huggingface-hub>=0.23.0" \
    transformers>=4.38.0 \
    xgboost>=2.0.0

COPY . /app
RUN mkdir -p /app/data /app/model/hf_model /app/model/checkpoints && \
    chown -R appuser:appuser /app/data /app/model

USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
    CMD python -c "import os; from urllib.request import urlopen; urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/v1/health')" || exit 1

CMD ["sh", "-c", "exec gunicorn app.main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-8000} --timeout 120"]
