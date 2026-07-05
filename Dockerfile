# Spam Email Detection — HF Spaces Docker Image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash user

WORKDIR /app

COPY --chown=user:user requirements-hf.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-hf.txt

COPY --chown=user:user . /app

RUN mkdir -p /app/data /app/model/hf_model /app/model/checkpoints && \
    chown -R user:user /app/data /app/model

USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Pre-download DeBERTa model into the image (as user, so cache is in /home/user)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Avijit070/spam-email-deberta-v3', local_dir='/app/model/hf_model')"

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
    CMD python -c "import os; from urllib.request import urlopen; urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"7860\")}/v1/health')" || exit 1

CMD ["sh", "-c", "exec gunicorn app.main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-7860} --timeout 120"]
