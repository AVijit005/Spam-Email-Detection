from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import uvicorn

from app import MODEL_PATH, TRAIN_MODEL_PATH, VECTORIZER_PATH, app
from runtime_config import load_runtime_config


BASE_DIR = Path(__file__).resolve().parent


def ensure_model_artifacts() -> None:
    config = load_runtime_config()
    should_train = config.train_on_start or (
        config.bootstrap_model_if_missing and not (MODEL_PATH.exists() and VECTORIZER_PATH.exists())
    )

    if not should_train:
        return

    result = subprocess.run(
        [sys.executable, str(TRAIN_MODEL_PATH)],
        cwd=str(BASE_DIR.parent),
        check=False,
        timeout=config.retrain_timeout_seconds,
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    config = load_runtime_config()
    ensure_model_artifacts()
    uvicorn.run(
        app,
        host=config.api_host,
        port=config.api_port,
        log_level=config.log_level,
    )


if __name__ == "__main__":
    main()
