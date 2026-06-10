from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import uvicorn

from app.config import settings
from app.main import app


BASE_DIR = Path(__file__).resolve().parent.parent


def ensure_model_artifacts() -> None:
    should_train = settings.train_on_start or (
        settings.bootstrap_model_if_missing
        and not (settings.model_path.exists() and settings.vectorizer_path.exists())
    )

    if not should_train:
        return

    result = subprocess.run(
        [sys.executable, str(settings.train_script_path)],
        cwd=str(BASE_DIR),
        check=False,
        timeout=settings.retrain_timeout_seconds,
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    ensure_model_artifacts()
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
