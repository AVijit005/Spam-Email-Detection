from __future__ import annotations

import subprocess
import sys
import threading

from fastapi import APIRouter, Depends, HTTPException

from app.config import settings
from app.core.auth import require_auth
from app.schemas.retrain import RetrainResponse
from app.storage.feedback import feedback_backend_name

router = APIRouter()

model_metadata: dict = {}
RETRAIN_LOCK = threading.Lock()


@router.post("/retrain", response_model=RetrainResponse, dependencies=[require_auth])
def retrain_model() -> RetrainResponse:
    if not RETRAIN_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Retraining is already in progress.")
    try:
        try:
            result = subprocess.run(
                [sys.executable, str(settings.train_script_path)],
                cwd=str(settings.train_script_path.parent.parent),
                capture_output=True, text=True, timeout=settings.retrain_timeout_seconds, check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise HTTPException(status_code=500, detail="Retraining timed out.") from error
        if result.returncode != 0:
            output_lines = [line for line in (result.stderr or "").splitlines() if line.strip()]
            output_lines.extend(line for line in (result.stdout or "").splitlines() if line.strip())
            detail = "\n".join(output_lines[-12:]) if output_lines else "Retraining failed."
            raise HTTPException(status_code=500, detail=detail)
        from app.main import load_resources
        load_resources()
        import app.api.v1.health as health_mod
        metadata = health_mod.model_metadata if health_mod.model_metadata else model_metadata
        training_info = metadata.get("feedback_training", {})
        selected_metrics = metadata.get("selected_metrics", {})
        spam_f1 = selected_metrics.get("spam_f1")
        model_version = str(metadata.get("model_name", "unknown"))
        trained_at = metadata.get("trained_at_utc")
        dataset_rows = int(metadata.get("dataset_rows", 0))
        return RetrainResponse(
            status="ok",
            model_version=model_version,
            feedback_backend=feedback_backend_name(settings.feedback_log_path),
            trained_at_utc=trained_at,
            dataset_rows=dataset_rows,
            feedback_rows_used=training_info.get("feedback_rows_used", 0),
            feedback_last_consumed_utc=training_info.get("last_feedback_at_utc"),
            spam_f1=float(spam_f1) if spam_f1 else None,
        )
    finally:
        RETRAIN_LOCK.release()
