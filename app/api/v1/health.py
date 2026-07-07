from __future__ import annotations

from fastapi import APIRouter

from app.config import settings
from app.core.detector import _is_ensemble_model
from app.schemas.health import HealthResponse
from app.storage.feedback import FeedbackStoreError, feedback_backend_name, feedback_summary

router = APIRouter()

model = None
vectorizer = None
user_whitelist_domains: set[str] = set()
trusted_domain_catalog: set[str] = set()
model_metadata: dict = {}


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        summary = feedback_summary(settings.feedback_log_path)
        backend_name = feedback_backend_name(settings.feedback_log_path)
        feedback_error = None
        status = "ok"
    except FeedbackStoreError as error:
        summary = {"feedback_count": 0, "verdict_counts": {"correct": 0, "false_positive": 0, "false_negative": 0}}
        backend_name = "unavailable"
        feedback_error = str(error)
        status = "degraded"

    training_info = model_metadata.get("feedback_training") or {} if model_metadata else {}

    return HealthResponse(
        status=status,
        model_loaded=model is not None,
        vectorizer_loaded=vectorizer is not None,
        ensemble_active=_is_ensemble_model(model) and getattr(model, "has_transformer", False) if model is not None else False,
        feedback_backend=backend_name,
        user_whitelist_count=len(user_whitelist_domains),
        trusted_domain_catalog_count=len(trusted_domain_catalog),
        feedback_count=summary["feedback_count"],
        feedback_rows_used=training_info.get("feedback_rows_used", 0),
        feedback_last_consumed_utc=training_info.get("last_feedback_at_utc"),
        feedback_store_error=feedback_error,
        trained_at_utc=model_metadata.get("trained_at_utc") if model_metadata else None,
        model_version=str(model_metadata.get("model_name", "untrained")) if model_metadata else "untrained",
        spam_threshold=settings.spam_threshold,
    )
