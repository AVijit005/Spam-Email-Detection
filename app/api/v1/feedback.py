from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.core.auth import require_auth
from app.schemas.feedback import FeedbackRequest, FeedbackResponse, FeedbackSummaryResponse
from app.storage.feedback import FeedbackStoreError, append_feedback_entry, feedback_summary
from app.utils.pii import redact_email_body, redact_subject

router = APIRouter()

model_metadata: dict = {}


def _normalize_user_label(label: str) -> str:
    normalized = (label or "").strip().lower()
    if normalized in {"spam", "junk"}:
        return "Spam"
    if normalized in {"not spam", "ham", "safe", "legitimate", "whitelisted"}:
        return "Not Spam"
    raise ValueError("user_label must be 'Spam' or 'Not Spam'")


def _feedback_verdict(predicted_label: str, user_label: str) -> str:
    normalized_predicted = _normalize_user_label(predicted_label)
    normalized_user = _normalize_user_label(user_label)
    if normalized_predicted == normalized_user:
        return "correct"
    if normalized_predicted == "Spam" and normalized_user == "Not Spam":
        return "false_positive"
    if normalized_predicted == "Not Spam" and normalized_user == "Spam":
        return "false_negative"
    return "correct"


@router.get("/feedback/summary", response_model=FeedbackSummaryResponse)
def feedback_summary_endpoint() -> FeedbackSummaryResponse:
    try:
        summary = feedback_summary(settings.feedback_log_path)
    except FeedbackStoreError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return FeedbackSummaryResponse(**summary)


@router.post("/feedback", response_model=FeedbackResponse, dependencies=[require_auth])
def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    try:
        normalized_user_label = _normalize_user_label(request.user_label)
        verdict = _feedback_verdict(request.predicted_label, normalized_user_label)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    stored_at_utc = datetime.now(timezone.utc).isoformat()
    feedback_id = f"fb_{request.prediction_id[:16]}_{stored_at_utc.replace(':', '').replace('-', '')[:14]}"
    redacted_subject = redact_subject(request.subject)
    redacted_body = redact_email_body(request.body)
    entry = {
        "feedback_id": feedback_id, "prediction_id": request.prediction_id,
        "stored_at_utc": stored_at_utc, "sender": request.sender,
        "subject": redacted_subject, "body": redacted_body,
        "predicted_label": request.predicted_label,
        "predicted_confidence": request.predicted_confidence,
        "user_label": normalized_user_label, "verdict": verdict,
        "notes": request.notes.strip(), "source": request.source,
        "model_version": str(model_metadata.get("model_name", "unknown")),
    }
    try:
        append_feedback_entry(entry, settings.feedback_log_path)
    except FeedbackStoreError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return FeedbackResponse(feedback_id=feedback_id, verdict=verdict, stored_at_utc=stored_at_utc)
