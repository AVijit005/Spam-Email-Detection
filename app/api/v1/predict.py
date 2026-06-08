from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.core.detector import predict_email
from app.schemas.email import BatchPredictionRequest, EmailRequest, PredictionResponse

router = APIRouter()

model = None
vectorizer = None
user_whitelist_domains: set[str] = set()
trusted_domain_catalog: set[str] = set()
model_metadata: dict = {}


def _ensure_model_ready() -> None:
    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")


@router.post("/predict", response_model=PredictionResponse)
def predict_spam(email: EmailRequest) -> PredictionResponse:
    _ensure_model_ready()
    result = predict_email(
        model=model, vectorizer=vectorizer,
        sender=email.sender, subject=email.subject, body=email.body,
        whitelist_domains=user_whitelist_domains,
        trusted_service_domains=trusted_domain_catalog,
        model_version=str(model_metadata.get("model_name", "unknown")),
        spam_threshold=settings.spam_threshold,
    )
    return PredictionResponse(**result.to_payload())


@router.post("/predict/batch", response_model=list[PredictionResponse])
def predict_spam_batch(request: BatchPredictionRequest) -> list[PredictionResponse]:
    _ensure_model_ready()
    return [
        PredictionResponse(**predict_email(
            model=model, vectorizer=vectorizer,
            sender=email.sender, subject=email.subject, body=email.body,
            whitelist_domains=user_whitelist_domains,
            trusted_service_domains=trusted_domain_catalog,
            model_version=str(model_metadata.get("model_name", "unknown")),
            spam_threshold=settings.spam_threshold,
        ).to_payload())
        for email in request.emails
    ]
