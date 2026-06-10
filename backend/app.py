from __future__ import annotations

import json
import pickle
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from feedback_store import FeedbackStoreError, append_feedback_entry, feedback_backend_name, feedback_summary
from runtime_config import load_runtime_config
from spam_detector_core import DEFAULT_SPAM_THRESHOLD, load_domain_catalog, load_user_whitelist, predict_email


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"

MODEL_PATH = MODEL_DIR / "spam_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
TRAIN_MODEL_PATH = MODEL_DIR / "train_model.py"
TRUSTED_DOMAINS_PATH = DATA_DIR / "trusted_domains.csv"
USER_WHITELIST_PATH = DATA_DIR / "whitelist.csv"
FEEDBACK_LOG_PATH = DATA_DIR / "feedback.jsonl"
RUNTIME_CONFIG = load_runtime_config()
TRAINING_TIMEOUT_SECONDS = RUNTIME_CONFIG.retrain_timeout_seconds

model: Any | None = None
vectorizer: Any | None = None
user_whitelist_domains: set[str] = set()
trusted_domain_catalog: set[str] = set()
model_metadata: dict[str, Any] = {}
RETRAIN_LOCK = threading.Lock()


def _load_metadata() -> dict[str, Any]:
    if not METADATA_PATH.exists():
        return {}

    with METADATA_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def current_model_version() -> str:
    return str(model_metadata.get("model_name", "untrained"))


def current_spam_threshold() -> float:
    try:
        return float(model_metadata.get("spam_threshold", DEFAULT_SPAM_THRESHOLD))
    except (TypeError, ValueError):
        return DEFAULT_SPAM_THRESHOLD


def feedback_rows_used_from_training() -> int:
    training_info = model_metadata.get("feedback_training") or {}
    try:
        return int(training_info.get("feedback_rows_used", 0))
    except (TypeError, ValueError):
        return 0


def feedback_last_consumed_utc() -> str | None:
    training_info = model_metadata.get("feedback_training") or {}
    value = training_info.get("last_feedback_at_utc")
    return str(value) if value else None


def current_dataset_rows() -> int:
    try:
        return int(model_metadata.get("dataset_rows", 0))
    except (TypeError, ValueError):
        return 0


def current_spam_f1() -> float | None:
    selected_metrics = model_metadata.get("selected_metrics") or {}
    value = selected_metrics.get("spam_f1")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def load_resources() -> None:
    global model, vectorizer, user_whitelist_domains, trusted_domain_catalog, model_metadata

    model_metadata = _load_metadata()

    if MODEL_PATH.exists() and VECTORIZER_PATH.exists():
        with MODEL_PATH.open("rb") as model_handle:
            model = pickle.load(model_handle)
        with VECTORIZER_PATH.open("rb") as vectorizer_handle:
            vectorizer = pickle.load(vectorizer_handle)
    else:
        model = None
        vectorizer = None

    user_whitelist_domains = load_user_whitelist(USER_WHITELIST_PATH)
    trusted_domain_catalog = load_domain_catalog(TRUSTED_DOMAINS_PATH)


def ensure_model_ready() -> None:
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Install requirements and run backend/model/train_model.py first.",
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_resources()
    yield


app = FastAPI(
    title="Spam Detector API",
    version="2.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=RUNTIME_CONFIG.allow_origin_regex,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class EmailRequest(BaseModel):
    sender: str = Field(default="")
    subject: str = Field(default="")
    body: str = Field(default="")


class BatchPredictionRequest(BaseModel):
    emails: list[EmailRequest]


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    reason: str
    analysis: str
    model_version: str
    sender_domain: str = ""
    rule_layer: str
    signals: list[str] = Field(default_factory=list)
    explanations: list[str] = Field(default_factory=list)
    prediction_id: str
    evaluated_at_utc: str
    spam_prob: float | None = None
    ham_prob: float | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vectorizer_loaded: bool
    feedback_backend: str
    user_whitelist_count: int
    trusted_domain_catalog_count: int
    feedback_count: int
    feedback_rows_used: int
    feedback_last_consumed_utc: str | None = None
    feedback_store_error: str | None = None
    trained_at_utc: str | None = None
    model_version: str
    spam_threshold: float


class FeedbackRequest(BaseModel):
    prediction_id: str
    sender: str = Field(default="")
    subject: str = Field(default="")
    body: str = Field(default="")
    predicted_label: str
    predicted_confidence: float | None = None
    user_label: str
    notes: str = Field(default="", max_length=1000)
    source: str = Field(default="extension_popup")


class FeedbackResponse(BaseModel):
    feedback_id: str
    verdict: str
    stored_at_utc: str


class FeedbackSummaryResponse(BaseModel):
    feedback_count: int
    verdict_counts: dict[str, int]


class RetrainResponse(BaseModel):
    status: str
    model_version: str
    feedback_backend: str
    trained_at_utc: str | None = None
    dataset_rows: int
    feedback_rows_used: int
    feedback_last_consumed_utc: str | None = None
    spam_f1: float | None = None


@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    return health()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        summary = feedback_summary(FEEDBACK_LOG_PATH)
        backend_name = feedback_backend_name(FEEDBACK_LOG_PATH)
        feedback_error = None
        status = "ok"
    except FeedbackStoreError as error:
        summary = {
            "feedback_count": 0,
            "verdict_counts": {
                "correct": 0,
                "false_positive": 0,
                "false_negative": 0,
            },
        }
        backend_name = "unavailable"
        feedback_error = str(error)
        status = "degraded"

    return HealthResponse(
        status=status,
        model_loaded=model is not None,
        vectorizer_loaded=vectorizer is not None,
        feedback_backend=backend_name,
        user_whitelist_count=len(user_whitelist_domains),
        trusted_domain_catalog_count=len(trusted_domain_catalog),
        feedback_count=summary["feedback_count"],
        feedback_rows_used=feedback_rows_used_from_training(),
        feedback_last_consumed_utc=feedback_last_consumed_utc(),
        feedback_store_error=feedback_error,
        trained_at_utc=model_metadata.get("trained_at_utc"),
        model_version=current_model_version(),
        spam_threshold=current_spam_threshold(),
    )


@app.get("/feedback/summary", response_model=FeedbackSummaryResponse)
def feedback_summary_endpoint() -> FeedbackSummaryResponse:
    try:
        summary = feedback_summary(FEEDBACK_LOG_PATH)
    except FeedbackStoreError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return FeedbackSummaryResponse(**summary)


@app.post("/predict", response_model=PredictionResponse)
def predict_spam(email: EmailRequest) -> PredictionResponse:
    ensure_model_ready()
    result = predict_email(
        model=model,
        vectorizer=vectorizer,
        sender=email.sender,
        subject=email.subject,
        body=email.body,
        whitelist_domains=user_whitelist_domains,
        trusted_service_domains=trusted_domain_catalog,
        model_version=current_model_version(),
        spam_threshold=current_spam_threshold(),
    )
    return PredictionResponse(**result.to_payload())


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_spam_batch(request: BatchPredictionRequest) -> list[PredictionResponse]:
    ensure_model_ready()

    return [
        PredictionResponse(
            **predict_email(
                model=model,
                vectorizer=vectorizer,
                sender=email.sender,
                subject=email.subject,
                body=email.body,
                whitelist_domains=user_whitelist_domains,
                trusted_service_domains=trusted_domain_catalog,
                model_version=current_model_version(),
                spam_threshold=current_spam_threshold(),
            ).to_payload()
        )
        for email in request.emails
    ]


@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    ensure_model_ready()

    try:
        normalized_user_label = _normalize_user_label(request.user_label)
        verdict = _feedback_verdict(request.predicted_label, normalized_user_label)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    stored_at_utc = datetime.now(timezone.utc).isoformat()
    feedback_id = f"fb_{request.prediction_id}_{stored_at_utc.replace(':', '').replace('-', '')}"

    entry = {
        "feedback_id": feedback_id,
        "prediction_id": request.prediction_id,
        "stored_at_utc": stored_at_utc,
        "sender": request.sender,
        "subject": request.subject,
        "body": request.body,
        "predicted_label": request.predicted_label,
        "predicted_confidence": request.predicted_confidence,
        "user_label": normalized_user_label,
        "verdict": verdict,
        "notes": request.notes.strip(),
        "source": request.source,
        "model_version": current_model_version(),
    }
    try:
        append_feedback_entry(entry, FEEDBACK_LOG_PATH)
    except FeedbackStoreError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    return FeedbackResponse(
        feedback_id=feedback_id,
        verdict=verdict,
        stored_at_utc=stored_at_utc,
    )


@app.post("/retrain", response_model=RetrainResponse)
def retrain_model() -> RetrainResponse:
    if not RETRAIN_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Retraining is already in progress.")

    try:
        try:
            result = subprocess.run(
                [sys.executable, str(TRAIN_MODEL_PATH)],
                cwd=str(BASE_DIR.parent),
                capture_output=True,
                text=True,
                timeout=TRAINING_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise HTTPException(status_code=500, detail="Retraining timed out.") from error

        if result.returncode != 0:
            output_lines = [line for line in (result.stderr or "").splitlines() if line.strip()]
            output_lines.extend(line for line in (result.stdout or "").splitlines() if line.strip())
            detail = "\n".join(output_lines[-12:]) if output_lines else "Retraining failed."
            raise HTTPException(status_code=500, detail=detail)

        load_resources()
        return RetrainResponse(
            status="ok",
            model_version=current_model_version(),
            feedback_backend=feedback_backend_name(FEEDBACK_LOG_PATH),
            trained_at_utc=model_metadata.get("trained_at_utc"),
            dataset_rows=current_dataset_rows(),
            feedback_rows_used=feedback_rows_used_from_training(),
            feedback_last_consumed_utc=feedback_last_consumed_utc(),
            spam_f1=current_spam_f1(),
        )
    finally:
        RETRAIN_LOCK.release()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=RUNTIME_CONFIG.api_host,
        port=RUNTIME_CONFIG.api_port,
        log_level=RUNTIME_CONFIG.log_level,
    )
