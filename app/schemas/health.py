from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vectorizer_loaded: bool
    ensemble_active: bool = False
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
