from __future__ import annotations

from pydantic import BaseModel


class RetrainResponse(BaseModel):
    status: str
    model_version: str
    feedback_backend: str
    trained_at_utc: str | None = None
    dataset_rows: int
    feedback_rows_used: int
    feedback_last_consumed_utc: str | None = None
    spam_f1: float | None = None
    ensemble_f1: float | None = None
