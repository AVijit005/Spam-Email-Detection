from __future__ import annotations

from pydantic import BaseModel, Field


class EmailRequest(BaseModel):
    sender: str = Field(default="", max_length=320)
    subject: str = Field(default="", max_length=998)
    body: str = Field(default="", max_length=100_000)


class BatchPredictionRequest(BaseModel):
    emails: list[EmailRequest] = Field(default_factory=list, max_length=50)


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
