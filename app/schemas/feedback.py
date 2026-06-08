from __future__ import annotations

from pydantic import BaseModel, Field


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
