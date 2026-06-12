from __future__ import annotations

import re

import nltk
from nltk.corpus import stopwords

from app.core.constants import (
    EMAIL_PATTERN,
    MONEY_PATTERN,
    NLTK_RESOURCES,
    PHONE_PATTERN,
    URL_PATTERN,
)


def _ensure_nltk_resources() -> None:
    for resource_path, download_name in NLTK_RESOURCES:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(download_name, quiet=True)
            except Exception:
                pass


_ensure_nltk_resources()

try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    STOPWORDS = set()

STOPWORDS -= {
    "free", "win", "won", "prize", "click", "now", "urgent",
    "limited", "cash", "offer", "call", "reply", "stop", "apply", "claim",
    "no", "not", "never", "don", "isn", "wasn", "aren", "won",
}


def preprocess_text(text: str) -> str:
    """Preprocess text for TF-IDF vectorization.

    Strategy: lowercase, replace structured tokens with placeholders,
    strip punctuation, remove stopwords. NO stemming or lemmatization —
    morphological variation is signal for spam detection, not noise.
    """
    if not isinstance(text, str):
        return ""
    value = text.lower()
    value = URL_PATTERN.sub(" urltoken ", value)
    value = EMAIL_PATTERN.sub(" emailtoken ", value)
    value = PHONE_PATTERN.sub(" phonetoken ", value)
    value = MONEY_PATTERN.sub(" moneytoken ", value)
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    tokens = [
        token for token in value.split()
        if token not in STOPWORDS and len(token) > 1
    ]
    return " ".join(tokens)
