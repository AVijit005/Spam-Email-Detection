from __future__ import annotations

import re
from typing import Iterable, Sequence

import numpy as np

from app.core.constants import (
    ACCOUNT_KEYWORDS, CALL_TO_ACTION_KEYWORDS, DEFAULT_SUBJECT_WEIGHT,
    META_FEATURE_NAMES, MIXED_TOKEN_PATTERN, MONEY_PATTERN,
    PHISHING_PHRASES, PHONE_PATTERN, URGENCY_KEYWORDS, URL_PATTERN,
)


def _count_keyword_hits(text: str, keywords: Iterable[str]) -> int:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    keyword_set = set(keywords)
    return sum(1 for token in tokens if token in keyword_set)


def matched_spam_phrases(subject: str, body: str) -> list[str]:
    combined_text = f"{subject} {body}".lower()
    return [phrase for phrase in PHISHING_PHRASES if phrase in combined_text]


def compose_email_text(subject: str, body: str, subject_weight: int = DEFAULT_SUBJECT_WEIGHT) -> str:
    subject_text = subject.strip()
    body_text = body.strip()
    parts: list[str] = []
    if subject_text:
        parts.extend([subject_text] * max(subject_weight, 1))
    if body_text:
        parts.append(body_text)
    return " ".join(parts).strip()


def _coerce_texts(texts: str | Sequence[str]) -> list[str]:
    if isinstance(texts, str):
        return [texts]
    return [text if isinstance(text, str) else "" for text in texts]


def extract_meta_features(texts: str | Sequence[str]) -> np.ndarray:
    rows: list[list[float]] = []
    for text in _coerce_texts(texts):
        n_chars = max(len(text), 1)
        n_letters = max(sum(char.isalpha() for char in text), 1)
        words = text.split()
        n_words = max(len(words), 1)
        avg_word_length = sum(len(word) for word in words) / n_words
        symbol_ratio = sum(not char.isalnum() and not char.isspace() for char in text) / n_chars
        rows.append([
            len(URL_PATTERN.findall(text)),
            sum(char.isupper() for char in text) / n_letters,
            text.count("!"),
            text.count("?"),
            len(MONEY_PATTERN.findall(text)),
            len(PHONE_PATTERN.findall(text)),
            n_words,
            avg_word_length,
            sum(char.isdigit() for char in text) / n_chars,
            len(matched_spam_phrases(text, "")),
            _count_keyword_hits(text, URGENCY_KEYWORDS),
            _count_keyword_hits(text, ACCOUNT_KEYWORDS),
            _count_keyword_hits(text, CALL_TO_ACTION_KEYWORDS),
            symbol_ratio,
            text.count("%"),
            len(MIXED_TOKEN_PATTERN.findall(text)),
        ])
    return np.array(rows, dtype=np.float32)


def _meta_feature_map(text: str) -> dict[str, float]:
    row = extract_meta_features(text)[0].tolist()
    return dict(zip(META_FEATURE_NAMES, row))


def _indicator_signals(raw_text: str) -> list[str]:
    feature_map = _meta_feature_map(raw_text)
    signals: list[str] = []
    if feature_map["url_count"] >= 1:
        signals.append("contains a link")
    if feature_map["money_count"] >= 1:
        signals.append("mentions money amounts")
    if feature_map["phone_count"] >= 1:
        signals.append("contains a phone number")
    if feature_map["exclamation_count"] >= 3:
        signals.append("uses aggressive punctuation")
    if feature_map["caps_ratio"] >= 0.28 and feature_map["word_count"] >= 5:
        signals.append("uses excessive uppercase")
    if feature_map["digit_ratio"] >= 0.12 and feature_map["word_count"] >= 5:
        signals.append("contains a high ratio of digits")
    if feature_map["urgency_hits"] >= 2:
        signals.append("contains urgency language")
    if feature_map["account_hits"] >= 2:
        signals.append("contains account-security terms")
    if feature_map["call_to_action_hits"] >= 2:
        signals.append("contains direct calls to action")
    return signals
