from __future__ import annotations

import re
from typing import Iterable, Sequence

import numpy as np

from app.core.constants import (
    ACCOUNT_KEYWORDS,
    ATTACHMENT_EXTENSIONS,
    ATTACHMENT_PHRASES,
    CALL_TO_ACTION_KEYWORDS,
    CREDENTIAL_HARVESTING_PHRASES,
    CSS_HIDDEN_PATTERN,
    DEFAULT_SUBJECT_WEIGHT,
    HIGH_RISK_TLDS,
    HOMOGRAPH_CHAR_PATTERN,
    HTML_COMMENT_PATTERN,
    HTML_TAG_PATTERN,
    IP_IN_URL_PATTERN,
    META_FEATURE_NAMES,
    MIXED_TOKEN_PATTERN,
    MONEY_PATTERN,
    PHISHING_PHRASES,
    PHONE_PATTERN,
    PROMOTIONAL_KEYWORDS,
    SHORTENED_URL_PATTERN,
    SUSPICIOUS_TLDS,
    UNICODE_OBFUSCATION_PATTERN,
    URGENCY_KEYWORDS,
    URL_PATTERN,
)


def _count_keyword_hits(text: str, keywords: Iterable[str]) -> int:
    lowered = text.lower()
    return sum(1 for keyword in keywords if keyword in lowered)


def _count_phrase_hits(text: str, phrases: Iterable[str]) -> int:
    lowered = text.lower()
    return sum(1 for phrase in phrases if phrase in lowered)


def _extract_urls(text: str) -> list[str]:
    return URL_PATTERN.findall(text)


def _extract_domain(url: str) -> str:
    url = url.lower().split("://")[-1].split("/")[0].split("?")[0]
    url = url.split("@")[-1].split(":")[0]
    url = url.removeprefix("www.").strip(".")
    return url


def _url_features(text: str) -> dict[str, float]:
    urls = _extract_urls(text)
    domains = [_extract_domain(u) for u in urls]
    unique_domains = len(set(domains))
    shortened = sum(1 for u in urls if SHORTENED_URL_PATTERN.search(u))
    ip_urls = sum(1 for u in urls if IP_IN_URL_PATTERN.search(u))
    suspicious = sum(
        1 for u in urls
        if any(u.lower().endswith(tld) for tld in SUSPICIOUS_TLDS)
    )
    high_risk = sum(
        1 for u in urls
        if any(u.lower().endswith(tld) for tld in HIGH_RISK_TLDS)
    )
    url_to_text = len("".join(urls)) / max(len(text), 1)
    return {
        "url_count": float(len(urls)),
        "unique_url_domains": float(unique_domains),
        "shortened_url_count": float(shortened),
        "ip_url_count": float(ip_urls),
        "suspicious_tld_count": float(suspicious),
        "high_risk_tld_count": float(high_risk),
        "url_to_text_ratio": url_to_text,
    }


def _html_features(text: str) -> dict[str, float]:
    tags = len(HTML_TAG_PATTERN.findall(text))
    comments = len(HTML_COMMENT_PATTERN.findall(text))
    hidden = len(CSS_HIDDEN_PATTERN.findall(text.lower()))
    return {
        "html_tag_count": float(tags + comments),
        "hidden_element_indicators": float(hidden),
    }


def _text_quality_features(text: str, words: list[str]) -> dict[str, float]:
    n_words = max(len(words), 1)
    unique_words = len(set(words))
    type_token_ratio = unique_words / n_words

    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    n_sentences = max(len(sentences), 1)
    avg_sentence_length = n_words / n_sentences

    total_syllables = 0
    for word in words:
        w = word.lower()
        syllables = len(re.findall(r"[aeiouy]+", w)) or 1
        total_syllables += syllables

    flesch = 206.835 - 1.015 * avg_sentence_length - 84.6 * (total_syllables / n_words)
    flesch = max(0.0, min(120.0, flesch))

    imperative_verbs = {"click", "buy", "send", "call", "open", "visit",
                        "download", "claim", "confirm", "verify", "register",
                        "submit", "update", "check", "review", "enter", "sign"}
    imperative_hits = sum(1 for w in words if w.lower() in imperative_verbs)
    imperative_ratio = imperative_hits / n_words

    return {
        "flesch_reading_ease": float(flesch),
        "type_token_ratio": float(type_token_ratio),
        "imperative_verb_ratio": float(imperative_ratio),
    }


def _obfuscation_features(text: str) -> dict[str, float]:
    n_chars = max(len(text), 1)
    unicode_chars = len(UNICODE_OBFUSCATION_PATTERN.findall(text))
    homograph_chars = len(HOMOGRAPH_CHAR_PATTERN.findall(text))
    return {
        "homograph_char_ratio": homograph_chars / n_chars,
        "unicode_obfuscation_ratio": unicode_chars / n_chars,
    }


def _attachment_features(text: str) -> dict[str, float]:
    text_lower = text.lower()
    phrase_hits = _count_phrase_hits(text_lower, ATTACHMENT_PHRASES)
    ext_hits = sum(1 for ext in ATTACHMENT_EXTENSIONS if ext.replace(".", "") in text_lower)
    return {"attachment_indicators": float(phrase_hits + ext_hits)}


def _credential_harvesting_features(text: str) -> dict[str, float]:
    hits = _count_phrase_hits(text, CREDENTIAL_HARVESTING_PHRASES)
    return {"credential_harvesting_hits": float(hits)}


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

        lowered = text.lower()
        phrase_hits = len(matched_spam_phrases("", text))
        promo_hits = _count_keyword_hits(lowered, PROMOTIONAL_KEYWORDS)

        url_f = _url_features(text)
        html_f = _html_features(text)
        quality_f = _text_quality_features(text, words)
        obfuscation_f = _obfuscation_features(text)
        attachment_f = _attachment_features(text)
        credential_f = _credential_harvesting_features(text)

        rows.append([
            url_f["url_count"],
            sum(char.isupper() for char in text) / n_letters,
            float(text.count("!")),
            float(text.count("?")),
            float(len(MONEY_PATTERN.findall(text))),
            float(len(PHONE_PATTERN.findall(text))),
            float(n_words),
            float(avg_word_length),
            sum(char.isdigit() for char in text) / n_chars,
            float(phrase_hits),
            float(_count_keyword_hits(lowered, URGENCY_KEYWORDS)),
            float(_count_keyword_hits(lowered, ACCOUNT_KEYWORDS)),
            float(_count_keyword_hits(lowered, CALL_TO_ACTION_KEYWORDS)),
            float(symbol_ratio),
            float(text.count("%")),
            float(len(MIXED_TOKEN_PATTERN.findall(text))),
            url_f["unique_url_domains"],
            url_f["shortened_url_count"],
            url_f["ip_url_count"],
            url_f["suspicious_tld_count"],
            url_f["high_risk_tld_count"],
            url_f["url_to_text_ratio"],
            attachment_f["attachment_indicators"],
            html_f["html_tag_count"],
            html_f["hidden_element_indicators"],
            obfuscation_f["homograph_char_ratio"],
            obfuscation_f["unicode_obfuscation_ratio"],
            quality_f["flesch_reading_ease"],
            quality_f["type_token_ratio"],
            quality_f["imperative_verb_ratio"],
            float(promo_hits),
            credential_f["credential_harvesting_hits"],
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
    if feature_map["shortened_url_count"] >= 1:
        signals.append("uses shortened URLs")
    if feature_map["ip_url_count"] >= 1:
        signals.append("links to IP addresses")
    if feature_map["suspicious_tld_count"] >= 1:
        signals.append("links to suspicious domains")
    if feature_map["attachment_indicators"] >= 1:
        signals.append("mentions attachments")
    if feature_map["hidden_element_indicators"] >= 1:
        signals.append("contains hidden text elements")
    if feature_map["credential_harvesting_hits"] >= 1:
        signals.append("matches credential harvesting patterns")
    return signals
