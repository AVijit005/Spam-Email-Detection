from __future__ import annotations

import csv
import math
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import nltk
import numpy as np
import scipy.sparse as sp
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


NLTK_RESOURCES = (
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
)

DEFAULT_SPAM_THRESHOLD = 0.55
DEFAULT_SUBJECT_WEIGHT = 1

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}")
PHONE_PATTERN = re.compile(r"\b\d[\d\-\(\)\s]{6,}\d\b")
MONEY_PATTERN = re.compile(r"[\$£€]\s*\d+[\d,\.]*|\d+[\d,\.]*\s*[\$£€]")
DOMAIN_PATTERN = re.compile(r"^(?:[a-z0-9-]+\.)+[a-z]{2,}$")
MIXED_TOKEN_PATTERN = re.compile(r"\b(?=\w*[a-z])(?=\w*\d)\w+\b", re.IGNORECASE)

URGENCY_KEYWORDS = {
    "urgent",
    "immediately",
    "asap",
    "suspended",
    "expire",
    "expired",
    "deadline",
    "warning",
}

ACCOUNT_KEYWORDS = {
    "account",
    "password",
    "login",
    "signin",
    "security",
    "verify",
    "verification",
    "identity",
    "otp",
    "bank",
}

CALL_TO_ACTION_KEYWORDS = {
    "click",
    "claim",
    "confirm",
    "reset",
    "verify",
    "open",
    "download",
    "visit",
    "login",
}

PROMOTIONAL_KEYWORDS = {
    "offer",
    "discount",
    "sale",
    "coupon",
    "deal",
    "shop",
    "weekend",
    "save",
    "percent",
    "shipping",
}

CONVERSATIONAL_KEYWORDS = {
    "lunch",
    "coffee",
    "dinner",
    "meeting",
    "office",
    "today",
    "tomorrow",
    "plans",
    "near",
    "still",
}

BUSINESS_KEYWORDS = {
    "review",
    "report",
    "slides",
    "project",
    "team",
    "agenda",
    "meeting",
    "office",
    "update",
    "client",
    "schedule",
}

PHISHING_PHRASES = [
    "you have won",
    "you've been selected",
    "claim your prize",
    "claim now",
    "winner",
    "won a lottery",
    "lottery prize",
    "free money",
    "million dollars",
    "million pound",
    "bitcoin",
    "cryptocurrency",
    "wire transfer",
    "western union",
    "moneygram",
    "click here to verify",
    "verify your account immediately",
    "account suspended",
    "account has been suspended",
    "confirm your identity",
    "password will expire",
    "urgent action required",
    "dear lucky winner",
]

META_FEATURE_NAMES = [
    "url_count",
    "caps_ratio",
    "exclamation_count",
    "question_count",
    "money_count",
    "phone_count",
    "word_count",
    "avg_word_length",
    "digit_ratio",
    "spam_phrase_hits",
    "urgency_hits",
    "account_hits",
    "call_to_action_hits",
    "symbol_ratio",
    "percent_hits",
    "mixed_token_hits",
]

META_FEATURE_LABELS = {
    "url_count": "contains links",
    "caps_ratio": "uses uppercase emphasis",
    "exclamation_count": "uses repeated exclamation marks",
    "question_count": "uses multiple question marks",
    "money_count": "mentions money values",
    "phone_count": "contains a phone number",
    "word_count": "message length",
    "avg_word_length": "long token pattern",
    "digit_ratio": "contains many digits",
    "spam_phrase_hits": "matches phishing phrases",
    "urgency_hits": "contains urgency language",
    "account_hits": "contains account-security language",
    "call_to_action_hits": "contains calls to action",
    "symbol_ratio": "contains many symbols",
    "percent_hits": "contains discount-style percentages",
    "mixed_token_hits": "contains mixed letter-number tokens",
}


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
    "free",
    "win",
    "won",
    "prize",
    "click",
    "now",
    "urgent",
    "limited",
    "cash",
    "offer",
    "call",
    "reply",
    "stop",
    "apply",
    "claim",
}

LEMMATIZER = WordNetLemmatizer()


@dataclass(frozen=True)
class RuleAssessment:
    is_spam: bool
    confidence: float
    reason: str
    signals: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BenignAssessment:
    is_benign: bool
    confidence: float
    reason: str
    analysis: str
    rule_layer: str
    signals: list[str] = field(default_factory=list)
    explanations: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PredictionResult:
    label: str
    confidence: float
    reason: str
    analysis: str
    model_version: str
    sender_domain: str = ""
    rule_layer: str = "ml"
    signals: list[str] = field(default_factory=list)
    explanations: list[str] = field(default_factory=list)
    prediction_id: str = ""
    evaluated_at_utc: str = ""
    spam_prob: float | None = None
    ham_prob: float | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: value for key, value in payload.items() if value is not None}


def _safe_lemmatize(token: str) -> str:
    try:
        return LEMMATIZER.lemmatize(token)
    except LookupError:
        return token


def normalize_domain(value: str | None) -> str:
    if not value:
        return ""

    candidate = value.strip().lower().strip(" <>\"'[]()")
    if not candidate:
        return ""

    email_match = EMAIL_PATTERN.search(candidate)
    if email_match:
        candidate = email_match.group(0).split("@", 1)[1]

    candidate = candidate.split("://", 1)[-1]
    candidate = candidate.split("/", 1)[0]
    candidate = candidate.split("?", 1)[0]
    candidate = candidate.split("#", 1)[0]
    candidate = candidate.split(":", 1)[0]
    candidate = candidate.removeprefix("www.").strip(".")

    return candidate if DOMAIN_PATTERN.fullmatch(candidate) else ""


def extract_sender_domain(sender: str | None) -> str:
    return normalize_domain(sender)


def _read_rows(path: Path) -> tuple[list[list[str]], list[str], bool]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if not rows:
        return [], [], False

    first_row = [cell.strip().lower() for cell in rows[0]]
    has_header = "domain" in first_row or "email" in first_row
    data_rows = rows[1:] if has_header else rows
    return data_rows, first_row, has_header


def load_domain_catalog(*paths: str | Path) -> set[str]:
    domains: set[str] = set()

    for raw_path in paths:
        if not raw_path:
            continue

        path = Path(raw_path)
        if not path.exists():
            continue

        data_rows, _, _ = _read_rows(path)
        for row in data_rows:
            for cell in row:
                domain = normalize_domain(cell)
                if domain:
                    domains.add(domain)

    return domains


def load_user_whitelist(*paths: str | Path) -> set[str]:
    whitelist_domains: set[str] = set()

    for raw_path in paths:
        if not raw_path:
            continue

        path = Path(raw_path)
        if not path.exists():
            continue

        data_rows, first_row, has_header = _read_rows(path)
        domain_index = first_row.index("domain") if has_header and "domain" in first_row else None
        email_index = first_row.index("email") if has_header and "email" in first_row else None

        for row in data_rows:
            candidates: list[str]
            if domain_index is not None and domain_index < len(row):
                candidates = [row[domain_index]]
            elif email_index is not None and email_index < len(row):
                candidates = [row[email_index]]
            else:
                candidates = row

            for cell in candidates:
                domain = normalize_domain(cell)
                if domain:
                    whitelist_domains.add(domain)

    return whitelist_domains


def load_trusted_domains(*paths: str | Path) -> set[str]:
    return load_domain_catalog(*paths)


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    value = text.lower()
    value = URL_PATTERN.sub(" urltoken ", value)
    value = EMAIL_PATTERN.sub(" emailtoken ", value)
    value = PHONE_PATTERN.sub(" phonetoken ", value)
    value = MONEY_PATTERN.sub(" moneytoken ", value)
    value = re.sub(r"[^a-z0-9\s]", " ", value)

    tokens = [
        _safe_lemmatize(token)
        for token in value.split()
        if token not in STOPWORDS and len(token) > 1
    ]
    return " ".join(tokens)


def _coerce_texts(texts: str | Sequence[str]) -> list[str]:
    if isinstance(texts, str):
        return [texts]
    return [text if isinstance(text, str) else "" for text in texts]


def _count_keyword_hits(text: str, keywords: Iterable[str]) -> int:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    keyword_set = set(keywords)
    return sum(1 for token in tokens if token in keyword_set)


def _meta_feature_map(text: str) -> dict[str, float]:
    row = extract_meta_features(text)[0].tolist()
    return dict(zip(META_FEATURE_NAMES, row))


def extract_meta_features(texts: str | Sequence[str]) -> np.ndarray:
    rows: list[list[float]] = []

    for text in _coerce_texts(texts):
        n_chars = max(len(text), 1)
        n_letters = max(sum(char.isalpha() for char in text), 1)
        words = text.split()
        n_words = max(len(words), 1)
        avg_word_length = sum(len(word) for word in words) / n_words
        symbol_ratio = sum(not char.isalnum() and not char.isspace() for char in text) / n_chars

        rows.append(
            [
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
            ]
        )

    return np.array(rows, dtype=np.float32)


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


def is_trusted_service_domain(sender_domain: str, trusted_service_domains: Iterable[str] | None = None) -> bool:
    catalog = set(trusted_service_domains or ())
    return any(sender_domain == domain or sender_domain.endswith(f".{domain}") for domain in catalog)


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


def assess_rule_based_spam(subject: str, body: str) -> RuleAssessment:
    phrases = matched_spam_phrases(subject, body)
    raw_text = compose_email_text(subject, body, subject_weight=1)
    signals = _indicator_signals(raw_text)

    if len(phrases) >= 2 or (len(phrases) >= 1 and len(signals) >= 2):
        confidence = min(0.86 + (0.04 * len(phrases)) + (0.02 * len(signals)), 0.99)
        top_phrases = ", ".join(phrases[:2])
        phrase_signal = f"matched phrases: {top_phrases}" if top_phrases else ""
        all_signals = [signal for signal in [phrase_signal, *signals] if signal][:5]
        return RuleAssessment(
            is_spam=True,
            confidence=round(confidence, 2),
            reason="Contains multiple high-risk phishing or spam indicators",
            signals=all_signals,
        )

    return RuleAssessment(is_spam=False, confidence=0.0, reason="", signals=signals[:3])


def assess_benign_email(subject: str, body: str) -> BenignAssessment:
    raw_text = compose_email_text(subject, body, subject_weight=1)
    lowered = raw_text.lower()
    feature_map = _meta_feature_map(raw_text)
    phrase_hits = matched_spam_phrases(subject, body)
    promotional_hits = _count_keyword_hits(lowered, PROMOTIONAL_KEYWORDS)
    conversational_hits = _count_keyword_hits(lowered, CONVERSATIONAL_KEYWORDS)
    business_hits = _count_keyword_hits(lowered, BUSINESS_KEYWORDS)

    has_high_risk_indicator = any(
        (
            feature_map["url_count"] >= 1,
            feature_map["money_count"] >= 1,
            feature_map["phone_count"] >= 1,
            feature_map["account_hits"] >= 1,
            feature_map["mixed_token_hits"] >= 2,
            feature_map["call_to_action_hits"] >= 2,
            len(phrase_hits) >= 1,
        )
    )

    if not has_high_risk_indicator and conversational_hits >= 2 and feature_map["word_count"] <= 40:
        signals = ["conversation-style wording"]
        if business_hits >= 1:
            signals.append("routine work context")
        return BenignAssessment(
            is_benign=True,
            confidence=0.82,
            reason="Looks like a routine personal or workplace conversation",
            analysis="Benign-context detection found conversational wording without phishing-style indicators.",
            rule_layer="benign_context",
            signals=signals,
            explanations=[
                "Conversation-style wording matched a low-risk pattern.",
                "No links, security prompts, or high-risk phishing markers were found.",
            ],
        )

    if (
        not has_high_risk_indicator
        and promotional_hits >= 2
        and feature_map["exclamation_count"] <= 1
        and feature_map["caps_ratio"] < 0.2
    ):
        return BenignAssessment(
            is_benign=True,
            confidence=0.76,
            reason="Looks like a low-risk promotional message, not phishing",
            analysis="Benign promotional detection found retail-style language without phishing-style indicators.",
            rule_layer="benign_promo",
            signals=["retail or promotional wording"],
            explanations=[
                "Promotional wording was present, but there were no links, account prompts, or credential requests.",
            ],
        )

    return BenignAssessment(
        is_benign=False,
        confidence=0.0,
        reason="",
        analysis="",
        rule_layer="ml",
    )


def _vectorizer_bundle(vectorizer: Any) -> dict[str, Any]:
    if isinstance(vectorizer, dict):
        return vectorizer
    return {
        "version": 1,
        "word_vectorizer": vectorizer,
        "char_vectorizer": None,
        "meta_feature_names": META_FEATURE_NAMES,
    }


def _build_feature_parts(vectorizer: Any, raw_text: str, processed_text: str) -> tuple[sp.csr_matrix, list[str], list[int]]:
    bundle = _vectorizer_bundle(vectorizer)
    feature_parts: list[sp.csr_matrix] = []
    feature_names: list[str] = []
    part_sizes: list[int] = []

    word_vectorizer = bundle.get("word_vectorizer")
    if word_vectorizer is not None:
        word_matrix = word_vectorizer.transform([processed_text])
        feature_parts.append(word_matrix)
        names = [f"word:{name}" for name in word_vectorizer.get_feature_names_out()]
        feature_names.extend(names)
        part_sizes.append(len(names))

    char_vectorizer = bundle.get("char_vectorizer")
    if char_vectorizer is not None:
        char_matrix = char_vectorizer.transform([raw_text.lower()])
        feature_parts.append(char_matrix)
        names = [f"char:{name}" for name in char_vectorizer.get_feature_names_out()]
        feature_names.extend(names)
        part_sizes.append(len(names))

    meta_names = bundle.get("meta_feature_names", META_FEATURE_NAMES)
    meta_matrix = sp.csr_matrix(extract_meta_features(raw_text))
    feature_parts.append(meta_matrix)
    meta_feature_names = [f"meta:{name}" for name in meta_names]
    feature_names.extend(meta_feature_names)
    part_sizes.append(len(meta_feature_names))

    matrix = sp.hstack(feature_parts, format="csr")
    return matrix, feature_names, part_sizes


def build_feature_matrix(
    vectorizer: Any,
    subject: str,
    body: str,
    *,
    subject_weight: int = DEFAULT_SUBJECT_WEIGHT,
) -> tuple[sp.csr_matrix, list[str]]:
    raw_text = compose_email_text(subject, body, subject_weight=subject_weight)
    processed_text = preprocess_text(raw_text)
    matrix, feature_names, _ = _build_feature_parts(vectorizer, raw_text, processed_text)
    return matrix, feature_names


def _probabilities_from_model(model: Any, features: sp.csr_matrix) -> tuple[float, float]:
    if hasattr(model, "predict_proba"):
        ham_probability, spam_probability = model.predict_proba(features)[0]
        return float(spam_probability), float(ham_probability)

    decision = float(model.decision_function(features)[0])
    spam_probability = 1.0 / (1.0 + math.exp(-decision))
    return spam_probability, 1.0 - spam_probability


def _format_feature_explanation(feature_name: str, is_spam: bool) -> str:
    if feature_name.startswith("meta:"):
        meta_name = feature_name.split(":", 1)[1]
        label = META_FEATURE_LABELS.get(meta_name, meta_name.replace("_", " "))
        return f"{'Suspicious' if is_spam else 'Legitimate'} signal: {label}"

    if feature_name.startswith("word:"):
        token = feature_name.split(":", 1)[1]
        return f"{'Suspicious' if is_spam else 'Legitimate'} token: \"{token}\""

    if feature_name.startswith("char:"):
        token = feature_name.split(":", 1)[1]
        return f"{'Suspicious' if is_spam else 'Legitimate'} pattern: \"{token}\""

    return feature_name


def explain_prediction(model: Any, features: sp.csr_matrix, feature_names: list[str], label: str) -> list[str]:
    if not hasattr(model, "coef_"):
        return []

    coefficients = np.asarray(model.coef_[0]).ravel()
    active_indices = features.indices
    active_values = features.data
    contributions = active_values * coefficients[active_indices]

    if label == "Spam":
        candidate_pairs = [
            (feature_names[index], contribution)
            for index, contribution in zip(active_indices, contributions)
            if contribution > 0
        ]
        candidate_pairs.sort(key=lambda item: item[1], reverse=True)
        top_pairs = candidate_pairs[:4]
        return [_format_feature_explanation(name, True) for name, _ in top_pairs]

    candidate_pairs = [
        (feature_names[index], contribution)
        for index, contribution in zip(active_indices, contributions)
        if contribution < 0
    ]
    candidate_pairs.sort(key=lambda item: item[1])
    top_pairs = candidate_pairs[:4]
    return [_format_feature_explanation(name, False) for name, _ in top_pairs]


def _base_result_payload(
    *,
    label: str,
    confidence: float,
    reason: str,
    analysis: str,
    model_version: str,
    sender_domain: str,
    rule_layer: str,
    signals: list[str],
    explanations: list[str],
    spam_prob: float | None = None,
    ham_prob: float | None = None,
) -> PredictionResult:
    return PredictionResult(
        label=label,
        confidence=confidence,
        reason=reason,
        analysis=analysis,
        model_version=model_version,
        sender_domain=sender_domain,
        rule_layer=rule_layer,
        signals=signals,
        explanations=explanations,
        prediction_id=uuid.uuid4().hex,
        evaluated_at_utc=datetime.now(timezone.utc).isoformat(),
        spam_prob=spam_prob,
        ham_prob=ham_prob,
    )


def predict_email(
    *,
    model: Any,
    vectorizer: Any,
    sender: str,
    subject: str,
    body: str,
    whitelist_domains: Iterable[str] | None = None,
    trusted_service_domains: Iterable[str] | None = None,
    model_version: str = "unknown",
    spam_threshold: float = DEFAULT_SPAM_THRESHOLD,
) -> PredictionResult:
    sender_domain = extract_sender_domain(sender)
    whitelisted = set(whitelist_domains or ())

    if sender_domain and sender_domain in whitelisted:
        return _base_result_payload(
            label="whitelisted",
            confidence=1.0,
            reason="Sender is in your trusted whitelist",
            analysis="Trusted sender matched your local whitelist.",
            model_version=model_version,
            sender_domain=sender_domain,
            rule_layer="whitelist",
            signals=["trusted sender domain"],
            explanations=["Whitelisted sender domain matched your local settings."],
        )

    if sender_domain and is_trusted_service_domain(sender_domain, trusted_service_domains):
        return _base_result_payload(
            label="Not Spam",
            confidence=0.97,
            reason="Sender recognised as a legitimate financial or service provider",
            analysis="Trusted service domain matched the curated built-in catalog.",
            model_version=model_version,
            sender_domain=sender_domain,
            rule_layer="trusted_service",
            signals=["trusted service domain"],
            explanations=["Trusted service domain matched the built-in service catalog."],
        )

    rule_assessment = assess_rule_based_spam(subject, body)
    if rule_assessment.is_spam:
        return _base_result_payload(
            label="Spam",
            confidence=rule_assessment.confidence,
            reason=rule_assessment.reason,
            analysis="Rule-based detection found multiple phishing-style signals before the ML model ran.",
            model_version=model_version,
            sender_domain=sender_domain,
            rule_layer="rules",
            signals=rule_assessment.signals,
            explanations=rule_assessment.signals[:4],
        )

    benign_assessment = assess_benign_email(subject, body)
    if benign_assessment.is_benign:
        return _base_result_payload(
            label="Not Spam",
            confidence=benign_assessment.confidence,
            reason=benign_assessment.reason,
            analysis=benign_assessment.analysis,
            model_version=model_version,
            sender_domain=sender_domain,
            rule_layer=benign_assessment.rule_layer,
            signals=benign_assessment.signals,
            explanations=benign_assessment.explanations,
        )

    features, feature_names = build_feature_matrix(vectorizer, subject, body)
    spam_probability, ham_probability = _probabilities_from_model(model, features)

    if spam_probability >= spam_threshold:
        return _base_result_payload(
            label="Spam",
            confidence=round(spam_probability, 2),
            reason="Machine learning model detected suspicious patterns",
            analysis=f"AI analysis: {spam_probability:.1%} spam probability based on text and phishing-oriented metadata features.",
            model_version=model_version,
            sender_domain=sender_domain,
            rule_layer="ml",
            signals=rule_assessment.signals,
            explanations=explain_prediction(model, features, feature_names, "Spam"),
            spam_prob=round(spam_probability, 4),
            ham_prob=round(ham_probability, 4),
        )

    return _base_result_payload(
        label="Not Spam",
        confidence=round(ham_probability, 2),
        reason="Appears to be a legitimate email",
        analysis=f"AI analysis: {ham_probability:.1%} confidence that the message is legitimate.",
        model_version=model_version,
        sender_domain=sender_domain,
        rule_layer="ml",
        signals=rule_assessment.signals,
        explanations=explain_prediction(model, features, feature_names, "Not Spam"),
        spam_prob=round(spam_probability, 4),
        ham_prob=round(ham_probability, 4),
    )
