from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

import scipy.sparse as sp

from app.core.constants import DEFAULT_SPAM_THRESHOLD, DEFAULT_SUBJECT_WEIGHT, META_FEATURE_NAMES
from app.core.domain import extract_sender_domain
from app.core.explain import explain_prediction
from app.core.features import compose_email_text, extract_meta_features
from app.core.rules import BenignAssessment, RuleAssessment, assess_benign_email, assess_rule_based_spam, is_trusted_service_domain
from app.core.text import preprocess_text
from app.utils.pii import redact_email_body, redact_subject


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


def _base_result_payload(
    *, label: str, confidence: float, reason: str, analysis: str,
    model_version: str, sender_domain: str, rule_layer: str,
    signals: list[str], explanations: list[str],
    spam_prob: float | None = None, ham_prob: float | None = None,
) -> PredictionResult:
    return PredictionResult(
        label=label, confidence=confidence, reason=reason, analysis=analysis,
        model_version=model_version, sender_domain=sender_domain, rule_layer=rule_layer,
        signals=signals, explanations=explanations,
        prediction_id=uuid.uuid4().hex,
        evaluated_at_utc=datetime.now(timezone.utc).isoformat(),
        spam_prob=spam_prob, ham_prob=ham_prob,
    )


def _vectorizer_bundle(vectorizer: Any) -> dict[str, Any]:
    if isinstance(vectorizer, dict):
        return vectorizer
    return {"version": 1, "word_vec": vectorizer, "model_type": "classical",
            "char_vectorizer": None, "meta_feature_names": META_FEATURE_NAMES}


def _is_transformer_model(model: Any) -> bool:
    try:
        return hasattr(model, "config") and hasattr(model.config, "model_type") and hasattr(model, "forward")
    except Exception:
        return False


def _is_ensemble_model(model: Any) -> bool:
    try:
        cls_name = type(model).__name__
    except Exception:
        cls_name = ""
    return cls_name == "EnsemblePredictor"


def _build_feature_parts(vectorizer: Any, raw_text: str, processed_text: str) -> tuple[sp.csr_matrix, list[str], list[int]]:
    bundle = _vectorizer_bundle(vectorizer)
    feature_parts: list[sp.csr_matrix] = []
    feature_names: list[str] = []
    part_sizes: list[int] = []

    word_vec = bundle.get("word_vec") or bundle.get("word_vectorizer")
    if word_vec is not None:
        word_matrix = word_vec.transform([processed_text])
        feature_parts.append(word_matrix)
        names = [f"word:{name}" for name in word_vec.get_feature_names_out()]
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


def _transformer_predict(model: Any, raw_text: str) -> tuple[float, float]:
    probs = model.predict_proba([raw_text])
    ham_probability, spam_probability = float(probs[0, 0]), float(probs[0, 1])
    return spam_probability, ham_probability


def _ensemble_predict(model: Any, features: sp.csr_matrix, raw_text: str) -> tuple[float, float]:
    probs = model.predict_proba(features, [raw_text])
    ham_probability, spam_probability = float(probs[0, 0]), float(probs[0, 1])
    return spam_probability, ham_probability


def build_feature_matrix(
    vectorizer: Any, subject: str, body: str, *, subject_weight: int = DEFAULT_SUBJECT_WEIGHT,
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


def predict_email(
    *, model: Any, vectorizer: Any, sender: str, subject: str, body: str,
    whitelist_domains: Iterable[str] | None = None,
    trusted_service_domains: Iterable[str] | None = None,
    model_version: str = "unknown", spam_threshold: float = DEFAULT_SPAM_THRESHOLD,
) -> PredictionResult:
    sender_domain = extract_sender_domain(sender)
    subject = redact_subject(subject)
    body = redact_email_body(body)
    whitelisted = set(whitelist_domains or ())

    if sender_domain and sender_domain in whitelisted:
        return _base_result_payload(
            label="whitelisted", confidence=1.0,
            reason="Sender is in your trusted whitelist",
            analysis="Trusted sender matched your local whitelist.",
            model_version=model_version, sender_domain=sender_domain,
            rule_layer="whitelist", signals=["trusted sender domain"],
            explanations=["Whitelisted sender domain matched your local settings."],
        )

    if sender_domain and is_trusted_service_domain(sender_domain, trusted_service_domains):
        return _base_result_payload(
            label="Not Spam", confidence=0.97,
            reason="Sender recognised as a legitimate financial or service provider",
            analysis="Trusted service domain matched the curated built-in catalog.",
            model_version=model_version, sender_domain=sender_domain,
            rule_layer="trusted_service", signals=["trusted service domain"],
            explanations=["Trusted service domain matched the built-in service catalog."],
        )

    rule_assessment = assess_rule_based_spam(subject, body)
    if rule_assessment.is_spam:
        return _base_result_payload(
            label="Spam", confidence=rule_assessment.confidence,
            reason=rule_assessment.reason,
            analysis="Rule-based detection found multiple phishing-style signals before the ML model ran.",
            model_version=model_version, sender_domain=sender_domain,
            rule_layer="rules", signals=rule_assessment.signals,
            explanations=rule_assessment.signals[:4],
        )

    benign_assessment = assess_benign_email(subject, body)
    if benign_assessment.is_benign:
        return _base_result_payload(
            label="Not Spam", confidence=benign_assessment.confidence,
            reason=benign_assessment.reason, analysis=benign_assessment.analysis,
            model_version=model_version, sender_domain=sender_domain,
            rule_layer=benign_assessment.rule_layer, signals=benign_assessment.signals,
            explanations=benign_assessment.explanations,
        )

    features, feature_names = build_feature_matrix(vectorizer, subject, body)

    if _is_ensemble_model(model):
        raw_text = compose_email_text(subject, body)
        spam_probability, ham_probability = _ensemble_predict(model, features, raw_text)
    elif _is_transformer_model(model):
        raw_text = compose_email_text(subject, body)
        spam_probability, ham_probability = _transformer_predict(model, raw_text)
    else:
        spam_probability, ham_probability = _probabilities_from_model(model, features)

    if spam_probability >= spam_threshold:
        return _base_result_payload(
            label="Spam", confidence=round(spam_probability, 2),
            reason="Machine learning model detected suspicious patterns",
            analysis=f"AI analysis: {spam_probability:.1%} spam probability based on text and metadata.",
            model_version=model_version, sender_domain=sender_domain,
            rule_layer="ml", signals=rule_assessment.signals,
            explanations=explain_prediction(model, features, feature_names, "Spam"),
            spam_prob=round(spam_probability, 4), ham_prob=round(ham_probability, 4),
        )

    return _base_result_payload(
        label="Not Spam", confidence=round(ham_probability, 2),
        reason="Appears to be a legitimate email",
        analysis=f"AI analysis: {ham_probability:.1%} confidence that the message is legitimate.",
        model_version=model_version, sender_domain=sender_domain,
        rule_layer="ml", signals=rule_assessment.signals,
        explanations=explain_prediction(model, features, feature_names, "Not Spam"),
        spam_prob=round(spam_probability, 4), ham_prob=round(ham_probability, 4),
    )
