from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from app.core.constants import BUSINESS_KEYWORDS, CONVERSATIONAL_KEYWORDS, PROMOTIONAL_KEYWORDS
from app.core.features import (
    _count_keyword_hits, _indicator_signals, _meta_feature_map,
    compose_email_text, matched_spam_phrases,
)


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


def is_trusted_service_domain(sender_domain: str, trusted_service_domains: Iterable[str] | None = None) -> bool:
    catalog = set(trusted_service_domains or ())
    return any(sender_domain == domain or sender_domain.endswith(f".{domain}") for domain in catalog)


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
            is_spam=True, confidence=round(confidence, 2),
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

    has_high_risk_indicator = any((
        feature_map["url_count"] >= 1, feature_map["money_count"] >= 1,
        feature_map["phone_count"] >= 1, feature_map["account_hits"] >= 1,
        feature_map["mixed_token_hits"] >= 2, feature_map["call_to_action_hits"] >= 2,
        len(phrase_hits) >= 1,
    ))

    if not has_high_risk_indicator and conversational_hits >= 2 and feature_map["word_count"] <= 40:
        signals = ["conversation-style wording"]
        if business_hits >= 1:
            signals.append("routine work context")
        return BenignAssessment(
            is_benign=True, confidence=0.82,
            reason="Looks like a routine personal or workplace conversation",
            analysis="Benign-context detection found conversational wording without phishing indicators.",
            rule_layer="benign_context", signals=signals,
            explanations=["Conversation-style wording matched a low-risk pattern.",
                          "No links, security prompts, or high-risk phishing markers were found."],
        )

    if (not has_high_risk_indicator and promotional_hits >= 2
            and feature_map["exclamation_count"] <= 1 and feature_map["caps_ratio"] < 0.2):
        return BenignAssessment(
            is_benign=True, confidence=0.76,
            reason="Looks like a low-risk promotional message, not phishing",
            analysis="Benign promotional detection found retail language without phishing indicators.",
            rule_layer="benign_promo", signals=["retail or promotional wording"],
            explanations=["Promotional wording was present, but there were no links, account prompts, or credential requests."],
        )

    return BenignAssessment(is_benign=False, confidence=0.0, reason="", analysis="", rule_layer="ml")
