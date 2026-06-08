from app.core.detector import PredictionResult, build_feature_matrix, predict_email
from app.core.domain import (
    extract_sender_domain, load_domain_catalog, load_trusted_domains,
    load_user_whitelist, normalize_domain,
)
from app.core.features import compose_email_text, extract_meta_features
from app.core.rules import (
    BenignAssessment, RuleAssessment, assess_benign_email,
    assess_rule_based_spam, is_trusted_service_domain,
)
from app.core.text import preprocess_text

__all__ = [
    "BenignAssessment", "PredictionResult", "RuleAssessment",
    "assess_benign_email", "assess_rule_based_spam", "build_feature_matrix",
    "compose_email_text", "extract_meta_features", "extract_sender_domain",
    "is_trusted_service_domain", "load_domain_catalog", "load_trusted_domains",
    "load_user_whitelist", "normalize_domain", "predict_email", "preprocess_text",
]
