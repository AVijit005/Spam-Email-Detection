from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from app.core.constants import META_FEATURE_LABELS


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
    active_indices = features.indices
    active_values = features.data

    inner = model
    if hasattr(model, "classical_model"):
        inner = model.classical_model

    if hasattr(inner, "coef_"):
        coefficients = np.asarray(inner.coef_[0]).ravel()
    elif hasattr(inner, "feature_importances_"):
        coefficients = inner.feature_importances_
    else:
        return []

    contributions = active_values * coefficients[active_indices]
    if label == "Spam":
        candidate_pairs = [(feature_names[i], c) for i, c in zip(active_indices, contributions) if c > 0]
        candidate_pairs.sort(key=lambda item: item[1], reverse=True)
        return [_format_feature_explanation(name, True) for name, _ in candidate_pairs[:4]]
    candidate_pairs = [(feature_names[i], c) for i, c in zip(active_indices, contributions) if c < 0]
    candidate_pairs.sort(key=lambda item: item[1])
    return [_format_feature_explanation(name, False) for name, _ in candidate_pairs[:4]]
