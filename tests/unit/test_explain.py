from __future__ import annotations

import unittest

import numpy as np
import scipy.sparse as sp

from app.core.explain import explain_prediction, _format_feature_explanation


class DummyModelNoCoef:
    pass


class DummyModel:
    coef_ = np.array([[-0.8, 1.5, -0.3, 0.1, -0.1, 0.6, 0.4, 0.7, 0.0, 0.0, 0.2, 0.1, 0.0]])


class TestExplainPrediction(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel()
        self.feature_names = [
            "word:a", "word:b", "word:c", "word:d", "word:e",
            "meta:url_count", "meta:caps_ratio", "char:!!!",
            "meta:word_count", "meta:avg_word_length", "meta:urgency_hits",
            "meta:account_hits", "meta:mixed_token_hits",
        ]

    def _make_features(self, active_indices):
        data = np.ones(len(active_indices), dtype=np.float64)
        indptr = np.array([0, len(active_indices)], dtype=np.int32)
        indices = np.array(active_indices, dtype=np.int32)
        return sp.csr_matrix((data, indices, indptr), shape=(1, len(self.feature_names)))

    def test_returns_top_contributors_for_spam(self):
        features = self._make_features([1, 10])
        result = explain_prediction(self.model, features, self.feature_names, "Spam")
        self.assertGreater(len(result), 0)
        result_str = " ".join(result)
        self.assertIn("Suspicious", result_str)

    def test_returns_top_contributors_for_not_spam(self):
        features = self._make_features([0, 2])
        result = explain_prediction(self.model, features, self.feature_names, "Not Spam")
        self.assertGreater(len(result), 0)
        result_str = " ".join(result)
        self.assertIn("Legitimate", result_str)

    def test_spam_explanations_use_positive_coefficients(self):
        features = self._make_features([1, 2, 3])
        result = explain_prediction(self.model, features, self.feature_names, "Spam")
        self.assertGreater(len(result), 0)
        self.assertIn("Suspicious token", result[0])

    def test_not_spam_explanations_use_negative_coefficients(self):
        features = self._make_features([0, 2, 4])
        result = explain_prediction(self.model, features, self.feature_names, "Not Spam")
        self.assertGreater(len(result), 0)
        self.assertIn("Legitimate token", result[0])

    def test_returns_empty_when_no_coef(self):
        model = DummyModelNoCoef()
        features = self._make_features([0])
        result = explain_prediction(model, features, self.feature_names, "Spam")
        self.assertEqual(result, [])

    def test_meta_feature_uses_label(self):
        features = self._make_features([5, 6])
        result = explain_prediction(self.model, features, self.feature_names, "Spam")
        result_str = " ".join(result)
        self.assertIn("Suspicious signal", result_str)
        self.assertIn("contains links", result_str)

    def test_char_feature_uses_pattern_label(self):
        features = self._make_features([7])
        result = explain_prediction(self.model, features, self.feature_names, "Spam")
        result_str = " ".join(result)
        self.assertIn("pattern", result_str)

    def test_unknown_feature_prefix_returns_as_is(self):
        result = _format_feature_explanation("raw_feature", True)
        self.assertEqual(result, "raw_feature")
