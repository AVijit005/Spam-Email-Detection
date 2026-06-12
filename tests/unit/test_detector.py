from __future__ import annotations

import unittest

import numpy as np
import scipy.sparse as sp

from app.core.detector import (
    _base_result_payload,
    _probabilities_from_model,
    _vectorizer_bundle,
    build_feature_matrix,
    predict_email,
)


class DummyModel:
    def predict_proba(self, features):
        return np.array([[0.2, 0.8]], dtype=np.float32)


class DummyModelDecisionFn:
    def decision_function(self, features):
        return np.array([1.5])


class DummyVectorizer:
    def __init__(self):
        self.feature_names = np.array(["free", "win"], dtype=object)

    def transform(self, texts):
        rows = [[1.0 if w in t.lower() else 0.0 for w in ["free", "win"]] for t in texts]
        return sp.csr_matrix(np.array(rows, dtype=np.float32))

    def get_feature_names_out(self):
        return self.feature_names


class TestProbabilitiesFromModel(unittest.TestCase):
    def test_predict_proba_path(self):
        model = DummyModel()
        features = sp.csr_matrix(np.array([[0.1, 0.5]], dtype=np.float32))
        spam, ham = _probabilities_from_model(model, features)
        self.assertAlmostEqual(spam, 0.8)
        self.assertAlmostEqual(ham, 0.2)

    def test_decision_function_path(self):
        model = DummyModelDecisionFn()
        features = sp.csr_matrix(np.array([[0.1, 0.5]], dtype=np.float32))
        spam, ham = _probabilities_from_model(model, features)
        self.assertGreater(spam, 0.5)
        self.assertLess(ham, 0.5)
        self.assertAlmostEqual(spam + ham, 1.0)


class TestVectorizerBundle(unittest.TestCase):
    def test_dict_passthrough(self):
        bundle = _vectorizer_bundle({"version": 2, "custom": True})
        self.assertEqual(bundle["version"], 2)

    def test_wraps_vectorizer_object(self):
        vec = DummyVectorizer()
        bundle = _vectorizer_bundle(vec)
        self.assertEqual(bundle["word_vec"], vec)
        self.assertIsNone(bundle["char_vectorizer"])


class TestBuildFeatureMatrix(unittest.TestCase):
    def test_returns_csr_matrix_and_names(self):
        vec = DummyVectorizer()
        matrix, names = build_feature_matrix(vec, "Free win", "Click here")
        self.assertTrue(sp.issparse(matrix))
        self.assertGreater(len(names), 0)
        self.assertIn("word:free", names)

    def test_with_dict_vectorizer(self):
        vec_dict = {"version": 1, "word_vectorizer": DummyVectorizer(), "char_vectorizer": None,
                    "meta_feature_names": ["url_count", "caps_ratio"]}
        matrix, names = build_feature_matrix(vec_dict, "Test", "Body")
        self.assertTrue(sp.issparse(matrix))
        self.assertIn("meta:url_count", names)


class TestBaseResultPayload(unittest.TestCase):
    def test_returns_prediction_result_with_all_fields(self):
        result = _base_result_payload(
            label="Spam", confidence=0.95, reason="Reason", analysis="Analysis",
            model_version="v1", sender_domain="example.com", rule_layer="rules",
            signals=["s1"], explanations=["e1"],
            spam_prob=0.95, ham_prob=0.05,
        )
        self.assertEqual(result.label, "Spam")
        self.assertEqual(result.confidence, 0.95)
        self.assertTrue(result.prediction_id)
        self.assertTrue(result.evaluated_at_utc)

    def test_none_probability_omitted_from_payload(self):
        result = _base_result_payload(
            label="Not Spam", confidence=0.90, reason="R", analysis="A",
            model_version="v1", sender_domain="", rule_layer="ml",
            signals=[], explanations=[],
            spam_prob=None, ham_prob=None,
        )
        payload = result.to_payload()
        self.assertNotIn("spam_prob", payload)
        self.assertNotIn("ham_prob", payload)


class TestPredictEmail(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel()
        self.vectorizer = DummyVectorizer()

    def test_whitelisted_sender_returns_whitelisted(self):
        result = predict_email(
            model=self.model, vectorizer=self.vectorizer,
            sender="boss@company.com", subject="Review", body="Report",
            whitelist_domains={"company.com"},
            model_version="v1",
        )
        self.assertEqual(result.label, "whitelisted")
        self.assertEqual(result.confidence, 1.0)
        self.assertEqual(result.rule_layer, "whitelist")

    def test_trusted_service_returns_not_spam(self):
        result = predict_email(
            model=self.model, vectorizer=self.vectorizer,
            sender="shipping@amazon.in", subject="Order", body="Shipped",
            trusted_service_domains={"amazon.in"},
            model_version="v1",
        )
        self.assertEqual(result.label, "Not Spam")
        self.assertEqual(result.rule_layer, "trusted_service")

    def test_whitelist_takes_priority_over_trusted(self):
        result = predict_email(
            model=self.model, vectorizer=self.vectorizer,
            sender="shipping@amazon.in", subject="Order", body="Shipped",
            whitelist_domains={"amazon.in"},
            trusted_service_domains={"amazon.in"},
            model_version="v1",
        )
        self.assertEqual(result.rule_layer, "whitelist")

    def test_ml_layer_spam_label(self):
        result = predict_email(
            model=self.model, vectorizer=self.vectorizer,
            sender="phish@bad.com", subject="Free win now", body="Claim your prize",
            model_version="v1", spam_threshold=0.5,
        )
        self.assertEqual(result.label, "Spam")
        self.assertEqual(result.rule_layer, "ml")

    def test_ml_layer_not_spam_label(self):
        class NotSpamModel:
            def predict_proba(self, features):
                return np.array([[0.9, 0.1]], dtype=np.float32)

        result = predict_email(
            model=NotSpamModel(), vectorizer=self.vectorizer,
            sender="friend@example.com", subject="Hello",
            body="Just checking in to see how things are going.",
            model_version="v1",
        )
        self.assertEqual(result.label, "Not Spam")
        self.assertEqual(result.rule_layer, "ml")

    def test_none_whitelist_works(self):
        result = predict_email(
            model=self.model, vectorizer=self.vectorizer,
            sender="phish@bad.com", subject="Free win", body="Click",
            whitelist_domains=None,
            trusted_service_domains=None,
            model_version="v1",
        )
        self.assertIn(result.label, ("Spam", "Not Spam"))

    def test_rule_based_spam_detected(self):
        result = predict_email(
            model=self.model, vectorizer=self.vectorizer,
            sender="fraud@unknown.biz", subject="URGENT: account suspended!!!",
            body="Click here to verify your account immediately wire transfer needed",
            model_version="v1",
        )
        self.assertEqual(result.rule_layer, "rules")

    def test_benign_conversation_detected(self):
        result = predict_email(
            model=self.model, vectorizer=self.vectorizer,
            sender="colleague@example.com", subject="Lunch today",
            body="Are we still meeting at 1pm near the office?",
            model_version="v1",
        )
        self.assertEqual(result.rule_layer, "benign_context")
        self.assertEqual(result.label, "Not Spam")
