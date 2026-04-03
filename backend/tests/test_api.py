from __future__ import annotations

import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from fastapi.testclient import TestClient


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import app as backend_app


class DummyVectorizer:
    def __init__(self):
        self.feature_names = np.array(["verify", "meeting"], dtype=object)

    def transform(self, texts):
        rows = []
        for text in texts:
            lowered = text.lower()
            rows.append([
                1.0 if "verify" in lowered else 0.0,
                1.0 if "meeting" in lowered else 0.0
            ])
        return sp.csr_matrix(np.array(rows, dtype=np.float32))

    def get_feature_names_out(self):
        return self.feature_names


class DummyModel:
    coef_ = np.array([[1.5, -1.2, 0.4, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.4, 0.5, 0.2, 0.0, 0.1, 0.0, 0.0]])

    def predict_proba(self, features):
        dense = features.toarray()
        scores = []
        for row in dense:
            verify = row[0]
            meeting = row[1]
            spam_probability = 0.88 if verify > 0 else 0.12
            if meeting > 0:
                spam_probability = 0.08
            scores.append([1.0 - spam_probability, spam_probability])
        return np.array(scores, dtype=np.float32)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        backend_app.load_resources = lambda: None
        backend_app.model = DummyModel()
        backend_app.vectorizer = DummyVectorizer()
        backend_app.user_whitelist_domains = {"company.com"}
        backend_app.trusted_domain_catalog = {"amazon.in", "google.com"}
        backend_app.model_metadata = {
            "model_name": "DummyModel",
            "spam_threshold": 0.55,
            "trained_at_utc": "2026-04-03T00:00:00+00:00",
        }
        backend_app.FEEDBACK_LOG_PATH = Path(self.temp_dir.name) / "feedback.jsonl"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_health_endpoint_reports_state(self):
        with TestClient(backend_app.app) as client:
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["model_version"], "DummyModel")
        self.assertTrue(payload["model_loaded"])
        self.assertEqual(payload["feedback_backend"], "file")
        self.assertEqual(payload["user_whitelist_count"], 1)
        self.assertEqual(payload["trusted_domain_catalog_count"], 2)
        self.assertEqual(payload["feedback_count"], 0)

    def test_predict_endpoint_respects_whitelist(self):
        with TestClient(backend_app.app) as client:
            response = client.post(
                "/predict",
                json={
                    "sender": "boss@company.com",
                    "subject": "Weekly review",
                    "body": "Please send the report."
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["label"], "whitelisted")
        self.assertTrue(payload["prediction_id"])

    def test_batch_prediction_endpoint_returns_multiple_results(self):
        with TestClient(backend_app.app) as client:
            response = client.post(
                "/predict/batch",
                json={
                    "emails": [
                        {
                            "sender": "boss@company.com",
                            "subject": "Weekly review",
                            "body": "Please send the report."
                        },
                        {
                            "sender": "fraud@unknown.biz",
                            "subject": "Please verify account",
                            "body": "Click here to verify your account."
                        }
                    ]
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload), 2)
        self.assertEqual(payload[0]["label"], "whitelisted")
        self.assertEqual(payload[1]["label"], "Spam")
        self.assertTrue(payload[1]["explanations"])

    def test_domain_catalog_is_not_treated_as_whitelist(self):
        with TestClient(backend_app.app) as client:
            response = client.post(
                "/predict",
                json={
                    "sender": "shipping@amazon.in",
                    "subject": "Your order has shipped",
                    "body": "Package arriving tomorrow."
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["label"], "Not Spam")
        self.assertEqual(payload["rule_layer"], "trusted_service")

    def test_feedback_endpoint_stores_user_label_and_updates_summary(self):
        with TestClient(backend_app.app) as client:
            prediction = client.post(
                "/predict",
                json={
                    "sender": "fraud@unknown.biz",
                    "subject": "Please verify account",
                    "body": "Click here to verify your account."
                },
            ).json()

            feedback_response = client.post(
                "/feedback",
                json={
                    "prediction_id": prediction["prediction_id"],
                    "sender": "fraud@unknown.biz",
                    "subject": "Please verify account",
                    "body": "Click here to verify your account.",
                    "predicted_label": prediction["label"],
                    "predicted_confidence": prediction["confidence"],
                    "user_label": "Spam",
                    "source": "unit_test"
                },
            )
            summary_response = client.get("/feedback/summary")

        self.assertEqual(feedback_response.status_code, 200)
        self.assertEqual(summary_response.status_code, 200)
        self.assertEqual(summary_response.json()["feedback_count"], 1)
        self.assertEqual(summary_response.json()["verdict_counts"]["correct"], 1)

    def test_retrain_endpoint_runs_training_and_reloads_metadata(self):
        def fake_load_resources():
            backend_app.model = DummyModel()
            backend_app.vectorizer = DummyVectorizer()
            backend_app.model_metadata = {
                "model_name": "RetrainedModel",
                "trained_at_utc": "2026-04-03T12:00:00+00:00",
                "spam_threshold": 0.55,
                "dataset_rows": 2608,
                "selected_metrics": {"spam_f1": 0.93},
                "feedback_training": {
                    "feedback_rows_used": 3,
                    "last_feedback_at_utc": "2026-04-03T11:55:00+00:00",
                },
            }

        backend_app.load_resources = fake_load_resources

        with mock.patch.object(
            backend_app.subprocess,
            "run",
            return_value=mock.Mock(returncode=0, stdout="ok", stderr=""),
        ) as mocked_run:
            with TestClient(backend_app.app) as client:
                response = client.post("/retrain")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["model_version"], "RetrainedModel")
        self.assertEqual(payload["feedback_backend"], "file")
        self.assertEqual(payload["feedback_rows_used"], 3)
        self.assertEqual(payload["dataset_rows"], 2608)
        self.assertEqual(payload["spam_f1"], 0.93)
        mocked_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
