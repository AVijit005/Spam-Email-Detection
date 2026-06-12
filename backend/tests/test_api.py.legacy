from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class DummyVectorizer:
    def __init__(self):
        self.feature_names = np.array(["verify", "meeting"], dtype=object)

    def transform(self, texts):
        rows = []
        for text in texts:
            lowered = text.lower()
            rows.append([
                1.0 if "verify" in lowered else 0.0,
                1.0 if "meeting" in lowered else 0.0,
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


def _inject_state():
    import app.api.v1.predict as predict_mod
    import app.api.v1.health as health_mod
    import app.api.v1.feedback as feedback_mod
    import app.api.v1.retrain as retrain_mod

    dummy_model = DummyModel()
    dummy_vectorizer = DummyVectorizer()
    whitelist = {"company.com"}
    trusted = {"amazon.in", "google.com"}
    metadata = {
        "model_name": "DummyModel",
        "spam_threshold": 0.55,
        "trained_at_utc": "2026-04-03T00:00:00+00:00",
    }

    predict_mod.model = dummy_model
    predict_mod.vectorizer = dummy_vectorizer
    predict_mod.user_whitelist_domains = whitelist
    predict_mod.trusted_domain_catalog = trusted
    predict_mod.model_metadata = metadata

    health_mod.model = dummy_model
    health_mod.vectorizer = dummy_vectorizer
    health_mod.user_whitelist_domains = whitelist
    health_mod.trusted_domain_catalog = trusted
    health_mod.model_metadata = metadata

    feedback_mod.model_metadata = metadata
    retrain_mod.model_metadata = metadata


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        from app.config import settings
        self._original_feedback_log_path = settings.feedback_log_path
        settings.feedback_log_path = Path(self.temp_dir.name) / "feedback.jsonl"

        self._load_resources_patch = mock.patch("app.main.load_resources")
        self._load_resources_patch.start()

        from app.main import create_app
        self.app = create_app()

        _inject_state()

    def tearDown(self):
        from app.config import settings
        settings.feedback_log_path = self._original_feedback_log_path
        self._load_resources_patch.stop()
        self.temp_dir.cleanup()

    def _client(self):
        from fastapi.testclient import TestClient
        return TestClient(self.app)

    def test_health_endpoint_reports_state(self):
        with self._client() as client:
            response = client.get("/v1/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["model_version"], "DummyModel")
        self.assertTrue(payload["model_loaded"])
        self.assertTrue(payload["vectorizer_loaded"])
        self.assertEqual(payload["feedback_backend"], "file")
        self.assertEqual(payload["user_whitelist_count"], 1)
        self.assertEqual(payload["trusted_domain_catalog_count"], 2)
        self.assertEqual(payload["spam_threshold"], 0.55)

    def test_predict_endpoint_respects_whitelist(self):
        with self._client() as client:
            response = client.post(
                "/v1/predict",
                json={
                    "sender": "boss@company.com",
                    "subject": "Weekly review",
                    "body": "Please send the report.",
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["label"], "whitelisted")
        self.assertEqual(payload["confidence"], 1.0)
        self.assertTrue(payload["prediction_id"])

    def test_batch_prediction_endpoint_returns_multiple_results(self):
        with self._client() as client:
            response = client.post(
                "/v1/predict/batch",
                json={
                    "emails": [
                        {
                            "sender": "boss@company.com",
                            "subject": "Weekly review",
                            "body": "Please send the report.",
                        },
                        {
                            "sender": "fraud@unknown.biz",
                            "subject": "Please verify account",
                            "body": "Click here to verify your account.",
                        },
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
        with self._client() as client:
            response = client.post(
                "/v1/predict",
                json={
                    "sender": "shipping@amazon.in",
                    "subject": "Your order has shipped",
                    "body": "Package arriving tomorrow.",
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["label"], "Not Spam")
        self.assertEqual(payload["rule_layer"], "trusted_service")

    def test_feedback_endpoint_stores_user_label_and_updates_summary(self):
        with self._client() as client:
            prediction = client.post(
                "/v1/predict",
                json={
                    "sender": "fraud@unknown.biz",
                    "subject": "Please verify account",
                    "body": "Click here to verify your account.",
                },
            ).json()

            feedback_response = client.post(
                "/v1/feedback",
                json={
                    "prediction_id": prediction["prediction_id"],
                    "sender": "fraud@unknown.biz",
                    "subject": "Please verify account",
                    "body": "Click here to verify your account.",
                    "predicted_label": prediction["label"],
                    "predicted_confidence": prediction["confidence"],
                    "user_label": "Spam",
                    "source": "unit_test",
                },
            )
            summary_response = client.get("/v1/feedback/summary")

        self.assertEqual(feedback_response.status_code, 200)
        self.assertEqual(summary_response.status_code, 200)
        self.assertEqual(summary_response.json()["feedback_count"], 1)
        self.assertEqual(summary_response.json()["verdict_counts"]["correct"], 1)

    def test_retrain_endpoint_runs_training_and_reloads_metadata(self):
        import app.api.v1.predict as predict_mod
        import app.api.v1.health as health_mod
        import app.api.v1.feedback as feedback_mod
        import app.api.v1.retrain as retrain_mod

        new_metadata = {
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

        def _fake_load():
            predict_mod.model = DummyModel()
            predict_mod.vectorizer = DummyVectorizer()
            predict_mod.model_metadata = new_metadata
            health_mod.model = DummyModel()
            health_mod.vectorizer = DummyVectorizer()
            health_mod.model_metadata = new_metadata
            feedback_mod.model_metadata = new_metadata
            retrain_mod.model_metadata = new_metadata

        with mock.patch("app.api.v1.retrain.subprocess") as mocked_subprocess, \
             mock.patch("app.main.load_resources", side_effect=_fake_load):
            mocked_subprocess.run.return_value = mock.Mock(
                returncode=0, stdout="ok", stderr=""
            )
            with self._client() as client:
                response = client.post("/v1/retrain")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["model_version"], "RetrainedModel")
        self.assertEqual(payload["feedback_backend"], "file")
        self.assertEqual(payload["feedback_rows_used"], 3)
        self.assertEqual(payload["dataset_rows"], 2608)
        self.assertEqual(payload["spam_f1"], 0.93)
        mocked_subprocess.run.assert_called_once()

    def test_predict_with_empty_fields_returns_result(self):
        with self._client() as client:
            response = client.post(
                "/v1/predict",
                json={"sender": "", "subject": "", "body": ""},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("label", payload)
        self.assertIn("confidence", payload)
        self.assertIn("prediction_id", payload)

    def test_predict_with_long_input_still_works(self):
        with self._client() as client:
            response = client.post(
                "/v1/predict",
                json={
                    "sender": "someone@example.com",
                    "subject": "A" * 998,
                    "body": "B" * 50000,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("label", payload)

    def test_feedback_with_invalid_label_returns_400(self):
        with self._client() as client:
            response = client.post(
                "/v1/feedback",
                json={
                    "prediction_id": "test123",
                    "sender": "spam@example.com",
                    "subject": "Test",
                    "body": "Test body",
                    "predicted_label": "Spam",
                    "user_label": "INVALID",
                    "source": "unit_test",
                },
            )

        self.assertEqual(response.status_code, 400)

    def test_feedback_summary_returns_zero_when_no_entries(self):
        with self._client() as client:
            response = client.get("/v1/feedback/summary")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["feedback_count"], 0)

    def test_batch_predict_with_empty_emails_returns_empty_list(self):
        with self._client() as client:
            response = client.post(
                "/v1/predict/batch",
                json={"emails": []},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload), 0)


if __name__ == "__main__":
    unittest.main()
