from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class DummyModel:
    coef_ = __import__("numpy").array([[0.1] * 34])
    def predict_proba(self, features):
        import numpy as np
        return np.array([[0.12, 0.88]], dtype=np.float32)


class DummyVectorizer:
    def __init__(self):
        import numpy as np
        self.feature_names = np.array(["free", "win"], dtype=object)
    def transform(self, texts):
        import numpy as np
        import scipy.sparse as sp
        rows = [[1.0 if w in t.lower() else 0.0 for w in ["free", "win"]] for t in texts]
        return sp.csr_matrix(np.array(rows, dtype=np.float32))
    def get_feature_names_out(self):
        return self.feature_names


def _inject_state():
    import app.api.v1.predict as predict_mod
    import app.api.v1.health as health_mod
    import app.api.v1.feedback as feedback_mod
    import app.api.v1.retrain as retrain_mod

    dummy_model = DummyModel()
    dummy_vectorizer = DummyVectorizer()
    whitelist = {"company.com"}
    trusted = {"amazon.in", "google.com"}
    metadata = {"model_name": "DummyModel", "spam_threshold": 0.55}

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


class TestApiAuth(unittest.TestCase):
    def setUp(self):
        self._load_patch = mock.patch("app.main.load_resources")
        self._load_patch.start()
        from app.main import create_app
        self.app = create_app()
        _inject_state()

    def tearDown(self):
        self._load_patch.stop()

    def _client(self):
        from fastapi.testclient import TestClient
        return TestClient(self.app)

    def test_feedback_endpoint_requires_auth_when_key_configured(self):
        with mock.patch("app.core.auth.settings") as mock_auth_settings:
            mock_auth_settings.api_key = "secret"
            with self._client() as client:
                response = client.post("/v1/feedback", json={
                    "prediction_id": "test-001",
                    "user_label": "Spam",
                    "predicted_label": "Spam",
                })
        self.assertEqual(response.status_code, 401)

    def test_feedback_endpoint_allows_when_no_key_configured(self):
        with mock.patch("app.core.auth.settings") as mock_auth_settings:
            mock_auth_settings.api_key = ""
            with self._client() as client:
                response = client.post("/v1/feedback", json={
                    "prediction_id": "test-001",
                    "user_label": "Spam",
                    "predicted_label": "Spam",
                })
        self.assertEqual(response.status_code, 200)

    def test_retrain_endpoint_requires_auth_when_key_configured(self):
        with mock.patch("app.core.auth.settings") as mock_auth_settings:
            mock_auth_settings.api_key = "secret"
            with self._client() as client:
                response = client.post("/v1/retrain")
        self.assertEqual(response.status_code, 401)

    def test_retrain_endpoint_allows_when_no_key_configured(self):
        with mock.patch("app.api.v1.retrain.subprocess") as mocked_subprocess, \
             mock.patch("app.core.auth.settings") as mock_auth_settings:
            mock_auth_settings.api_key = ""
            mocked_subprocess.run.return_value = mock.Mock(returncode=0, stdout="ok", stderr="")
            with self._client() as client:
                response = client.post("/v1/retrain")
        self.assertEqual(response.status_code, 200)

    def test_predict_always_available_no_auth(self):
        with mock.patch("app.core.auth.settings") as mock_auth_settings:
            mock_auth_settings.api_key = "secret"
            with self._client() as client:
                response = client.post("/v1/predict", json={
                    "sender": "test@example.com",
                    "subject": "Hello",
                    "body": "World",
                })
        self.assertEqual(response.status_code, 200)
