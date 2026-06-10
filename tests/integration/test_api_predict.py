from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _clear_model_state():
    import app.api.v1.predict as predict_mod
    predict_mod.model = None
    predict_mod.vectorizer = None


class TestApiPredictEdgeCases(unittest.TestCase):
    def setUp(self):
        self._load_patch = mock.patch("app.main.load_resources")
        self._load_patch.start()
        _clear_model_state()
        from app.main import create_app
        self.app = create_app()

    def tearDown(self):
        self._load_patch.stop()

    def _client(self):
        from fastapi.testclient import TestClient
        return TestClient(self.app)

    def test_predict_returns_500_when_model_not_loaded(self):
        with self._client() as client:
            response = client.post("/v1/predict", json={
                "sender": "test@example.com", "subject": "Hello", "body": "World",
            })
        self.assertEqual(response.status_code, 500)
        self.assertIn("Model not loaded", response.json()["detail"])

    def test_batch_predict_returns_500_when_model_not_loaded(self):
        with self._client() as client:
            response = client.post("/v1/predict/batch", json={
                "emails": [{"sender": "a@b.com", "subject": "s", "body": "b"}],
            })
        self.assertEqual(response.status_code, 500)
        self.assertIn("Model not loaded", response.json()["detail"])
