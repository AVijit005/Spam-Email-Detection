from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _inject_state():
    import app.api.v1.predict as predict_mod
    import app.api.v1.health as health_mod
    import app.api.v1.feedback as feedback_mod
    import app.api.v1.retrain as retrain_mod
    import numpy as np

    class DummyModel:
        coef_ = np.array([[0.1] * 18])
        def predict_proba(self, f):
            return np.array([[0.1, 0.9]])
    class DummyVectorizer:
        def __init__(self):
            self.feature_names = np.array(["free"], dtype=object)
        def transform(self, t):
            import scipy.sparse as sp
            return sp.csr_matrix(np.array([[0.0]]))
        def get_feature_names_out(self):
            return self.feature_names

    dummy_model = DummyModel()
    dummy_vectorizer = DummyVectorizer()
    metadata = {"model_name": "DummyModel", "spam_threshold": 0.55}

    predict_mod.model = dummy_model
    predict_mod.vectorizer = dummy_vectorizer
    predict_mod.user_whitelist_domains = set()
    predict_mod.trusted_domain_catalog = set()
    predict_mod.model_metadata = metadata

    health_mod.model = dummy_model
    health_mod.vectorizer = dummy_vectorizer
    health_mod.user_whitelist_domains = set()
    health_mod.trusted_domain_catalog = set()
    health_mod.model_metadata = metadata

    feedback_mod.model_metadata = metadata
    retrain_mod.model_metadata = metadata


class TestApiRetrain(unittest.TestCase):
    def setUp(self):
        self._load_patch = mock.patch("app.main.load_resources")
        self._load_patch.start()
        from app.main import create_app
        self.app = create_app()
        _inject_state()

    def tearDown(self):
        self._load_patch.stop()
        import app.api.v1.retrain as retrain_mod
        try:
            retrain_mod.RETRAIN_LOCK.release()
        except RuntimeError:
            pass

    def _client(self):
        from fastapi.testclient import TestClient
        return TestClient(self.app)

    def test_retrain_returns_409_when_already_in_progress(self):
        import app.api.v1.retrain as retrain_mod
        retrain_mod.RETRAIN_LOCK.acquire(blocking=False)

        with mock.patch("app.core.auth.settings") as mock_auth_settings, \
             mock.patch("app.api.v1.retrain.subprocess") as mocked_subprocess:
            mock_auth_settings.api_key = ""
            mocked_subprocess.run.return_value = mock.Mock(returncode=0, stdout="ok", stderr="")
            with self._client() as client:
                response = client.post("/v1/retrain")

        self.assertEqual(response.status_code, 409)
        self.assertIn("already in progress", response.json()["detail"].lower())

    def test_retrain_succeeds_when_not_locked(self):
        def _fake_load():
            import app.api.v1.predict as predict_mod
            import app.api.v1.health as health_mod
            import app.api.v1.feedback as feedback_mod
            import app.api.v1.retrain as retrain_mod
            import numpy as np

            class D2:
                coef_ = np.array([[0.1] * 18])
                def predict_proba(self, f):
                    return np.array([[0.1, 0.9]])
            class V2:
                def __init__(self):
                    self.feature_names = np.array(["free"], dtype=object)
                def transform(self, t):
                    import scipy.sparse as sp
                    return sp.csr_matrix(np.array([[0.0]]))
                def get_feature_names_out(self):
                    return self.feature_names

            metadata = {
                "model_name": "Retrained", "trained_at_utc": "2026-04-03T12:00:00+00:00",
                "spam_threshold": 0.55, "dataset_rows": 2608,
                "selected_metrics": {"spam_f1": 0.93},
                "feedback_training": {"feedback_rows_used": 3, "last_feedback_at_utc": "2026-04-03T11:55:00+00:00"},
            }
            predict_mod.model = D2()
            predict_mod.vectorizer = V2()
            predict_mod.model_metadata = metadata
            health_mod.model = D2()
            health_mod.vectorizer = V2()
            health_mod.model_metadata = metadata
            feedback_mod.model_metadata = metadata
            retrain_mod.model_metadata = metadata

        with mock.patch("app.core.auth.settings") as mock_auth_settings, \
             mock.patch("app.api.v1.retrain.subprocess") as mocked_subprocess, \
             mock.patch("app.main.load_resources", side_effect=_fake_load):
            mock_auth_settings.api_key = ""
            mocked_subprocess.run.return_value = mock.Mock(returncode=0, stdout="ok", stderr="")
            with self._client() as client:
                response = client.post("/v1/retrain")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_version"], "Retrained")
