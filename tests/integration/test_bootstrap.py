from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.ml.registry import save_model


class DummyModel:
    pass


class DummyVectorizer:
    pass


class TestBootstrapIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.model_path = self.tmp_path / "spam_model.pkl"
        self.vec_path = self.tmp_path / "vectorizer.pkl"
        self.meta_path = self.tmp_path / "model_metadata.json"
        self.whitelist_path = self.tmp_path / "whitelist.csv"
        self.trusted_path = self.tmp_path / "trusted_domains.csv"
        self.feedback_path = self.tmp_path / "feedback.jsonl"

        save_model(
            DummyModel(), DummyVectorizer(),
            {"model_name": "BootstrapTest", "trained_at_utc": "2026-01-01T00:00:00+00:00", "spam_threshold": 0.55},
            self.model_path, self.vec_path, self.meta_path,
        )
        self.whitelist_path.write_text("company.com\n", encoding="utf-8")
        self.trusted_path.write_text("google.com\n", encoding="utf-8")

        from app.main import settings as main_settings
        self._settings_patches = [
            mock.patch.object(main_settings, "model_path", self.model_path),
            mock.patch.object(main_settings, "vectorizer_path", self.vec_path),
            mock.patch.object(main_settings, "metadata_path", self.meta_path),
            mock.patch.object(main_settings, "whitelist_path", self.whitelist_path),
            mock.patch.object(main_settings, "trusted_domains_path", self.trusted_path),
            mock.patch.object(main_settings, "feedback_log_path", self.feedback_path),
            mock.patch.object(main_settings, "spam_threshold", 0.55),
            mock.patch.object(main_settings, "allow_origin_regex", r"^http://localhost(:\d+)?$"),
        ]
        for p in self._settings_patches:
            p.start()

    def tearDown(self):
        for p in self._settings_patches:
            p.stop()
        self.tmp.cleanup()

    def test_create_app_starts_and_serves_health(self):
        from app.main import create_app
        app = create_app()

        from fastapi.testclient import TestClient
        with TestClient(app) as client:
            response = client.get("/v1/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["model_version"], "BootstrapTest")
        self.assertTrue(payload["model_loaded"])
        self.assertTrue(payload["vectorizer_loaded"])
        self.assertEqual(payload["user_whitelist_count"], 1)
        self.assertEqual(payload["trusted_domain_catalog_count"], 1)

    def test_load_resources_sets_module_state(self):
        import app.api.v1.health as health_mod
        import app.api.v1.predict as predict_mod
        import app.api.v1.feedback as feedback_mod
        import app.api.v1.retrain as retrain_mod

        health_mod.model = None
        health_mod.vectorizer = None

        from app.main import load_resources
        load_resources()

        self.assertIsNotNone(health_mod.model)
        self.assertIsNotNone(health_mod.vectorizer)
        self.assertIsNotNone(predict_mod.model)
        self.assertEqual(health_mod.model_metadata.get("model_name"), "BootstrapTest")
        self.assertEqual(feedback_mod.model_metadata.get("model_name"), "BootstrapTest")
        self.assertEqual(retrain_mod.model_metadata.get("model_name"), "BootstrapTest")
        self.assertIn("company.com", health_mod.user_whitelist_domains)
        self.assertIn("google.com", health_mod.trusted_domain_catalog)

    def test_load_resources_handles_missing_model_gracefully(self):
        import app.api.v1.health as health_mod
        self.model_path.unlink()
        self.vec_path.unlink()

        from app.main import load_resources
        load_resources()

        self.assertIsNone(health_mod.model)
