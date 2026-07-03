from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pydantic_settings import BaseSettings, SettingsConfigDict


class TestSettings(unittest.TestCase):
    def test_settings_loads_env_vars_with_spam_prefix(self):
        env = {
            "SPAM_API_HOST": "192.168.1.1",
            "SPAM_API_PORT": "9090",
            "SPAM_LOG_LEVEL": "debug",
            "SPAM_FEEDBACK_BACKEND": "file",
            "SPAM_RETRAIN_TIMEOUT_SECONDS": "600",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            from app.config import Settings
            instance = Settings()
        self.assertEqual(instance.api_host, "192.168.1.1")
        self.assertEqual(instance.api_port, 9090)
        self.assertEqual(instance.log_level, "debug")
        self.assertEqual(instance.feedback_backend, "file")
        self.assertEqual(instance.retrain_timeout_seconds, 600)

    def test_settings_defaults_when_no_env(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            from app.config import Settings
            instance = Settings()
        self.assertEqual(instance.api_host, "0.0.0.0")
        self.assertEqual(instance.api_port, 8000)
        self.assertEqual(instance.log_level, "info")
        self.assertIsInstance(instance.feedback_log_path, Path)
        self.assertTrue(instance.bootstrap_model_if_missing)
        self.assertFalse(instance.train_on_start)
        self.assertEqual(instance.retrain_timeout_seconds, 900)
        self.assertEqual(instance.spam_threshold, 0.55)
        self.assertEqual(instance.environment, "development")

    def test_boolean_env_vars_parsed(self):
        with mock.patch.dict(os.environ, {
            "SPAM_TRAIN_ON_START": "true",
            "SPAM_BOOTSTRAP_MODEL_IF_MISSING": "false",
        }, clear=True):
            from app.config import Settings
            instance = Settings()
        self.assertTrue(instance.train_on_start)
        self.assertFalse(instance.bootstrap_model_if_missing)

    def test_empty_api_key_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            from app.config import Settings
            instance = Settings()
        self.assertEqual(instance.api_key, "")
        self.assertEqual(instance.jwt_secret_key, "")

    def test_api_key_read_from_env(self):
        with mock.patch.dict(os.environ, {"SPAM_API_KEY": "secret-abc"}, clear=True):
            from app.config import Settings
            instance = Settings()
        self.assertEqual(instance.api_key, "secret-abc")
