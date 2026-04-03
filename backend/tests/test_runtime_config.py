from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from runtime_config import load_runtime_config


class RuntimeConfigTests(unittest.TestCase):
    def test_default_runtime_config(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            config = load_runtime_config()

        self.assertEqual(config.api_host, "127.0.0.1")
        self.assertEqual(config.api_port, 8000)
        self.assertEqual(config.log_level, "info")
        self.assertTrue(config.bootstrap_model_if_missing)
        self.assertFalse(config.train_on_start)

    def test_runtime_config_reads_environment_overrides(self):
        with mock.patch.dict(
            os.environ,
            {
                "SPAM_API_HOST": "0.0.0.0",
                "SPAM_API_PORT": "8081",
                "SPAM_LOG_LEVEL": "warning",
                "SPAM_RETRAIN_TIMEOUT_SECONDS": "1200",
                "SPAM_TRAIN_ON_START": "true",
                "SPAM_BOOTSTRAP_MODEL_IF_MISSING": "false",
            },
            clear=False,
        ):
            config = load_runtime_config()

        self.assertEqual(config.api_host, "0.0.0.0")
        self.assertEqual(config.api_port, 8081)
        self.assertEqual(config.log_level, "warning")
        self.assertEqual(config.retrain_timeout_seconds, 1200)
        self.assertTrue(config.train_on_start)
        self.assertFalse(config.bootstrap_model_if_missing)


if __name__ == "__main__":
    unittest.main()
