from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from feedback_store import feedback_backend_name, resolve_feedback_store


class FeedbackStoreTests(unittest.TestCase):
    def test_default_backend_is_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_path = Path(temp_dir) / "feedback.jsonl"
            with mock.patch.dict(os.environ, {}, clear=False):
                config = resolve_feedback_store(feedback_path)

        self.assertEqual(config.backend, "file")
        self.assertEqual(feedback_backend_name(feedback_path), "file")

    def test_mysql_backend_is_selected_when_env_present(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_path = Path(temp_dir) / "feedback.jsonl"
            with mock.patch.dict(
                os.environ,
                {
                    "SPAM_FEEDBACK_BACKEND": "auto",
                    "SPAM_DB_HOST": "127.0.0.1",
                    "SPAM_DB_PORT": "3306",
                    "SPAM_DB_USER": "root",
                    "SPAM_DB_NAME": "spam_detector",
                },
                clear=False,
            ):
                config = resolve_feedback_store(feedback_path)

        self.assertEqual(config.backend, "mysql")
        self.assertEqual(config.host, "127.0.0.1")
        self.assertEqual(config.port, 3306)
        self.assertEqual(config.database, "spam_detector")


if __name__ == "__main__":
    unittest.main()
