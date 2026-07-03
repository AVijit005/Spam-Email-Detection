from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.storage.feedback import (
    FeedbackStoreConfig,
    FeedbackStoreError,
    _TABLE_NAME_RE,
    _append_feedback_mysql,
    _feedback_summary_mysql,
    _load_feedback_entries_mysql,
    append_feedback_entry,
    feedback_backend_name,
    feedback_summary,
    load_feedback_entries,
    resolve_feedback_store,
)


class TestFeedbackStoreConfig(unittest.TestCase):
    def test_resolve_defaults_to_file_when_no_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "feedback.jsonl"
            with mock.patch.dict(os.environ, {}, clear=True):
                config = resolve_feedback_store(log_path)

        self.assertEqual(config.backend, "file")
        self.assertEqual(config.log_path, log_path.resolve())

    def test_resolve_prefers_mysql_when_env_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "feedback.jsonl"
            with mock.patch.dict(os.environ, {
                "SPAM_FEEDBACK_BACKEND": "auto",
                "SPAM_DB_HOST": "127.0.0.1",
                "SPAM_DB_PORT": "3306",
                "SPAM_DB_USER": "root",
                "SPAM_DB_NAME": "spam_detector",
            }, clear=True):
                config = resolve_feedback_store(log_path)

        self.assertEqual(config.backend, "mysql")
        self.assertEqual(config.host, "127.0.0.1")
        self.assertEqual(config.port, 3306)
        self.assertEqual(config.database, "spam_detector")

    def test_resolve_rejects_invalid_table_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "feedback.jsonl"
            with mock.patch.dict(os.environ, {
                "SPAM_DB_HOST": "127.0.0.1",
                "SPAM_DB_USER": "root",
                "SPAM_DB_NAME": "spam_detector",
                "SPAM_DB_TABLE": "feedback; DROP TABLE users;",
            }, clear=True):
                with self.assertRaises(FeedbackStoreError) as ctx:
                    resolve_feedback_store(log_path)
            self.assertIn("Invalid table name", str(ctx.exception))

    def test_resolve_rejects_invalid_port(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "feedback.jsonl"
            with mock.patch.dict(os.environ, {
                "SPAM_DB_HOST": "127.0.0.1",
                "SPAM_DB_PORT": "bad",
                "SPAM_DB_USER": "root",
                "SPAM_DB_NAME": "spam_detector",
            }, clear=True):
                with self.assertRaises(FeedbackStoreError) as ctx:
                    resolve_feedback_store(log_path)
            self.assertIn("Invalid port", str(ctx.exception))

    def test_resolve_rejects_invalid_backend_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "feedback.jsonl"
            with mock.patch.dict(os.environ, {
                "SPAM_FEEDBACK_BACKEND": "postgres",
            }, clear=True):
                with self.assertRaises(FeedbackStoreError):
                    resolve_feedback_store(log_path)

    def test_resolve_force_mysql_mode_requires_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "feedback.jsonl"
            with mock.patch.dict(os.environ, {
                "SPAM_FEEDBACK_BACKEND": "mysql",
            }, clear=True):
                with self.assertRaises(FeedbackStoreError):
                    resolve_feedback_store(log_path)


class TestFeedbackFileBackend(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log_path = Path(self.tmp.name) / "feedback.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def test_append_feedback_file_writes_jsonl(self):
        payload = {
            "feedback_id": "fb-001",
            "prediction_id": "pred-001",
            "stored_at_utc": "2026-04-03T10:00:00+00:00",
            "sender": "spam@example.com",
            "subject": "Test",
            "body": "Test body",
            "predicted_label": "Spam",
            "user_label": "Spam",
            "verdict": "correct",
            "source": "unit_test",
        }
        with mock.patch.dict(os.environ, {}, clear=True):
            append_feedback_entry(payload, self.log_path)

        self.assertTrue(self.log_path.exists())
        entries = load_feedback_entries(self.log_path)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["feedback_id"], "fb-001")

    def test_append_feedback_creates_parent_directory(self):
        deep_path = Path(self.tmp.name) / "nested" / "subdir" / "feedback.jsonl"
        payload = {"feedback_id": "fb-002", "prediction_id": "p2", "user_label": "Spam", "verdict": "correct", "source": "test"}

        with mock.patch.dict(os.environ, {}, clear=True):
            append_feedback_entry(payload, str(deep_path))

        self.assertTrue(deep_path.exists())

    def test_load_feedback_entries_skips_invalid_json(self):
        self.log_path.write_text(
            '{"feedback_id": "ok", "verdict": "correct"}\n'
            'not-json\n'
            '{"feedback_id": "also-ok", "verdict": "false_positive"}\n',
            encoding="utf-8",
        )
        entries = load_feedback_entries(self.log_path)
        self.assertEqual(len(entries), 2)

    def test_load_feedback_entries_empty_file_returns_empty(self):
        self.assertEqual(load_feedback_entries(self.log_path), [])

    def test_load_feedback_entries_missing_file_returns_empty(self):
        missing = Path(self.tmp.name) / "no_such_file.jsonl"
        self.assertEqual(load_feedback_entries(missing), [])

    def test_feedback_summary_counts_verdicts(self):
        entries = [
            {"feedback_id": "1", "verdict": "correct"},
            {"feedback_id": "2", "verdict": "false_positive"},
            {"feedback_id": "3", "verdict": "correct"},
            {"feedback_id": "4", "verdict": "false_negative"},
            {"feedback_id": "5", "verdict": "correct"},
        ]
        with mock.patch("app.storage.feedback._load_feedback_entries_file", return_value=entries), \
             mock.patch.dict(os.environ, {}, clear=True):
            summary = feedback_summary(self.log_path)

        self.assertEqual(summary["feedback_count"], 5)
        self.assertEqual(summary["verdict_counts"]["correct"], 3)
        self.assertEqual(summary["verdict_counts"]["false_positive"], 1)
        self.assertEqual(summary["verdict_counts"]["false_negative"], 1)

    def test_feedback_summary_empty_log_returns_zero(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            summary = feedback_summary(self.log_path)

        self.assertEqual(summary["feedback_count"], 0)
        self.assertEqual(summary["verdict_counts"]["correct"], 0)

    def test_feedback_backend_name_returns_file(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            name = feedback_backend_name(self.log_path)
        self.assertEqual(name, "file")


class TestTableNameValidation(unittest.TestCase):
    def test_valid_table_names(self):
        self.assertTrue(_TABLE_NAME_RE.fullmatch("feedback_entries"))
        self.assertTrue(_TABLE_NAME_RE.fullmatch("a"))
        self.assertTrue(_TABLE_NAME_RE.fullmatch("Feedback_123"))

    def test_invalid_table_names(self):
        self.assertIsNone(_TABLE_NAME_RE.fullmatch("feedback; drop table"))
        self.assertIsNone(_TABLE_NAME_RE.fullmatch("feedback entries"))
        self.assertIsNone(_TABLE_NAME_RE.fullmatch("1feedback"))
        self.assertIsNone(_TABLE_NAME_RE.fullmatch(""))


class TestFeedbackMysqlBackend(unittest.TestCase):
    def setUp(self):
        self.config = FeedbackStoreConfig(
            backend="mysql", log_path=Path("/tmp/feedback.jsonl"),
            host="127.0.0.1", port=3306, user="root",
            password="", database="spam_test", table="feedback_entries",
        )
        self._pymysql_patch = mock.patch("pymysql.connect")
        self.mock_connect = self._pymysql_patch.start()

    def tearDown(self):
        self._pymysql_patch.stop()

    def _mock_cursor(self, fetchall_return=None, fetchone_return=None):
        mock_conn = mock.MagicMock()
        mock_cursor = mock.MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        if fetchall_return is not None:
            mock_cursor.fetchall.return_value = fetchall_return
        if fetchone_return is not None:
            mock_cursor.fetchone.return_value = fetchone_return
        self.mock_connect.return_value = mock_conn
        return mock_conn, mock_cursor

    def test_append_feedback_mysql_creates_table_and_inserts(self):
        mock_conn, mock_cursor = self._mock_cursor()
        payload = {
            "feedback_id": "fb-001", "prediction_id": "pred-001",
            "stored_at_utc": "2026-04-03T10:00:00+00:00",
            "sender": "spam@example.com", "subject": "Test", "body": "Test body",
            "predicted_label": "Spam", "predicted_confidence": 0.95,
            "user_label": "Spam", "verdict": "correct",
            "notes": "", "source": "unit_test", "model_version": "v1",
        }
        _append_feedback_mysql(payload, self.config)
        self.assertEqual(mock_cursor.execute.call_count, 2)
        self.assertIn("CREATE TABLE", mock_cursor.execute.call_args_list[0][0][0])
        mock_conn.close.assert_called_once()

    def test_load_feedback_entries_mysql(self):
        rows = [
            {"feedback_id": "1", "verdict": "correct", "stored_at_utc": "2026-01-01"},
            {"feedback_id": "2", "verdict": "false_positive", "stored_at_utc": "2026-01-02"},
        ]
        mock_conn, mock_cursor = self._mock_cursor(fetchall_return=rows)
        entries = _load_feedback_entries_mysql(self.config)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["feedback_id"], "1")
        mock_conn.close.assert_called_once()

    def test_load_feedback_entries_mysql_empty(self):
        mock_conn, mock_cursor = self._mock_cursor(fetchall_return=[])
        entries = _load_feedback_entries_mysql(self.config)
        self.assertEqual(entries, [])
        mock_conn.close.assert_called_once()

    def test_feedback_summary_mysql(self):
        mock_conn, mock_cursor = self._mock_cursor(
            fetchone_return={"count": 5},
            fetchall_return=[
                {"verdict": "correct", "count": 3},
                {"verdict": "false_positive", "count": 1},
                {"verdict": "false_negative", "count": 1},
            ],
        )
        summary = _feedback_summary_mysql(self.config)
        self.assertEqual(summary["feedback_count"], 5)
        self.assertEqual(summary["verdict_counts"]["correct"], 3)
        self.assertEqual(summary["verdict_counts"]["false_positive"], 1)
        self.assertEqual(summary["verdict_counts"]["false_negative"], 1)
        mock_conn.close.assert_called_once()

    def test_feedback_summary_mysql_empty(self):
        mock_conn, mock_cursor = self._mock_cursor(
            fetchone_return={"count": 0},
            fetchall_return=[],
        )
        summary = _feedback_summary_mysql(self.config)
        self.assertEqual(summary["feedback_count"], 0)
        self.assertEqual(summary["verdict_counts"]["correct"], 0)
        mock_conn.close.assert_called_once()

    def test_append_feedback_routes_to_mysql_when_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "feedback.jsonl"
            with mock.patch.dict(os.environ, {
                "SPAM_DB_HOST": "127.0.0.1", "SPAM_DB_USER": "root",
                "SPAM_DB_NAME": "spam_test",
            }, clear=True):
                mock_conn, mock_cursor = self._mock_cursor()
                payload = {
                    "feedback_id": "fb-x", "prediction_id": "p-x",
                    "stored_at_utc": "2026-01-01T00:00:00+00:00",
                    "predicted_label": "Spam", "user_label": "Spam",
                    "verdict": "correct", "source": "test",
                }
                append_feedback_entry(payload, log_path)
            self.assertGreater(mock_cursor.execute.call_count, 0)
            mock_conn.close.assert_called_once()
