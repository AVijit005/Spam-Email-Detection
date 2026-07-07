from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_TABLE_NAME = "feedback_entries"
_TABLE_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class FeedbackStoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class FeedbackStoreConfig:
    backend: str
    log_path: Path
    host: str = ""
    port: int = 3306
    user: str = ""
    password: str = ""
    database: str = ""
    table: str = DEFAULT_TABLE_NAME


def _env_first(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and value != "":
            return value
    return default


def resolve_feedback_store(log_path: str | Path, backend_override: str | None = None) -> FeedbackStoreConfig:
    path = Path(log_path)
    if backend_override and backend_override.strip():
        mode = backend_override.strip().lower()
    else:
        mode = _env_first("SPAM_FEEDBACK_BACKEND", default="auto").strip().lower() or "auto"
    host = _env_first("SPAM_DB_HOST", "MYSQL_HOST")
    port_text = _env_first("SPAM_DB_PORT", "MYSQL_PORT", default="3306")
    user = _env_first("SPAM_DB_USER", "MYSQL_USER")
    password = _env_first("SPAM_DB_PASSWORD", "MYSQL_PASSWORD")
    database = _env_first("SPAM_DB_NAME", "SPAM_DB_DATABASE", "MYSQL_DATABASE")
    table = _env_first("SPAM_DB_TABLE", "MYSQL_TABLE", default=DEFAULT_TABLE_NAME)
    if not _TABLE_NAME_RE.fullmatch(table):
        raise FeedbackStoreError(f"Invalid table name: {table!r}. Must match '^[a-zA-Z_][a-zA-Z0-9_]*$'.")
    try:
        port = int(port_text)
    except ValueError as error:
        raise FeedbackStoreError("Invalid port number.") from error
    mysql_configured = bool(host and user and database)
    if mode not in {"auto", "file", "mysql"}:
        raise FeedbackStoreError("SPAM_FEEDBACK_BACKEND must be one of: auto, file, mysql.")
    if mode == "mysql" or (mode == "auto" and mysql_configured):
        if not mysql_configured:
            raise FeedbackStoreError("MySQL feedback storage requires host, user, and database.")
        return FeedbackStoreConfig(backend="mysql", log_path=path, host=host, port=port,
                                   user=user, password=password, database=database, table=table)
    return FeedbackStoreConfig(backend="file", log_path=path)


def feedback_backend_name(log_path: str | Path) -> str:
    return resolve_feedback_store(log_path).backend


def append_feedback_entry(payload: dict[str, Any], log_path: str | Path) -> None:
    config = resolve_feedback_store(log_path)
    if config.backend == "mysql":
        _append_feedback_mysql(payload, config)
        return
    _append_feedback_file(payload, config.log_path)


def _append_feedback_file(payload: dict[str, Any], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _append_feedback_mysql(payload: dict[str, Any], config: FeedbackStoreConfig) -> None:
    import pymysql
    connection = pymysql.connect(host=config.host, port=config.port, user=config.user,
                                  password=config.password, database=config.database,
                                  charset="utf8mb4", autocommit=True,
                                  cursorclass=pymysql.cursors.DictCursor)
    try:
        ddl = f"""CREATE TABLE IF NOT EXISTS `{config.table}` (
            feedback_id VARCHAR(255) NOT NULL PRIMARY KEY,
            prediction_id VARCHAR(255) NOT NULL, stored_at_utc VARCHAR(64) NOT NULL,
            sender TEXT NULL, subject TEXT NULL, body MEDIUMTEXT NULL,
            predicted_label VARCHAR(32) NOT NULL, predicted_confidence DOUBLE NULL,
            user_label VARCHAR(32) NOT NULL, verdict VARCHAR(32) NOT NULL,
            notes TEXT NULL, source VARCHAR(128) NOT NULL, model_version VARCHAR(128) NULL,
            INDEX idx_feedback_stored_at (stored_at_utc),
            INDEX idx_feedback_prediction (prediction_id)
        ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"""
        with connection.cursor() as cursor:
            cursor.execute(ddl)
            cursor.execute(f"""INSERT INTO `{config.table}` (
                feedback_id, prediction_id, stored_at_utc, sender, subject, body,
                predicted_label, predicted_confidence, user_label, verdict, notes,
                source, model_version
            ) VALUES (%(feedback_id)s,%(prediction_id)s,%(stored_at_utc)s,%(sender)s,
                %(subject)s,%(body)s,%(predicted_label)s,%(predicted_confidence)s,
                %(user_label)s,%(verdict)s,%(notes)s,%(source)s,%(model_version)s)""", payload)
    finally:
        connection.close()


def load_feedback_entries(log_path: str | Path) -> list[dict[str, Any]]:
    config = resolve_feedback_store(log_path)
    if config.backend == "mysql":
        return _load_feedback_entries_mysql(config)
    return _load_feedback_entries_file(config.log_path)


def _load_feedback_entries_file(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _load_feedback_entries_mysql(config: FeedbackStoreConfig) -> list[dict[str, Any]]:
    import pymysql
    connection = pymysql.connect(host=config.host, port=config.port, user=config.user,
                                  password=config.password, database=config.database,
                                  charset="utf8mb4", autocommit=True,
                                  cursorclass=pymysql.cursors.DictCursor)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT * FROM `{config.table}` ORDER BY stored_at_utc ASC")
            return [dict(row) for row in cursor.fetchall()]
    finally:
        connection.close()


def feedback_summary(log_path: str | Path) -> dict[str, Any]:
    config = resolve_feedback_store(log_path)
    if config.backend == "mysql":
        return _feedback_summary_mysql(config)
    return _feedback_summary_from_entries(_load_feedback_entries_file(config.log_path))


def _feedback_summary_from_entries(entries: list[dict[str, Any]]) -> dict[str, Any]:
    verdict_counts = {"correct": 0, "false_positive": 0, "false_negative": 0}
    for entry in entries:
        verdict = entry.get("verdict")
        if verdict in verdict_counts:
            verdict_counts[verdict] += 1
    return {"feedback_count": len(entries), "verdict_counts": verdict_counts}


def _feedback_summary_mysql(config: FeedbackStoreConfig) -> dict[str, Any]:
    import pymysql
    connection = pymysql.connect(host=config.host, port=config.port, user=config.user,
                                  password=config.password, database=config.database,
                                  charset="utf8mb4", autocommit=True,
                                  cursorclass=pymysql.cursors.DictCursor)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) AS count FROM `{config.table}`")
            total = cursor.fetchone() or {"count": 0}
            cursor.execute(f"SELECT verdict, COUNT(*) AS count FROM `{config.table}` GROUP BY verdict")
            verdict_counts = {"correct": 0, "false_positive": 0, "false_negative": 0}
            for row in cursor.fetchall():
                v = row.get("verdict")
                if v in verdict_counts:
                    verdict_counts[v] = int(row.get("count", 0))
        return {"feedback_count": int(total.get("count", 0)), "verdict_counts": verdict_counts}
    finally:
        connection.close()
