from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_ALLOW_ORIGIN_REGEX = (
    r"^(chrome-extension://.*|moz-extension://.*|http://localhost(:\d+)?|http://127\.0\.0\.1(:\d+)?)$"
)


def _env_text(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        result = default
    else:
        try:
            result = int(raw_value)
        except ValueError as error:
            raise RuntimeError(f"{name} must be an integer.") from error

    if minimum is not None:
        result = max(result, minimum)
    if maximum is not None:
        result = min(result, maximum)
    return result


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean-like value.")


@dataclass(frozen=True)
class RuntimeConfig:
    api_host: str
    api_port: int
    log_level: str
    retrain_timeout_seconds: int
    allow_origin_regex: str
    train_on_start: bool
    bootstrap_model_if_missing: bool


def load_runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        api_host=_env_text("SPAM_API_HOST", "127.0.0.1"),
        api_port=_env_int("SPAM_API_PORT", 8000, minimum=1, maximum=65535),
        log_level=_env_text("SPAM_LOG_LEVEL", "info").lower(),
        retrain_timeout_seconds=_env_int("SPAM_RETRAIN_TIMEOUT_SECONDS", 15 * 60, minimum=30),
        allow_origin_regex=_env_text("SPAM_ALLOW_ORIGIN_REGEX", DEFAULT_ALLOW_ORIGIN_REGEX),
        train_on_start=_env_bool("SPAM_TRAIN_ON_START", False),
        bootstrap_model_if_missing=_env_bool("SPAM_BOOTSTRAP_MODEL_IF_MISSING", True),
    )
