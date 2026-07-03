from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SPAM_", env_file=".env", env_file_encoding="utf-8")

    api_host: str = "0.0.0.0"
    api_port: int = int(os.environ.get("PORT", "8000"))
    log_level: str = "info"

    allow_origin_regex: str = (
        r"^(chrome-extension://[a-z]{32,64}|moz-extension://[a-z0-9-]{8,64}|"
        r"http://localhost(:\d+)?|http://127\.0\.0\.1(:\d+)?|"
        r"https://[a-zA-Z0-9-]+\.hf\.space)$"
    )

    train_on_start: bool = False
    bootstrap_model_if_missing: bool = True
    retrain_timeout_seconds: int = 900
    spam_threshold: float = 0.55

    feedback_backend: str = "file"
    feedback_log_path: Path = DATA_DIR / "feedback.jsonl"

    spam_csv_path: Path = DATA_DIR / "spam.csv"
    trusted_domains_path: Path = DATA_DIR / "trusted_domains.csv"
    whitelist_path: Path = DATA_DIR / "whitelist.csv"

    model_path: Path = MODEL_DIR / "spam_model.pkl"
    vectorizer_path: Path = MODEL_DIR / "vectorizer.pkl"
    metadata_path: Path = MODEL_DIR / "model_metadata.json"
    train_script_path: Path = MODEL_DIR / "train_model.py"
    model_dir: Path = MODEL_DIR

    environment: str = "development"
    prediction_retention_days: int = 30
    feedback_retention_days: int = 180

    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 15

    api_key: str = ""

    # --- Ensemble / Transformer Configuration ---
    enable_transformer: bool = True
    transformer_model_dir: Path = MODEL_DIR / "hf_model"
    transformer_model_name: str = "microsoft/deberta-v3-base"
    transformer_device: str = "cpu"


settings = Settings()
