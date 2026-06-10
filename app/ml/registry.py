from __future__ import annotations

import hashlib
import hmac
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ModelIntegrityError(RuntimeError):
    pass


@dataclass
class ModelArtifact:
    model: Any
    vectorizer: Any
    metadata: dict[str, Any]


def _compute_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_hash(path: Path, expected: str) -> None:
    actual = _compute_hash(path)
    if not hmac.compare_digest(actual, expected):
        raise ModelIntegrityError(
            f"SHA-256 mismatch for {path.name}: integrity check failed."
        )


def _hash_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def save_model(
    model: Any, vectorizer: Any, metadata: dict[str, Any],
    model_path: Path, vectorizer_path: Path, metadata_path: Path,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    _hash_path(model_path).write_text(_compute_hash(model_path))
    _hash_path(vectorizer_path).write_text(_compute_hash(vectorizer_path))


def load_model(
    model_path: Path, vectorizer_path: Path, metadata_path: Path,
) -> ModelArtifact | None:
    if not (model_path.exists() and vectorizer_path.exists()):
        return None
    model_hash_path = _hash_path(model_path)
    vectorizer_hash_path = _hash_path(vectorizer_path)
    if model_hash_path.exists():
        _verify_hash(model_path, model_hash_path.read_text().strip())
    if vectorizer_hash_path.exists():
        _verify_hash(vectorizer_path, vectorizer_hash_path.read_text().strip())
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    return ModelArtifact(model=model, vectorizer=vectorizer, metadata=metadata)
