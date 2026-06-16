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

    vectorizer: Any = None
    if vectorizer_path.suffix == ".pkl":
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
    elif vectorizer_path.is_dir():
        vectorizer = str(vectorizer_path)

    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    return ModelArtifact(model=model, vectorizer=vectorizer, metadata=metadata)


def load_transformer(
    model_path: Path,
    tokenizer_path: Path,
    model_name: str,
    device: str = "cpu",
    cache_dir: str | None = None,
) -> tuple[Any, Any] | None:
    if not (model_path.exists() and tokenizer_path.is_dir()):
        return None

    sha_path = _hash_path(model_path)
    if sha_path.exists():
        _verify_hash(model_path, sha_path.read_text().strip())

    try:
        import torch  # noqa: F811
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        import logging
        logging.getLogger(__name__).warning(
            "transformers/torch not installed — running XGBoost-only."
        )
        return None

    try:
        hf_config = AutoConfig.from_pretrained(
            model_name, num_labels=2,
            cache_dir=cache_dir,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=hf_config,
            cache_dir=cache_dir,
        )
    except (OSError, EnvironmentError):
        import logging
        logging.getLogger(__name__).warning(
            "Could not download %s (network unavailable, model not cached). "
            "Running XGBoost-only. Pre-download with: "
            "python -c \"from transformers import AutoModel; AutoModel.from_pretrained('%s')\"",
            model_name, model_name,
        )
        return None

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
