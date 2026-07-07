from __future__ import annotations

import hashlib
import hmac
import json
import os
import pickle
import shutil
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
        vectorizer = {"version": 1, "word_vec": None, "model_type": "transformer",
                      "char_vectorizer": None, "meta_feature_names": []}

    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    return ModelArtifact(model=model, vectorizer=vectorizer, metadata=metadata)


def load_transformer(
    model_dir: Path,
    device: str = "cpu",
    pt_model_path: Path | None = None,
    pt_tokenizer_dir: Path | None = None,
) -> tuple[Any, Any] | None:
    """Load transformer model and tokenizer.

    Tries two strategies:
      1. HF format in ``model_dir`` (config.json + model.safetensors)
      2. Original trained format (pt_model_path .pt + pt_tokenizer_dir)

    Falls back gracefully if ``transformers`` / ``torch`` are not installed.
    """
    try:
        import torch  # noqa: F811
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        import logging
        logging.getLogger(__name__).warning(
            "transformers/torch not installed — running XGBoost-only."
        )
        return None

    # Strategy 1: HF format in model_dir
    _ensure_hf_model_available(model_dir)
    if model_dir.is_dir() and (model_dir / "config.json").exists():
        safetensors_path = model_dir / "model.safetensors"
        sha_path = _hash_path(safetensors_path)
        if sha_path.exists():
            _verify_hash(safetensors_path, sha_path.read_text().strip())

        try:
            hf_model = AutoModelForSequenceClassification.from_pretrained(
                str(model_dir), local_files_only=True,
            )
            hf_tokenizer = AutoTokenizer.from_pretrained(
                str(model_dir), local_files_only=True,
            )
            if hf_tokenizer.pad_token is None:
                hf_tokenizer.pad_token = hf_tokenizer.eos_token
            hf_model.to(device)
            hf_model.eval()
            return hf_model, hf_tokenizer
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "HF format load failed from %s: %s — trying .pt fallback.", model_dir, exc,
            )

    # Strategy 2: Original trained .pt format
    if pt_model_path and pt_model_path.exists() and pt_tokenizer_dir and pt_tokenizer_dir.is_dir():
        try:
            import logging
            logger = logging.getLogger(__name__)
            logger.info("Loading transformer from .pt file: %s", pt_model_path)
            pt_model = torch.load(str(pt_model_path), map_location=device, weights_only=False)
            if hasattr(pt_model, "eval"):
                pt_model.eval()
            pt_tokenizer = AutoTokenizer.from_pretrained(
                str(pt_tokenizer_dir), local_files_only=True,
            )
            if pt_tokenizer.pad_token is None:
                pt_tokenizer.pad_token = pt_tokenizer.eos_token
            return pt_model, pt_tokenizer
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                ".pt load failed from %s: %s — running XGBoost-only.", pt_model_path, exc,
            )

    return None


def _ensure_hf_model_available(model_dir: Path) -> None:
    """Download the HF model from HuggingFace Hub if the local directory is empty.

    HF Spaces clones the repo which excludes large model files via .gitignore.
    This downloads them on first startup.
    """
    if (model_dir / "config.json").exists():
        return

    repo_id = os.environ.get("HF_MODEL_REPO_ID", "pavitra55/spam-email-deberta-v3")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        import logging
        logging.getLogger(__name__).warning(
            "huggingface_hub not installed — cannot download model from HF Hub."
        )
        return

    import logging
    logger = logging.getLogger(__name__)
    logger.info("Model files not found locally. Downloading %s from HuggingFace Hub...", repo_id)
    try:
        model_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )
        logger.info("Download complete. Model cached at %s", model_dir)
    except Exception as exc:
        logger.warning(
            "Failed to download model from HF Hub: %s. Running XGBoost-only.", exc
        )
        if model_dir.exists():
            for f in model_dir.iterdir():
                f.unlink(missing_ok=True)
