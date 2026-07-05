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
        vectorizer = str(vectorizer_path)

    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    return ModelArtifact(model=model, vectorizer=vectorizer, metadata=metadata)


def load_transformer(
    model_dir: Path,
    device: str = "cpu",
) -> tuple[Any, Any] | None:
    """Load transformer model and tokenizer from a Hugging Face model directory.

    Expects ``model_dir`` to contain:
      - config.json
      - model.safetensors (preferred) or pytorch_model.bin
      - tokenizer.json
      - tokenizer_config.json

    Falls back gracefully if ``transformers`` / ``torch`` are not installed
    or if the directory is missing required files.
    """
    safetensors_path = model_dir / "model.safetensors"
    _ensure_hf_model_available(model_dir)

    if not model_dir.is_dir() or not (model_dir / "config.json").exists():
        return None

    sha_path = _hash_path(safetensors_path)
    if sha_path.exists():
        _verify_hash(safetensors_path, sha_path.read_text().strip())

    try:
        import torch  # noqa: F811
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        import logging
        logging.getLogger(__name__).warning(
            "transformers/torch not installed — running XGBoost-only."
        )
        return None

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), local_files_only=True,
        )
    except (OSError, EnvironmentError, FileNotFoundError) as exc:
        import logging
        logging.getLogger(__name__).warning(
            "Could not load transformer from %s: %s — running XGBoost-only.",
            model_dir, exc,
        )
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), local_files_only=True,
        )
    except (OSError, EnvironmentError, FileNotFoundError) as exc:
        import logging
        logging.getLogger(__name__).warning(
            "Could not load tokenizer from %s: %s — running XGBoost-only.",
            model_dir, exc,
        )
        return None

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()

    return model, tokenizer


def _ensure_hf_model_available(model_dir: Path) -> None:
    """Download the HF model from HuggingFace Hub if the local directory is empty.

    HF Spaces clones the repo which excludes large model files via .gitignore.
    This downloads them on first startup. Also checks the HF cache directory
    (used by preload_from_hub) as a fallback.
    """
    if (model_dir / "config.json").exists():
        return

    import logging
    import shutil
    logger = logging.getLogger(__name__)

    repo_id = os.environ.get("HF_MODEL_REPO_ID", "Avijit070/spam-email-deberta-v3")
    hf_token = os.environ.get("HF_TOKEN")

    cache_snapshot = _find_in_hf_cache(repo_id)
    if cache_snapshot is not None:
        logger.info("Model found in HF cache at %s — copying to %s", cache_snapshot, model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        for item in cache_snapshot.iterdir():
            dest = model_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
        if (model_dir / "config.json").exists():
            logger.info("Successfully copied model from HF cache.")
            return
        logger.warning("HF cache copy incomplete (missing config.json). Downloading...")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.warning(
            "huggingface_hub not installed — cannot download model from HF Hub."
        )
        return

    import time
    logger.info("Model files not found locally. Downloading %s from HuggingFace Hub...", repo_id)

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            model_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(model_dir),
                token=hf_token,
            )
            if (model_dir / "config.json").exists():
                downloaded = [f.name for f in model_dir.iterdir() if f.is_file()]
                logger.info("Download complete. Files: %s", downloaded)
                return
            else:
                logger.warning(
                    "Download completed but config.json missing (attempt %d/%d).",
                    attempt, max_retries,
                )
        except Exception as exc:
            logger.warning(
                "Download attempt %d/%d failed: %s", attempt, max_retries, exc,
            )

        if model_dir.exists():
            for f in model_dir.iterdir():
                if f.is_file():
                    f.unlink(missing_ok=True)

        if attempt < max_retries:
            wait = 2 ** attempt
            logger.info("Retrying in %ds...", wait)
            time.sleep(wait)

    logger.warning("All download attempts failed. Running XGBoost-only.")


def _find_in_hf_cache(repo_id: str) -> Path | None:
    """Find model files inside the HF Hub cache directory.

    preload_from_hub downloads models to ~/.cache/huggingface/hub during
    Docker build. This function checks that cache as a fallback when the
    model is not in the expected local directory.
    """
    cache_base = os.environ.get("HF_HOME") or os.environ.get(
        "TRANSFORMERS_CACHE"
    ) or os.environ.get("XDG_CACHE_HOME")

    if cache_base:
        cache_dir = Path(cache_base)
        if not cache_dir.name == "hub":
            cache_dir = cache_dir / "hub"
    else:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

    model_cache = cache_dir / ("models--" + repo_id.replace("/", "--"))
    if not model_cache.is_dir():
        return None

    snapshots_dir = model_cache / "snapshots"
    if not snapshots_dir.is_dir():
        return None

    for snapshot in sorted(snapshots_dir.iterdir()):
        if snapshot.is_dir() and (snapshot / "config.json").exists():
            return snapshot

    return None
