from __future__ import annotations

import hashlib
import json
import pickle
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from feedback_store import FeedbackStoreError, feedback_backend_name, load_feedback_entries
from spam_detector_core import (
    DEFAULT_SPAM_THRESHOLD,
    META_FEATURE_NAMES,
    compose_email_text,
    extract_meta_features,
    preprocess_text,
)


DATA_PATH = CURRENT_DIR.parent / "data" / "spam.csv"
FEEDBACK_PATH = CURRENT_DIR.parent / "data" / "feedback.jsonl"
MODEL_PATH = CURRENT_DIR / "spam_model.pkl"
VECTORIZER_PATH = CURRENT_DIR / "vectorizer.pkl"
METADATA_PATH = CURRENT_DIR / "model_metadata.json"

BASE_SAMPLE_WEIGHT = 1.0
FEEDBACK_CONFIRMATION_WEIGHT = 1.5
FEEDBACK_CORRECTION_WEIGHT = 3.0


def create_word_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
    )


def create_char_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=5000,
        min_df=2,
    )


def load_dataset() -> pd.DataFrame:
    try:
        dataframe = pd.read_csv(DATA_PATH, encoding="latin-1")
    except FileNotFoundError as error:
        raise SystemExit(f"Dataset not found: {DATA_PATH}") from error

    if {"Body", "Label"}.issubset(dataframe.columns):
        dataframe = dataframe[["Body", "Label"]].rename(columns={"Body": "message", "Label": "label"})
    elif {"v1", "v2"}.issubset(dataframe.columns):
        dataframe = dataframe[["v2", "v1"]].rename(columns={"v2": "message", "v1": "label"})
        dataframe["label"] = dataframe["label"].map({"ham": 0, "spam": 1})
    else:
        raise SystemExit(f"Unsupported dataset schema: {dataframe.columns.tolist()}")

    dataframe.dropna(subset=["message", "label"], inplace=True)
    dataframe["message"] = dataframe["message"].astype(str)
    dataframe["label"] = dataframe["label"].astype(int)
    dataframe["sample_weight"] = BASE_SAMPLE_WEIGHT
    dataframe["dataset_source"] = "base_dataset"
    return dataframe


def normalize_feedback_label(raw_label: Any) -> int | None:
    normalized = str(raw_label or "").strip().lower()
    if normalized in {"spam", "junk"}:
        return 1
    if normalized in {"not spam", "ham", "safe", "legitimate", "whitelisted"}:
        return 0
    return None


def feedback_sample_weight(verdict: Any) -> float:
    verdict_text = str(verdict or "").strip().lower()
    if verdict_text in {"false_positive", "false_negative"}:
        return FEEDBACK_CORRECTION_WEIGHT
    return FEEDBACK_CONFIRMATION_WEIGHT


def build_feedback_message(subject: Any, body: Any) -> str:
    return compose_email_text(str(subject or ""), str(body or ""), subject_weight=1)


def feedback_fingerprint(subject: Any, body: Any) -> str:
    message = build_feedback_message(subject, body).strip().lower()
    return hashlib.sha256(message.encode("utf-8")).hexdigest()


def empty_feedback_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "message": pd.Series(dtype="string"),
            "label": pd.Series(dtype="int64"),
            "sample_weight": pd.Series(dtype="float64"),
            "dataset_source": pd.Series(dtype="string"),
        }
    )


def load_feedback_dataset(feedback_path: str | Path = FEEDBACK_PATH) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(feedback_path)
    stats: dict[str, Any] = {
        "feedback_backend": "file",
        "feedback_file_present": path.exists(),
        "feedback_lines_read": 0,
        "parsed_feedback_entries": 0,
        "invalid_json_lines": 0,
        "skipped_invalid_label": 0,
        "skipped_empty_message": 0,
        "feedback_rows_used": 0,
        "duplicates_collapsed": 0,
        "correction_rows_used": 0,
        "confirmation_rows_used": 0,
        "label_counts": {"0": 0, "1": 0},
        "last_feedback_at_utc": None,
        "sample_weight_total": 0.0,
    }

    try:
        stats["feedback_backend"] = feedback_backend_name(path)
    except FeedbackStoreError as error:
        raise SystemExit(f"Could not load feedback dataset: {error}") from error

    if stats["feedback_backend"] == "file" and not path.exists():
        return empty_feedback_dataframe(), stats

    if stats["feedback_backend"] == "file":
        entries: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stats["feedback_lines_read"] += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    stats["invalid_json_lines"] += 1
    else:
        try:
            entries = load_feedback_entries(path)
        except FeedbackStoreError as error:
            raise SystemExit(f"Could not load feedback dataset: {error}") from error

    latest_by_fingerprint: dict[str, dict[str, Any]] = {}

    for entry in entries:
        if stats["feedback_backend"] != "file":
            stats["feedback_lines_read"] += 1

        label = normalize_feedback_label(entry.get("user_label"))
        if label is None:
            stats["skipped_invalid_label"] += 1
            continue

        message = build_feedback_message(entry.get("subject", ""), entry.get("body", ""))
        if not message.strip():
            stats["skipped_empty_message"] += 1
            continue

        fingerprint = feedback_fingerprint(entry.get("subject", ""), entry.get("body", ""))
        latest_by_fingerprint[fingerprint] = {
            "message": message,
            "label": label,
            "sample_weight": feedback_sample_weight(entry.get("verdict")),
            "dataset_source": "feedback",
            "stored_at_utc": str(entry.get("stored_at_utc") or ""),
            "verdict": str(entry.get("verdict") or ""),
        }
        stats["parsed_feedback_entries"] += 1

    if not latest_by_fingerprint:
        return empty_feedback_dataframe(), stats

    feedback_dataframe = pd.DataFrame(latest_by_fingerprint.values())
    stats["feedback_rows_used"] = int(len(feedback_dataframe))
    stats["duplicates_collapsed"] = int(stats["parsed_feedback_entries"] - len(feedback_dataframe))
    stats["correction_rows_used"] = int(
        feedback_dataframe["verdict"].isin(["false_positive", "false_negative"]).sum()
    )
    stats["confirmation_rows_used"] = int((feedback_dataframe["verdict"] == "correct").sum())
    stats["sample_weight_total"] = round(float(feedback_dataframe["sample_weight"].sum()), 2)

    label_counts = {"0": 0, "1": 0}
    for key, value in feedback_dataframe["label"].value_counts().to_dict().items():
        label_counts[str(key)] = int(value)
    stats["label_counts"] = label_counts

    stored_times = [value for value in feedback_dataframe["stored_at_utc"].tolist() if value]
    stats["last_feedback_at_utc"] = max(stored_times) if stored_times else None

    return feedback_dataframe[["message", "label", "sample_weight", "dataset_source"]], stats


def prepare_training_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    prepared = dataframe.copy()
    prepared["message"] = prepared["message"].astype(str)
    prepared["processed"] = prepared["message"].apply(preprocess_text)
    prepared["sample_weight"] = prepared["sample_weight"].astype(float)
    prepared["label"] = prepared["label"].astype(int)
    return prepared


def score_candidate(
    name: str,
    estimator,
    x_train,
    x_test,
    y_train,
    y_test,
    sample_weight_train,
) -> tuple[dict[str, Any], object]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        estimator.fit(x_train, y_train, sample_weight=sample_weight_train)
    predictions = estimator.predict(x_test)
    report = classification_report(
        y_test,
        predictions,
        target_names=["Ham", "Spam"],
        output_dict=True,
        zero_division=0,
    )

    spam_metrics = report["Spam"]
    accuracy = accuracy_score(y_test, predictions)
    metrics = {
        "model_name": name,
        "accuracy": round(float(accuracy), 4),
        "spam_precision": round(float(spam_metrics["precision"]), 4),
        "spam_recall": round(float(spam_metrics["recall"]), 4),
        "spam_f1": round(float(spam_metrics["f1-score"]), 4),
        "support": int(spam_metrics["support"]),
    }

    print(f"\n--- {name} ---")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Spam F1  : {metrics['spam_f1']:.4f}")
    print(classification_report(y_test, predictions, target_names=["Ham", "Spam"], zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    return metrics, estimator


def train() -> dict[str, Any]:
    print("=" * 60)
    print("  Spam Detector - Training")
    print("=" * 60)

    base_dataframe = load_dataset()
    feedback_dataframe, feedback_stats = load_feedback_dataset()

    print(f"Base dataset shape : {base_dataframe.shape}")
    print(f"Base class counts  : {base_dataframe['label'].value_counts().to_dict()}")
    print(
        "Feedback samples  : "
        f"{feedback_stats['feedback_rows_used']} used"
        f" ({feedback_stats['correction_rows_used']} corrections, "
        f"{feedback_stats['confirmation_rows_used']} confirmations)"
    )

    print("\n[1/4] Preprocessing text...")
    base_dataframe = prepare_training_rows(base_dataframe)
    feedback_dataframe = prepare_training_rows(feedback_dataframe) if not feedback_dataframe.empty else empty_feedback_dataframe()

    print("[2/4] Splitting base dataset before fitting text features...")
    base_train_df, base_test_df = train_test_split(
        base_dataframe,
        test_size=0.20,
        random_state=42,
        stratify=base_dataframe["label"],
    )
    train_df = pd.concat([base_train_df, feedback_dataframe], ignore_index=True)
    train_df["label"] = train_df["label"].astype(int)
    train_df["sample_weight"] = train_df["sample_weight"].astype(float)
    test_df = base_test_df.copy()
    print(f"Base train rows     : {len(base_train_df)}")
    print(f"Base test rows      : {len(base_test_df)}")
    print(f"Feedback train rows : {len(feedback_dataframe)}")
    print(f"Eval train rows     : {len(train_df)}")

    print("[3/4] Building holdout features and evaluating candidates...")
    evaluation_word_vectorizer = create_word_vectorizer()
    evaluation_char_vectorizer = create_char_vectorizer()
    x_train_word = evaluation_word_vectorizer.fit_transform(train_df["processed"])
    x_test_word = evaluation_word_vectorizer.transform(test_df["processed"])
    x_train_char = evaluation_char_vectorizer.fit_transform(train_df["message"].str.lower())
    x_test_char = evaluation_char_vectorizer.transform(test_df["message"].str.lower())
    x_train_meta = sp.csr_matrix(extract_meta_features(train_df["message"].tolist()))
    x_test_meta = sp.csr_matrix(extract_meta_features(test_df["message"].tolist()))
    x_train = sp.hstack([x_train_word, x_train_char, x_train_meta], format="csr")
    x_test = sp.hstack([x_test_word, x_test_char, x_test_meta], format="csr")
    y_train = train_df["label"].values
    y_test = test_df["label"].values
    sample_weight_train = train_df["sample_weight"].values
    print(f"Holdout train matrix : {x_train.shape}")
    print(f"Holdout test matrix  : {x_test.shape}")

    candidates = {
        "LogisticRegression": LogisticRegression(
            C=1.0,
            max_iter=3000,
            class_weight=None,
            solver="lbfgs",
        ),
    }

    evaluations: list[dict[str, Any]] = []
    best_metrics: dict[str, Any] | None = None
    best_model_name: str | None = None
    best_estimator = None

    for name, estimator in candidates.items():
        metrics, trained_model = score_candidate(
            name,
            estimator,
            x_train,
            x_test,
            y_train,
            y_test,
            sample_weight_train,
        )
        evaluations.append(metrics)
        if best_metrics is None or (metrics["spam_f1"], metrics["accuracy"]) > (
            best_metrics["spam_f1"],
            best_metrics["accuracy"],
        ):
            best_metrics = metrics
            best_model_name = name
            best_estimator = clone(trained_model)

    if best_metrics is None or best_model_name is None or best_estimator is None:
        raise SystemExit("Training failed: no model candidates were evaluated.")

    print(f"\nSelected model : {best_model_name}")

    print("[4/4] Retraining selected model on the full dataset and saving artefacts...")
    full_training_df = pd.concat([base_dataframe, feedback_dataframe], ignore_index=True)
    full_training_df["label"] = full_training_df["label"].astype(int)
    full_training_df["sample_weight"] = full_training_df["sample_weight"].astype(float)
    final_word_vectorizer = create_word_vectorizer()
    final_char_vectorizer = create_char_vectorizer()
    x_all_word = final_word_vectorizer.fit_transform(full_training_df["processed"])
    x_all_char = final_char_vectorizer.fit_transform(full_training_df["message"].str.lower())
    x_all_meta = sp.csr_matrix(extract_meta_features(full_training_df["message"].tolist()))
    x_all = sp.hstack([x_all_word, x_all_char, x_all_meta], format="csr")
    y_all = full_training_df["label"].values
    sample_weight_all = full_training_df["sample_weight"].values
    print(f"Full training matrix : {x_all.shape}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        best_estimator.fit(x_all, y_all, sample_weight=sample_weight_all)

    vectorizer_bundle = {
        "version": 2,
        "word_vectorizer": final_word_vectorizer,
        "char_vectorizer": final_char_vectorizer,
        "meta_feature_names": META_FEATURE_NAMES,
    }

    with MODEL_PATH.open("wb") as model_handle:
        pickle.dump(best_estimator, model_handle)
    with VECTORIZER_PATH.open("wb") as vectorizer_handle:
        pickle.dump(vectorizer_bundle, vectorizer_handle)

    full_label_counts = full_training_df["label"].value_counts().to_dict()
    metadata = {
        "model_name": best_model_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "spam_threshold": DEFAULT_SPAM_THRESHOLD,
        "data_leakage_fixed": True,
        "retrained_on_full_dataset": True,
        "feature_shape": [int(x_all.shape[0]), int(x_all.shape[1])],
        "feature_sources": {
            "word_tfidf_features": int(x_all_word.shape[1]),
            "char_tfidf_features": int(x_all_char.shape[1]),
            "meta_features": int(x_all_meta.shape[1]),
        },
        "dataset_rows": int(len(full_training_df)),
        "base_dataset_rows": int(len(base_dataframe)),
        "class_counts": {str(key): int(value) for key, value in full_label_counts.items()},
        "holdout_split": {
            "train_rows": int(len(base_train_df)),
            "test_rows": int(len(base_test_df)),
            "feedback_rows_added_to_training": int(len(feedback_dataframe)),
            "eval_train_rows": int(len(train_df)),
        },
        "selected_metrics": best_metrics,
        "candidate_metrics": evaluations,
        "feedback_training": feedback_stats,
    }
    with METADATA_PATH.open("w", encoding="utf-8") as metadata_handle:
        json.dump(metadata, metadata_handle, indent=2)

    print(f"Model      -> {MODEL_PATH}")
    print(f"Vectorizer -> {VECTORIZER_PATH}")
    print(f"Metadata   -> {METADATA_PATH}")
    print("\nTraining complete.")
    return metadata


if __name__ == "__main__":
    train()
