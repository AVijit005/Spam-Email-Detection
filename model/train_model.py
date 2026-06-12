"""Model training orchestrator — runs Track A and Track B, compares, saves best.

Usage:
    python train_model.py              # classical only (production)
    python train_model.py --kaggle     # both tracks (competition, GPU required)

Accepts --track-a/--track-b flags for selective training.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.constants import DEFAULT_SPAM_THRESHOLD, META_FEATURE_NAMES
from app.core.features import compose_email_text
from app.core.text import preprocess_text
from app.storage.feedback import (
    FeedbackStoreError, feedback_backend_name, load_feedback_entries,
)
from model.shared import (
    EvalMetrics, print_leaderboard, print_cross_track_summary,
    save_artifacts, ram_report,
)

DATA_PATH = PROJECT_ROOT / "data" / "spam.csv"
FEEDBACK_PATH = PROJECT_ROOT / "data" / "feedback.jsonl"
MODEL_PATH = CURRENT_DIR / "spam_model.pkl"
VECTORIZER_PATH = CURRENT_DIR / "vectorizer.pkl"
METADATA_PATH = CURRENT_DIR / "model_metadata.json"

BASE_SAMPLE_WEIGHT = 1.0
FEEDBACK_CONFIRMATION_WEIGHT = 1.5
FEEDBACK_CORRECTION_WEIGHT = 3.0


def load_dataset() -> pd.DataFrame:
    try:
        dataframe = pd.read_csv(DATA_PATH, encoding="utf-8")
    except (FileNotFoundError, UnicodeDecodeError):
        try:
            dataframe = pd.read_csv(DATA_PATH, encoding="latin-1")
        except FileNotFoundError as error:
            raise SystemExit(f"Dataset not found: {DATA_PATH}") from error

    if {"text", "label"}.issubset(dataframe.columns):
        dataframe = dataframe[["text", "label"]].rename(columns={"text": "message", "label": "label"})
        dataframe["label"] = dataframe["label"].map({"ham": 0, "spam": 1})
    elif {"Body", "Label"}.issubset(dataframe.columns):
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


def prepare_training_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    prepared = dataframe.copy()
    prepared["message"] = prepared["message"].astype(str)
    prepared["processed"] = prepared["message"].apply(preprocess_text)
    prepared["sample_weight"] = prepared["sample_weight"].astype(float)
    prepared["label"] = prepared["label"].astype(int)
    return prepared


def train(*, run_track_a: bool = True, run_track_b: bool = False,
          competition: bool = False, device: str = "cuda") -> dict[str, Any]:
    mode = "competition" if competition else "production"
    print("=" * 70)
    print(f"  Spam Detector — Training Orchestrator [{mode}]")
    print(f"  Track A (Classical): {'ON' if run_track_a else 'OFF'}")
    print(f"  Track B (Transformer): {'ON' if run_track_b else 'OFF'}")
    print("=" * 70)
    print(f"  {ram_report('Start')}")

    from sklearn.model_selection import train_test_split

    base_dataframe = load_dataset()
    print(f"\n  Dataset: {len(base_dataframe):,} rows  "
          f"({(base_dataframe['label'] == 1).sum():,} spam, "
          f"{(base_dataframe['label'] == 0).sum():,} ham)")

    print("\n  Preprocessing text...")
    t0 = time.perf_counter()
    base_dataframe = prepare_training_rows(base_dataframe)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")
    print(f"  {ram_report('After preprocess')}")

    print("\n  Splitting dataset (80/20, stratified)...")
    base_train_df, base_test_df = train_test_split(
        base_dataframe, test_size=0.20, random_state=42,
        stratify=base_dataframe["label"],
    )
    print(f"  Train: {len(base_train_df):,}  Test: {len(base_test_df):,}")

    all_metrics: list[EvalMetrics] = []
    track_a_best: EvalMetrics | None = None
    track_b_best: EvalMetrics | None = None
    track_a_features: dict[str, Any] = {}
    track_b_features: dict[str, Any] = {}

    best_model: Any = None
    best_vectorizer_bundle: dict[str, Any] = {}
    best_metadata: dict[str, Any] = {}

    if run_track_a:
        from model.train_classical import train_classical
        metrics_a, best_a, feats_a, word_vec, best_est_a = train_classical(
            base_train_df, base_test_df, competition=competition,
        )
        all_metrics.extend(metrics_a)
        track_a_best = best_a
        track_a_features = feats_a
        best_model = best_est_a
        best_vectorizer_bundle = {
            "version": 2, "word_vectorizer": word_vec,
            "meta_feature_names": META_FEATURE_NAMES,
        }

    if run_track_b:
        from model.train_transformer import (
            train_transformer, build_transformer_vectorizer_bundle,
        )
        train_texts = base_train_df["message"].tolist()
        test_texts = base_test_df["message"].tolist()
        metrics_b, best_b, feats_b, best_wrapper = train_transformer(
            train_texts, test_texts, base_train_df, base_test_df, device=device,
        )
        all_metrics.extend(metrics_b)
        track_b_best = best_b
        track_b_features = feats_b

        if best_b is not None and best_wrapper is not None:
            if track_a_best is None or best_b.spam_f1 > track_a_best.spam_f1:
                best_model = best_wrapper
                best_vectorizer_bundle = build_transformer_vectorizer_bundle(
                    best_wrapper, META_FEATURE_NAMES,
                )
                print("\n  >> Transformer outperforms classical — deploying Track B model")

    if run_track_a and run_track_b:
        print_leaderboard(all_metrics, "CROSS-TRACK LEADERBOARD")
        print_cross_track_summary(track_a_best, track_b_best)
    elif run_track_a:
        from model.shared import print_leaderboard
        print_leaderboard(all_metrics, "TRACK A LEADERBOARD")

    if best_model is None:
        raise SystemExit("No model produced by any track.")

    print("\n  Retraining best model on full dataset...")
    full_train_df = base_dataframe.copy()
    full_train_df["label"] = full_train_df["label"].astype(int)
    y_all = full_train_df["label"].values

    if isinstance(best_vectorizer_bundle.get("model_type"), str) and \
       best_vectorizer_bundle["model_type"] == "transformer":
        from model.train_transformer import retrain_on_full_dataset, TRANSFORMER_MODELS
        winner_hf = TRANSFORMER_MODELS.get(best_model.model_name, "distilbert/distilbert-base-uncased")
        best_model = retrain_on_full_dataset(
            winner_hf, full_train_df["message"].tolist(), y_all, device=device,
        )
        best_vectorizer_bundle["tokenizer"] = best_model.tokenizer
        retrained_on_full = True
    else:
        from model.train_classical import create_word_vectorizer
        import scipy.sparse as sp
        import warnings
        from app.core.features import extract_meta_features
        final_vec = create_word_vectorizer(competition=competition)
        x_all_word = final_vec.fit_transform(full_train_df["processed"])
        x_all_meta = sp.csr_matrix(extract_meta_features(full_train_df["message"].tolist()))
        x_all = sp.hstack([x_all_word, x_all_meta], format="csr")
        sample_weight_all = full_train_df["sample_weight"].values
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            best_model.fit(x_all, y_all, sample_weight=sample_weight_all)
        best_vectorizer_bundle["word_vectorizer"] = final_vec
        retrained_on_full = True

    model_hash, vec_hash = save_artifacts(
        best_model, best_vectorizer_bundle, {}, MODEL_PATH, VECTORIZER_PATH, METADATA_PATH,
    )

    metadata = {
        "model_name": getattr(best_model, "model_name", type(best_model).__name__)
        if track_b_best else (track_a_best.model_name if track_a_best else "unknown"),
        "model_type": best_vectorizer_bundle.get("model_type", "classical"),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_mode": mode,
        "spam_threshold": DEFAULT_SPAM_THRESHOLD,
        "data_leakage_fixed": True,
        "retrained_on_full_dataset": retrained_on_full,
        "tracks_run": {
            "track_a": run_track_a,
            "track_b": run_track_b,
        },
        "track_a_features": track_a_features,
        "track_b_features": track_b_features,
        "dataset_rows": int(len(full_train_df)),
        "class_counts": {str(k): int(v) for k, v in full_train_df["label"].value_counts().to_dict().items()},
        "track_a_best": track_a_best.to_dict() if track_a_best else None,
        "track_b_best": track_b_best.to_dict() if track_b_best else None,
        "all_candidates": [m.to_dict() for m in all_metrics],
        "model_size_bytes": MODEL_PATH.stat().st_size,
        "vectorizer_size_bytes": VECTORIZER_PATH.stat().st_size,
    }

    model_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    vec_mb = VECTORIZER_PATH.stat().st_size / (1024 * 1024)

    print(f"\n  Model saved      -> {MODEL_PATH} ({model_mb:.1f} MB)")
    print(f"  Vectorizer saved  -> {VECTORIZER_PATH} ({vec_mb:.1f} MB)")
    print(f"  SHA256 model      -> {model_hash[:16]}...")
    print(f"  SHA256 vectorizer -> {vec_hash[:16]}...")
    print(f"  {ram_report('Final')}")

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nTraining complete.")
    return metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle", action="store_true")
    parser.add_argument("--track-a", dest="track_a", action="store_true")
    parser.add_argument("--track-b", dest="track_b", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.kaggle:
        run_a = True
        run_b = True
        comp = True
    else:
        run_a = args.track_a or (not args.track_b)
        run_b = args.track_b
        comp = False

    train(run_track_a=run_a, run_track_b=run_b, competition=comp, device=args.device)
