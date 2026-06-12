"""Shared training infrastructure — dataset, holdout, evaluation, artifact I/O.

Used by both Track A (classical ML) and Track B (transformers) to ensure
identical data splits, identical metrics, and artifact compatibility.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

MODEL_DIR = Path(__file__).resolve().parent


def _ram_usage_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return -1.0


def ram_report(label: str) -> str:
    mb = _ram_usage_mb()
    return f"{label}: RAM {mb:.0f} MB" if mb >= 0 else f"{label}: RAM N/A"


@dataclass
class EvalMetrics:
    model_name: str
    accuracy: float
    spam_precision: float
    spam_recall: float
    spam_f1: float
    roc_auc: float | None
    train_time_seconds: float
    support: int
    track: str
    confusion_matrix: list[list[int]] | None = None
    model_size_bytes: int = 0
    converged: bool = True
    effective_iterations: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "track": self.track,
            "model_name": self.model_name,
            "accuracy": round(self.accuracy, 4),
            "spam_precision": round(self.spam_precision, 4),
            "spam_recall": round(self.spam_recall, 4),
            "spam_f1": round(self.spam_f1, 4),
            "roc_auc": round(self.roc_auc, 4) if self.roc_auc is not None else None,
            "train_time_seconds": round(self.train_time_seconds, 1),
            "support": self.support,
            "converged": self.converged,
            "effective_iterations": self.effective_iterations,
            "model_size_bytes": self.model_size_bytes,
        }


def score_model(
    name: str,
    track: str,
    estimator: Any,
    x_train: Any,
    x_test: Any,
    y_train: np.ndarray,
    y_test: np.ndarray,
    sample_weight_train: np.ndarray | None = None,
) -> EvalMetrics:
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        try:
            estimator.fit(x_train, y_train, sample_weight=sample_weight_train)
        except TypeError:
            estimator.fit(x_train, y_train)
    train_time = time.perf_counter() - t0

    predictions = estimator.predict(x_test)
    report = classification_report(
        y_test, predictions, target_names=["Ham", "Spam"],
        output_dict=True, zero_division=0,
    )

    try:
        if hasattr(estimator, "predict_proba"):
            probs = estimator.predict_proba(x_test)[:, 1]
        else:
            probs = estimator.decision_function(x_test)
        roc_auc = float(roc_auc_score(y_test, probs))
    except (AttributeError, ValueError):
        roc_auc = None

    n_iter = None
    converged = True
    if hasattr(estimator, "n_iter_"):
        val = estimator.n_iter_
        n_iter = int(val[0]) if hasattr(val, "__len__") else int(val)

    spam_metrics = report["Spam"]
    cm = confusion_matrix(y_test, predictions)

    metrics = EvalMetrics(
        model_name=name,
        track=track,
        accuracy=float(report["accuracy"]),
        spam_precision=float(spam_metrics["precision"]),
        spam_recall=float(spam_metrics["recall"]),
        spam_f1=float(spam_metrics["f1-score"]),
        roc_auc=roc_auc,
        train_time_seconds=train_time,
        support=int(spam_metrics["support"]),
        confusion_matrix=cm.tolist(),
        converged=n_iter is not None or track == "transformer",
        effective_iterations=n_iter,
    )

    print(f"\n--- [{track}] {name} ---")
    print(f"Accuracy        : {metrics.accuracy:.4f}")
    print(f"Spam F1         : {metrics.spam_f1:.4f}")
    print(f"Spam Precision  : {metrics.spam_precision:.4f}")
    print(f"Spam Recall     : {metrics.spam_recall:.4f}")
    print(f"ROC-AUC         : {metrics.roc_auc}")
    print(f"Train time      : {metrics.train_time_seconds:.1f}s")
    print("Confusion matrix:")
    print(cm)

    return metrics


def print_leaderboard(all_metrics: list[EvalMetrics], title: str = "LEADERBOARD") -> None:
    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")
    header = f"{'Track':<14s} {'Model':<26s} {'F1':>7s} {'Prec':>7s} {'Recall':>7s} {'ROC-AUC':>8s} {'Time':>8s} {'Size':>7s}"
    print(header)
    print("-" * 90)

    sorted_metrics = sorted(all_metrics, key=lambda e: e.spam_f1, reverse=True)
    best_overall_f1 = sorted_metrics[0].spam_f1 if sorted_metrics else 0.0

    for m in sorted_metrics:
        f1_s = f"{m.spam_f1:.4f}"
        prec_s = f"{m.spam_precision:.4f}"
        rec_s = f"{m.spam_recall:.4f}"
        roc_s = f"{m.roc_auc:.4f}" if m.roc_auc else "N/A"
        t_s = f"{m.train_time_seconds:.0f}s"
        sz_s = f"{m.model_size_bytes / 1024:.0f}KB" if m.model_size_bytes else "?"
        mark = ">" if m.spam_f1 == best_overall_f1 else " "
        track_icon = "A" if m.track == "classical" else "B"
        print(f"{mark}Track {track_icon:<10s} {m.model_name:<26s} {f1_s:>7s} {prec_s:>7s} {rec_s:>7s} {roc_s:>8s} {t_s:>8s} {sz_s:>7s}")
    print("=" * 90)


def print_cross_track_summary(track_a: EvalMetrics | None, track_b: EvalMetrics | None) -> None:
    print(f"\n{'=' * 90}")
    print("  CROSS-TRACK COMPARISON")
    print(f"{'=' * 90}")
    if track_a:
        print(f"  Track A Best (Classical): {track_a.model_name} → Spam F1 = {track_a.spam_f1:.4f}  @ {track_a.train_time_seconds:.0f}s")
    if track_b:
        print(f"  Track B Best (Transformer): {track_b.model_name} → Spam F1 = {track_b.spam_f1:.4f}  @ {track_b.train_time_seconds:.0f}s")
    if track_a and track_b:
        delta = track_b.spam_f1 - track_a.spam_f1
        print(f"  F1 Delta (B - A): {delta:+.4f}")
        if track_b.model_size_bytes and track_a.model_size_bytes:
            size_ratio = track_b.model_size_bytes / max(track_a.model_size_bytes, 1)
            print(f"  Size ratio (B / A): {size_ratio:.1f}x")
    print("=" * 90)


def save_artifacts(
    model: Any,
    vectorizer_bundle: dict[str, Any],
    metadata: dict[str, Any],
    model_path: Path,
    vectorizer_path: Path,
    metadata_path: Path,
) -> tuple[str, str]:
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer_bundle, f)

    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    vec_hash = hashlib.sha256(vectorizer_path.read_bytes()).hexdigest()

    (model_path.parent / (model_path.name + ".sha256")).write_text(model_hash)
    (vectorizer_path.parent / (vectorizer_path.name + ".sha256")).write_text(vec_hash)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return model_hash, vec_hash
