"""Track A — Production Classical ML Pipeline.

TF-IDF + Linear/Tree/NN candidates. Evaluates on the shared holdout split.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.constants import META_FEATURE_NAMES
from app.core.features import extract_meta_features
from model.shared import EvalMetrics, score_model, ram_report

WORD_MAX_FEATURES = 35000
WORD_MIN_DF = 20
WORD_MAX_DF = 0.70
WORD_NGRAM = (1, 2)


COMPETITION_WORD_MAX_FEATURES = 50000
COMPETITION_WORD_MIN_DF = 10
COMPETITION_WORD_MAX_DF = 0.60


def create_word_vectorizer(competition: bool = False) -> TfidfVectorizer:
    if competition:
        return TfidfVectorizer(
            max_features=COMPETITION_WORD_MAX_FEATURES,
            ngram_range=(1, 3),
            sublinear_tf=True,
            min_df=COMPETITION_WORD_MIN_DF,
            max_df=COMPETITION_WORD_MAX_DF,
        )
    return TfidfVectorizer(
        max_features=WORD_MAX_FEATURES,
        ngram_range=WORD_NGRAM,
        sublinear_tf=True,
        min_df=WORD_MIN_DF,
        max_df=WORD_MAX_DF,
    )


def build_classical_features(
    word_vec: TfidfVectorizer,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    x_train_word = word_vec.fit_transform(train_df["processed"])
    x_test_word = word_vec.transform(test_df["processed"])
    x_train_meta = sp.csr_matrix(extract_meta_features(train_df["message"].tolist()))
    x_test_meta = sp.csr_matrix(extract_meta_features(test_df["message"].tolist()))
    x_train = sp.hstack([x_train_word, x_train_meta], format="csr")
    x_test = sp.hstack([x_test_word, x_test_meta], format="csr")
    y_train = train_df["label"].values
    y_test = test_df["label"].values
    sample_weight_train = train_df["sample_weight"].values

    print(f"  Train matrix : {x_train.shape} ({x_train.nnz:,} nnz)")
    print(f"  Test matrix  : {x_test.shape} ({x_test.nnz:,} nnz)")
    print(f"  Features     : word={x_train_word.shape[1]}, meta={x_train_meta.shape[1]}")
    print(f"  Sparse mem   : ~{(x_train.nnz + x_test.nnz) * 12 / (1024**2):.0f} MB")

    return x_train, x_test, y_train, y_test, sample_weight_train


def build_candidates(competition: bool = False) -> dict[str, Any]:
    c = {
        "LogisticRegression": LogisticRegression(
            C=1.0, max_iter=5000, class_weight="balanced",
            solver="saga", penalty="elasticnet", l1_ratio=0.15,
            tol=1e-4, random_state=42, n_jobs=-1,
        ),
        "ComplementNB": ComplementNB(alpha=0.1, norm=False),
        "SGDClassifier_log_loss": SGDClassifier(
            loss="log_loss", penalty="elasticnet", max_iter=2000,
            tol=1e-3, class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "SGDClassifier_hinge": SGDClassifier(
            loss="hinge", penalty="elasticnet", max_iter=2000,
            tol=1e-3, class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "LinearSVC": LinearSVC(
            C=1.0, max_iter=5000, class_weight="balanced",
            dual=False, tol=1e-4, random_state=42,
        ),
        "MLPClassifier": MLPClassifier(
            hidden_layer_sizes=(128, 32), activation="relu",
            solver="adam", max_iter=200, early_stopping=True,
            validation_fraction=0.1, random_state=42,
        ),
    }

    if competition:
        c["MLPClassifier_deep"] = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), activation="relu",
            solver="adam", max_iter=300, early_stopping=True,
            validation_fraction=0.1, random_state=42,
        )

    try:
        import xgboost as xgb
        c["XGBoost"] = xgb.XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            n_jobs=-1, verbosity=0,
        )
        if competition:
            c["XGBoost_large"] = xgb.XGBClassifier(
                n_estimators=500, max_depth=10, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.6, random_state=42,
                n_jobs=-1, verbosity=0,
            )
    except ImportError:
        pass

    try:
        import lightgbm as lgb
        c["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=300, max_depth=10, num_leaves=63,
            learning_rate=0.1, class_weight="balanced",
            random_state=42, n_jobs=-1, verbose=-1,
        )
    except ImportError:
        pass

    return c


def train_classical(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    competition: bool = False,
) -> tuple[list[EvalMetrics], EvalMetrics, dict[str, Any], TfidfVectorizer, Any]:
    print("\n" + "=" * 60)
    print("  TRACK A — Classical ML Pipeline")
    if competition:
        print("  MODE: Competition (wider features, deeper models)")
    print("=" * 60)

    word_vec = create_word_vectorizer(competition=competition)
    print(f"\n  Vectorizer: max_features={word_vec.max_features}, "
          f"ngram={word_vec.ngram_range}, min_df={word_vec.min_df}, "
          f"max_df={word_vec.max_df}")

    x_train, x_test, y_train, y_test, sw_train = build_classical_features(
        word_vec, train_df, test_df
    )
    print(ram_report("After features"))

    candidates = build_candidates(competition=competition)
    print(f"\n  Evaluating {len(candidates)} candidates...")

    all_metrics: list[EvalMetrics] = []
    best_metrics: EvalMetrics | None = None
    best_estimator = None

    for idx, (name, estimator) in enumerate(candidates.items(), 1):
        print(f"\n  [{idx}/{len(candidates)}] {name}")
        met = score_model(name, "classical", estimator, x_train, x_test, y_train, y_test, sw_train)
        all_metrics.append(met)
        print(f"  {ram_report('')}")

        if best_metrics is None or (met.spam_f1, met.spam_recall, met.accuracy) > (
            best_metrics.spam_f1, best_metrics.spam_recall, best_metrics.accuracy,
        ):
            best_metrics = met
            best_estimator = clone(estimator)

    if best_metrics is None or best_estimator is None:
        raise SystemExit("Track A: no candidates evaluated.")

    features_config = {
        "max_features": word_vec.max_features,
        "ngram_range": list(word_vec.ngram_range),
        "min_df": getattr(word_vec, "min_df", WORD_MIN_DF),
        "max_df": getattr(word_vec, "max_df", WORD_MAX_DF),
        "sublinear_tf": True,
        "meta_feature_names": META_FEATURE_NAMES,
        "word_features": int(x_train.shape[1] - len(META_FEATURE_NAMES)),
        "meta_features": int(len(META_FEATURE_NAMES)),
        "total_features": int(x_train.shape[1]),
        "train_matrix_shape": list(x_train.shape),
        "train_nnz": int(x_train.nnz),
        "test_nnz": int(x_test.nnz),
    }

    return all_metrics, best_metrics, features_config, word_vec, best_estimator
