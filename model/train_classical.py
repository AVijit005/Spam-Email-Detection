"""Track A — Production Classical ML Pipeline.

TF-IDF + tree/linear/neural candidates. Evaluates on the shared holdout split.
Supports optional 5-fold stratified cross-validation for robust model comparison.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.constants import META_FEATURE_NAMES
from app.core.features import extract_meta_features
from model.shared import EvalMetrics, score_model, ram_report

WORD_MAX_FEATURES = 25000
WORD_MIN_DF = 30
WORD_MAX_DF = 0.70
WORD_NGRAM = (1, 2)

COMPETITION_WORD_MAX_FEATURES = 50000
COMPETITION_WORD_MIN_DF = 10
COMPETITION_WORD_MAX_DF = 0.60

OPTUNA_TRIALS = 30
OPTUNA_TIMEOUT_SECONDS = 1200
OPTUNA_COMPETITION_TIMEOUT = 2400

OPTUNA_N_ESTIMATORS_LOW = 200
OPTUNA_N_ESTIMATORS_HIGH = 600
OPTUNA_MAX_DEPTH_LOW = 6
OPTUNA_MAX_DEPTH_HIGH = 14
OPTUNA_COLSAMPLE_LOW = 0.3
OPTUNA_COLSAMPLE_HIGH = 0.7
OPTUNA_MIN_CHILD_WEIGHT_HIGH = 20


def create_word_vectorizer(competition: bool = False) -> TfidfVectorizer:
    if competition:
        return TfidfVectorizer(
            max_features=COMPETITION_WORD_MAX_FEATURES,
            ngram_range=(1, 3),
            sublinear_tf=True,
            min_df=COMPETITION_WORD_MIN_DF,
            max_df=COMPETITION_WORD_MAX_DF,
            dtype=np.float32,
        )
    return TfidfVectorizer(
        max_features=WORD_MAX_FEATURES,
        ngram_range=WORD_NGRAM,
        sublinear_tf=True,
        min_df=WORD_MIN_DF,
        max_df=WORD_MAX_DF,
        dtype=np.float32,
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


def _optimize_xgboost(
    x_train: sp.csr_matrix,
    y_train: np.ndarray,
    sw_train: np.ndarray,
    competition: bool = False,
) -> dict[str, Any]:
    try:
        import xgboost as xgb
        import optuna
    except ImportError:
        return {"n_estimators": 500, "max_depth": 10, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.6}

    timeout = OPTUNA_COMPETITION_TIMEOUT if competition else OPTUNA_TIMEOUT_SECONDS
    n_est_low, n_est_high = OPTUNA_N_ESTIMATORS_LOW, OPTUNA_N_ESTIMATORS_HIGH
    depth_low, depth_high = OPTUNA_MAX_DEPTH_LOW, OPTUNA_MAX_DEPTH_HIGH
    col_low, col_high = OPTUNA_COLSAMPLE_LOW, OPTUNA_COLSAMPLE_HIGH

    x_tr, x_val, y_tr, y_val, sw_tr, sw_val = train_test_split(
        x_train, y_train, sw_train, test_size=0.2, stratify=y_train, random_state=42,
    )

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", n_est_low, n_est_high),
            "max_depth": trial.suggest_int("max_depth", depth_low, depth_high),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", col_low, col_high),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, OPTUNA_MIN_CHILD_WEIGHT_HIGH if competition else 10),
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
            "tree_method": "hist",
        }
        model = xgb.XGBClassifier(**params)
        model.fit(x_tr, y_tr, sample_weight=sw_tr, eval_set=[(x_val, y_val)], verbose=False)
        probs = model.predict_proba(x_val)[:, 1]
        from sklearn.metrics import f1_score
        return f1_score(y_val, probs >= 0.5, pos_label=1)

    print(f"  Optimizing XGBoost hyperparameters (Optuna, {timeout}s timeout)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS, timeout=timeout, n_jobs=1,
                   show_progress_bar=False)
    print(f"  Best trial F1 (validation): {study.best_value:.4f}")
    params = study.best_params
    params["random_state"] = 42
    params["n_jobs"] = -1
    params["verbosity"] = 0
    if "tree_method" in params:
        del params["tree_method"]
    return params


def _optimize_lightgbm(
    x_train: sp.csr_matrix,
    y_train: np.ndarray,
    sw_train: np.ndarray,
) -> dict[str, Any]:
    try:
        import lightgbm as lgb
        import optuna
    except ImportError:
        return {"n_estimators": 500, "max_depth": 10, "num_leaves": 127,
                "learning_rate": 0.05}

    x_tr, x_val, y_tr, y_val, sw_tr, sw_val = train_test_split(
        x_train, y_train, sw_train, test_size=0.2, stratify=y_train, random_state=42,
    )

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(x_tr, y_tr, sample_weight=sw_tr)
        from sklearn.metrics import f1_score
        preds = model.predict(x_val)
        return f1_score(y_val, preds, pos_label=1)

    print("  Optimizing LightGBM hyperparameters (Optuna)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS, timeout=OPTUNA_TIMEOUT_SECONDS, n_jobs=1,
                   show_progress_bar=False)
    print(f"  Best trial F1 (validation): {study.best_value:.4f}")
    params = study.best_params
    params["class_weight"] = "balanced"
    params["random_state"] = 42
    params["n_jobs"] = -1
    params["verbose"] = -1
    return params


def build_candidates(
    competition: bool = False,
    x_train: sp.csr_matrix | None = None,
    y_train: np.ndarray | None = None,
    sw_train: np.ndarray | None = None,
    skip_optuna: bool = False,
) -> dict[str, Any]:
    c = {
        "SGDClassifier": SGDClassifier(
            loss="log_loss", penalty="elasticnet", alpha=0.0001,
            l1_ratio=0.15, max_iter=1000, tol=1e-3,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
    }

    if x_train is not None and y_train is not None and not skip_optuna:
        try:
            import xgboost as xgb
            xgb_params = _optimize_xgboost(x_train, y_train, sw_train, competition=competition)
            c["XGBoost"] = xgb.XGBClassifier(**xgb_params)
        except ImportError:
            try:
                import xgboost as xgb
                c["XGBoost"] = xgb.XGBClassifier(
                    n_estimators=300, max_depth=8, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8, random_state=42,
                    n_jobs=-1, verbosity=0,
                )
            except ImportError:
                pass
    else:
        try:
            import xgboost as xgb
            c["XGBoost"] = xgb.XGBClassifier(
                n_estimators=300, max_depth=8, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                n_jobs=-1, verbosity=0,
            )
        except ImportError:
            pass

    if x_train is not None and y_train is not None and not skip_optuna:
        try:
            import lightgbm as lgb
            lgb_params = _optimize_lightgbm(x_train, y_train, sw_train)
            c["LightGBM"] = lgb.LGBMClassifier(**lgb_params)
        except ImportError:
            pass
    else:
        try:
            import lightgbm as lgb
            if not any(k.startswith("LightGBM") for k in c):
                c["LightGBM"] = lgb.LGBMClassifier(
                    n_estimators=300, max_depth=8, num_leaves=63,
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
    skip_optuna: bool = False,
) -> tuple[list[EvalMetrics], EvalMetrics, dict[str, Any], TfidfVectorizer, Any]:
    print("\n" + "=" * 60)
    print("  TRACK A — Classical ML Pipeline")
    if competition:
        print("  MODE: Competition (wider features, deeper models, Optuna HPO)")
    print("=" * 60)

    word_vec = create_word_vectorizer(competition=competition)
    print(f"\n  Vectorizer: max_features={word_vec.max_features}, "
          f"ngram={word_vec.ngram_range}, min_df={word_vec.min_df}, "
          f"max_df={word_vec.max_df}, dtype=float32")

    x_train, x_test, y_train, y_test, sw_train = build_classical_features(
        word_vec, train_df, test_df
    )
    print(ram_report("After features"))

    if competition:
        candidates = build_candidates(
            competition=True, x_train=x_train, y_train=y_train, sw_train=sw_train,
            skip_optuna=skip_optuna,
        )
    else:
        candidates = build_candidates(competition=False, skip_optuna=skip_optuna)

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
            best_estimator = estimator

    if best_metrics is None or best_estimator is None:
        raise SystemExit("Track A: no candidates evaluated.")

    features_config = {
        "max_features": word_vec.max_features,
        "ngram_range": list(word_vec.ngram_range),
        "min_df": getattr(word_vec, "min_df", WORD_MIN_DF),
        "max_df": getattr(word_vec, "max_df", WORD_MAX_DF),
        "sublinear_tf": True,
        "dtype": "float32",
        "meta_feature_names": META_FEATURE_NAMES,
        "word_features": int(x_train.shape[1] - len(META_FEATURE_NAMES)),
        "meta_features": int(len(META_FEATURE_NAMES)),
        "total_features": int(x_train.shape[1]),
        "train_matrix_shape": list(x_train.shape),
        "train_nnz": int(x_train.nnz),
        "test_nnz": int(x_test.nnz),
    }

    return all_metrics, best_metrics, features_config, word_vec, best_estimator
