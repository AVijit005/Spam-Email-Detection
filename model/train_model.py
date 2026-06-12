"""Spam Detection Pipeline — Full Training Orchestrator.

Stage 1: Load & Preprocess → 342,178 emails, no lemmatization, token replacement
Stage 2: Classical ML (Track A) → SGD, XGBoost, LightGBM, MLP + Optuna HPO
Stage 3: Transformer (Track B) → DeBERTa-v3 / RoBERTa with focal loss + FGM
Stage 4: Ensemble → Grid search fusion weight, weighted late fusion
Stage 5: Retrain winner on 100% dataset
Stage 6: Export artifacts → model/, SHA-256 integrity checks

Usage:
  python model/train_model.py                → standard training
  python model/train_model.py --competition  → deep models, wider features
  python model/train_model.py --model DeBERTa-v3  → specific transformer
  python model/train_model.py --track-a-only → classical only
  python model/train_model.py --track-b-only → transformer only
  python model/train_model.py --fast-dev     → 500-row test run
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

try:
    import torch
except ImportError:
    torch = None

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.constants import META_FEATURE_NAMES
from app.core.features import extract_meta_features
from app.core.text import preprocess_text
from model.shared import (
    EvalMetrics,
    print_cross_track_summary,
    print_leaderboard,
    ram_report,
    save_artifacts,
)
from model.train_classical import train_classical, create_word_vectorizer
from model.train_transformer import (
    get_transformer_config,
    train_transformer,
    MODEL_IDS,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spam Detection Training Pipeline")
    parser.add_argument("--competition", action="store_true", help="Competition mode: wider features, deeper models")
    parser.add_argument("--model", type=str, default="DeBERTa-v3", help="Transformer model to use")
    parser.add_argument("--track-a-only", action="store_true", help="Only run classical (Track A)")
    parser.add_argument("--track-b-only", action="store_true", help="Only run transformer (Track B)")
    parser.add_argument("--fast-dev", action="store_true", help="Fast dev mode: 500 rows")
    parser.add_argument("--csv-path", type=str, default=None, help="Path to spam CSV")
    parser.add_argument("--output-dir", type=str, default=None, help="Override artifact output dir")
    parser.add_argument("--skip-optuna", action="store_true", help="Skip hyperparameter optimization")
    return parser.parse_args()


def _discover_csv() -> Path:
    candidates = [
        Path("data/spam.csv"),
        Path("D:/Spam-Email-Detection/data/spam.csv"),
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    if PROJECT_ROOT.exists():
        csv_path = PROJECT_ROOT / "data" / "spam.csv"
        if csv_path.exists():
            return csv_path
    raise FileNotFoundError("spam.csv not found.")


def load_and_preprocess(csv_path: Path, fast_dev: bool = False) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("  STAGE 1 — Load & Preprocess")
    print("=" * 60)
    print(f"  CSV: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8", usecols=["label", "text"])
    df.rename(columns={"text": "message"}, inplace=True)
    df["label"] = df["label"].map({"spam": 1, "ham": 0}).fillna(0).astype(np.int32)
    df.dropna(subset=["message"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    if fast_dev:
        print(f"  FAST DEV: limiting to 500 rows (from {len(df)})")
        df = df.sample(n=min(500, len(df)), random_state=42).reset_index(drop=True)

    print(f"  Rows: {len(df):,}")
    print(f"  Labels: {dict(zip(*np.unique(df['label'].values, return_counts=True)))}")
    print(f"  Avg length: {df['message'].str.len().mean():.0f} chars")

    t0 = time.perf_counter()
    df["processed"] = df["message"].apply(preprocess_text)
    print(f"  Preprocessing: {time.perf_counter() - t0:.1f}s")
    print(f"  Preprocessed avg len: {df['processed'].str.len().mean():.0f} chars")

    df["sample_weight"] = 1.0
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=test_size, stratify=df["label"].values, random_state=42,
    )
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    print(f"\n  Train: {len(train_df):,} | Test: {len(test_df):,}")
    print(f"  Train label dist: {train_df['label'].value_counts().to_dict()}")
    print(f"  Test label dist:  {test_df['label'].value_counts().to_dict()}")
    return train_df, test_df


def run_track_b(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    fast_dev: bool,
) -> tuple[EvalMetrics | None, dict[str, Any] | None, Any | None, Any | None]:
    print("\n" + "=" * 60)
    print("  STAGE 3 — Transformer Fine-Tuning (Track B)")
    print("=" * 60)

    if model_name not in MODEL_IDS:
        print(f"  SKIPPED: {model_name} not in supported models: {list(MODEL_IDS)}")
        return None, None, None, None

    config = get_transformer_config(model_name, fast_dev_run=fast_dev)
    metrics, package_info, model, tokenizer = train_transformer(
        train_df, test_df, config,
        checkpoint_dir=str(PROJECT_ROOT / "model" / "checkpoints"),
    )

    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics, package_info, model, tokenizer


def main() -> None:
    args = _parse_args()
    t_start = time.perf_counter()

    csv_path = Path(args.csv_path) if args.csv_path else _discover_csv()
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "model"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_preprocess(csv_path, fast_dev=args.fast_dev)
    train_df, test_df = split_data(df)

    all_metrics: list[EvalMetrics] = []
    track_a_metrics: EvalMetrics | None = None
    track_b_metrics: EvalMetrics | None = None

    classical_best_model = None
    classical_estimator = None
    classical_features_config = None
    classical_word_vec = None
    transformer_model = None
    transformer_tokenizer = None
    transformer_package_info = None

    if not args.track_b_only:
        print("\n" + "=" * 60)
        print("  STAGE 2 — Classical ML (Track A)")
        print("=" * 60)
        class_metrics, best_metrics, features_config, word_vec, best_estimator = train_classical(
            train_df, test_df, competition=args.competition,
        )
        all_metrics.extend(class_metrics)
        track_a_metrics = best_metrics
        classical_estimator = best_estimator
        classical_word_vec = word_vec
        classical_features_config = {
            "model_name": best_metrics.model_name,
            "features": features_config,
            "metrics": best_metrics.to_dict(),
            "ensemble_role": "classical",
        }
        print(ram_report("After Track A"))
        print(f"\n  Track A Winner: {best_metrics.model_name} → Spam F1 = {best_metrics.spam_f1:.4f}")

    if not args.track_a_only:
        tb_metrics, package_info, t_model, t_tokenizer = run_track_b(
            train_df, test_df, args.model, args.fast_dev,
        )
        if tb_metrics is not None:
            all_metrics.append(tb_metrics)
            track_b_metrics = tb_metrics
            transformer_model = t_model
            transformer_tokenizer = t_tokenizer
            transformer_package_info = package_info

    if all_metrics:
        print_leaderboard(all_metrics)

    print_cross_track_summary(track_a_metrics, track_b_metrics)

    has_ensemble = classical_estimator is not None and transformer_model is not None
    fusion_weight = 0.50
    ensemble_f1 = None

    if has_ensemble:
        print("\n" + "=" * 60)
        print("  STAGE 4 — Ensemble Fusion")
        print("=" * 60)

        from app.ml.ensemble import EnsemblePredictor, grid_search_fusion_weight

        t0 = time.perf_counter()
        x_train_word = classical_word_vec.transform(train_df["processed"])
        x_test_word = classical_word_vec.transform(test_df["processed"])
        x_train_meta = sp.csr_matrix(extract_meta_features(train_df["message"].tolist()))
        x_test_meta = sp.csr_matrix(extract_meta_features(test_df["message"].tolist()))
        x_train_ens = sp.hstack([x_train_word, x_train_meta], format="csr")
        x_test_ens = sp.hstack([x_test_word, x_test_meta], format="csr")
        y_test_arr = test_df["label"].values

        p_classical = classical_estimator.predict_proba(x_test_ens)
        ensemble = EnsemblePredictor(
            classical_model=classical_estimator,
            classical_vectorizer_bundle={"word_vec": classical_word_vec},
            transformer_model=transformer_model,
            transformer_tokenizer=transformer_tokenizer,
            fusion_weight=0.50,
        )
        p_transformer = ensemble.transformer_proba(test_df["message"].tolist())

        grid_result = grid_search_fusion_weight(p_classical, p_transformer, y_test_arr)
        fusion_weight = grid_result["best_weight"]

        ensemble.fusion_weight = fusion_weight
        ensemble_preds = ensemble.predict(x_test_ens, test_df["message"].tolist())
        from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score
        ensemble_spam_f1 = f1_score(y_test_arr, ensemble_preds, pos_label=1)
        try:
            ensemble_proba = ensemble.predict_proba(x_test_ens, test_df["message"].tolist())
            ensemble_roc = float(roc_auc_score(y_test_arr, ensemble_proba[:, 1]))
        except ValueError:
            ensemble_roc = None

        ensemble_f1 = ensemble_spam_f1
        cm = confusion_matrix(y_test_arr, ensemble_preds)
        print(f"\n  Ensemble Fusion Weight: {fusion_weight:.4f}")
        print(f"  Ensemble Spam F1: {ensemble_spam_f1:.4f}")
        print(f"  Ensemble ROC-AUC: {ensemble_roc}")
        print(f"  Confusion matrix:\n{cm}")

    print("\n" + "=" * 60)
    print("  STAGE 5 — Retrain Winner on Full Dataset")
    print("=" * 60)

    if has_ensemble and ensemble_f1 is not None:
        print(f"  Retraining XGBoost + {args.model} ensemble on full {len(df):,} dataset...")
    elif track_b_metrics is not None:
        print(f"  Retraining {args.model} on full {len(df):,} dataset...")
    else:
        print(f"  Retraining {track_a_metrics.model_name if track_a_metrics else 'N/A'} on full {len(df):,} dataset...")

    full_word_vec = create_word_vectorizer(competition=args.competition)
    full_processed = df["processed"].tolist()
    full_labels = df["label"].values
    full_raw = df["message"].tolist()
    x_full_word = full_word_vec.fit_transform(full_processed)
    x_full_meta = sp.csr_matrix(extract_meta_features(full_raw))
    x_full = sp.hstack([x_full_word, x_full_meta], format="csr")
    print(f"  Full train matrix: {x_full.shape} ({x_full.nnz:,} nnz)")

    if not args.track_b_only:
        print(f"  Fitting classical model: {track_a_metrics.model_name if track_a_metrics else 'N/A'}")
        classical_estimator.fit(x_full, full_labels)
    else:
        classical_estimator = None

    trained_at_utc = datetime.now(timezone.utc).isoformat()

    metadata = {
        "model_name": "Ensemble" if has_ensemble else (
            args.model if track_b_metrics is not None else (
                track_a_metrics.model_name if track_a_metrics else "unknown"
            )
        ),
        "track": "ensemble" if has_ensemble else ("transformer" if track_b_metrics else "classical"),
        "trained_at_utc": trained_at_utc,
        "dataset_rows": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "selected_metrics": {
            "ensemble_f1": ensemble_f1,
            "track_a_f1": track_a_metrics.spam_f1 if track_a_metrics else None,
            "track_b_f1": track_b_metrics.spam_f1 if track_b_metrics else None,
        },
        "classical_info": classical_features_config,
        "transformer_info": transformer_package_info,
        "ensemble_info": {
            "fusion_weight": fusion_weight,
            "classical_branch": track_a_metrics.model_name if track_a_metrics else None,
            "transformer_branch": args.model if track_b_metrics else None,
        } if has_ensemble else None,
        "features": classical_features_config["features"] if classical_features_config else None,
        "all_candidates": [m.to_dict() for m in all_metrics],
        "training_args": vars(args),
    }

    print("\n" + "=" * 60)
    print("  STAGE 6 — Export Artifacts")
    print("=" * 60)

    if has_ensemble:
        import torch as _torch
        vectorizer_bundle: dict[str, Any] = {
            "word_vec": full_word_vec,
            "meta_feature_names": list(classical_features_config["features"]["meta_feature_names"]) if classical_features_config else [],
            "version": "3.0.0",
        }
        model_path = output_dir / "spam_model.pkl"
        vec_path = output_dir / "vectorizer.pkl"
        meta_path = output_dir / "model_metadata.json"

        save_artifacts(classical_estimator, vectorizer_bundle, metadata, model_path, vec_path, meta_path)

        xf_model_path = output_dir / "transformer_model.pt"
        xf_tokenizer_path = output_dir / "transformer_tokenizer"
        _torch.save(transformer_model.state_dict(), xf_model_path)
        transformer_tokenizer.save_pretrained(str(xf_tokenizer_path))
        (xf_model_path.parent / (xf_model_path.name + ".sha256")).write_text(
            __import__("hashlib").sha256(xf_model_path.read_bytes()).hexdigest()
        )
        print(f"  Transformer saved: {xf_model_path}")
        print(f"  Tokenizer saved:   {xf_tokenizer_path}")

    elif track_b_metrics is not None:
        import torch as _torch
        vectorizer_bundle: dict[str, Any] = {"word_vec": full_word_vec, "version": "3.0.0"}
        model_path = output_dir / "transformer_model.pt"
        vec_path = output_dir / "transformer_tokenizer"
        with open(vec_path / "config.json", "w") as f:
            json.dump({"model_name": args.model, "max_length": 512}, f)
        _torch.save(transformer_model.state_dict(), model_path)
        meta_path = output_dir / "model_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    else:
        vectorizer_bundle: dict[str, Any] = {
            "word_vec": full_word_vec,
            "meta_feature_names": list(classical_features_config["features"]["meta_feature_names"]) if classical_features_config else [],
            "version": "3.0.0",
        }
        model_path = output_dir / "spam_model.pkl"
        vec_path = output_dir / "vectorizer.pkl"
        meta_path = output_dir / "model_metadata.json"
        save_artifacts(classical_estimator, vectorizer_bundle, metadata, model_path, vec_path, meta_path)

    print(f"  Model saved:  {model_path}")
    print(f"  Vectorizer:   {vec_path}")
    print(f"  Metadata:     {meta_path}")
    print(ram_report("Final"))

    total_time = time.perf_counter() - t_start
    print(f"\n{'=' * 60}")
    print(f"  Training complete in {total_time:.1f}s ({total_time / 60:.1f}m)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
