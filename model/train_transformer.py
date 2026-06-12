"""Track B — Maximum-Performance Transformer Pipeline.

Fine-tunes DistilBERT, RoBERTa, and DeBERTa-v3 on the same holdout split.
Produces a TransformerWrapper compatible with the prediction API.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.constants import META_FEATURE_NAMES
from app.core.features import compose_email_text, extract_meta_features
from model.shared import EvalMetrics, score_model, ram_report

TRANSFORMER_MODELS = {
    "DistilBERT": "distilbert/distilbert-base-uncased",
    "RoBERTa": "roberta-base",
    "DeBERTa-v3": "microsoft/deberta-v3-base",
}

MAX_LENGTH = 512
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01


class TransformerWrapper(BaseEstimator, ClassifierMixin):
    """sklearn-compatible wrapper that delegates prediction to a HF model."""

    def __init__(self, model: Any = None, tokenizer: Any = None,
                 model_name: str = "", device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = device

    def fit(self, X, y):
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        probs = self.predict_proba(texts)
        return (probs[:, 1] >= 0.5).astype(int)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        import torch
        results = []
        tokenizer = self.tokenizer
        model = self.model
        device = next(model.parameters()).device

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            enc = tokenizer(
                batch, truncation=True, padding=True,
                max_length=MAX_LENGTH, return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits
                batch_probs = torch.softmax(logits, dim=-1).cpu().numpy()
            results.append(batch_probs)
        return np.vstack(results)

    def __sklearn_is_fitted__(self):
        return self.model is not None and self.tokenizer is not None


def _build_transformer_features(
    train_texts: list[str],
    test_texts: list[str],
) -> tuple[list[str], list[str]]:
    return train_texts, test_texts


def _train_single_transformer(
    hf_name: str,
    model_name: str,
    train_texts: list[str],
    train_labels: np.ndarray,
    test_texts: list[str],
    test_labels: np.ndarray,
    device: str,
) -> tuple[EvalMetrics, TransformerWrapper]:
    import torch
    from torch.utils.data import DataLoader, Dataset, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers import get_linear_schedule_with_warmup, AdamW

    print(f"\n  Loading {model_name}: {hf_name}")

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_name, num_labels=2,
    )
    model.to(device)

    print(f"  Tokenizing {len(train_texts):,} texts...")
    train_enc = tokenizer(
        train_texts, truncation=True, padding=True,
        max_length=MAX_LENGTH, return_tensors="pt",
    )
    train_dataset = TensorDataset(
        train_enc["input_ids"], train_enc["attention_mask"],
        torch.tensor(train_labels, dtype=torch.long),
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    print(f"  Fine-tuning {EPOCHS} epochs, {total_steps} steps...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            if (step + 1) % 500 == 0:
                print(f"    Epoch {epoch+1}/{EPOCHS}, Step {step+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        print(f"    Epoch {epoch+1}/{EPOCHS} done, avg loss: {total_loss / len(train_loader):.4f}")
        print(f"    {ram_report('')}")

    model.eval()
    wrapper = TransformerWrapper(model=model, tokenizer=tokenizer, model_name=model_name, device=device)

    import time as _time
    t0 = _time.perf_counter()
    all_preds = []
    for i in range(0, len(test_texts), BATCH_SIZE):
        batch = test_texts[i:i + BATCH_SIZE]
        preds = wrapper.predict(batch)
        all_preds.append(preds)
    predictions = np.concatenate(all_preds)
    train_time = _time.perf_counter() - t0

    met = EvalMetrics(
        model_name=model_name,
        track="transformer",
        accuracy=float((predictions == test_labels).mean()),
        spam_precision=float(0.0),
        spam_recall=float(0.0),
        spam_f1=float(0.0),
        roc_auc=None,
        train_time_seconds=train_time,
        support=int(test_labels.sum()),
    )

    from sklearn.metrics import (
        accuracy_score, classification_report, roc_auc_score, confusion_matrix,
    )
    report = classification_report(
        test_labels, predictions, target_names=["Ham", "Spam"],
        output_dict=True, zero_division=0,
    )
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

    probs = wrapper.predict_proba(test_texts)[:, 1]
    cm = confusion_matrix(test_labels, predictions)

    met.accuracy = float(accuracy_score(test_labels, predictions))
    met.spam_precision = float(report["Spam"]["precision"])
    met.spam_recall = float(report["Spam"]["recall"])
    met.spam_f1 = float(report["Spam"]["f1-score"])
    met.roc_auc = float(roc_auc_score(test_labels, probs))
    met.confusion_matrix = cm.tolist()

    print(f"\n--- [transformer] {model_name} ---")
    print(f"Accuracy        : {met.accuracy:.4f}")
    print(f"Spam F1         : {met.spam_f1:.4f}")
    print(f"Spam Precision  : {met.spam_precision:.4f}")
    print(f"Spam Recall     : {met.spam_recall:.4f}")
    print(f"ROC-AUC         : {met.roc_auc:.4f}")
    print(f"Eval time       : {met.train_time_seconds:.1f}s")
    print("Confusion matrix:")
    print(cm)

    return met, wrapper


def train_transformer(
    train_texts: list[str],
    test_texts: list[str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: str = "cuda",
) -> tuple[list[EvalMetrics], EvalMetrics | None, dict[str, Any], TransformerWrapper | None]:
    print("\n" + "=" * 60)
    print("  TRACK B — Transformer Pipeline")
    print(f"  Device: {device}")
    print("=" * 60)

    try:
        import torch
        import transformers
    except ImportError as exc:
        print(f"  Transformers not available: {exc}")
        return [], None, {}, None

    train_labels = train_df["label"].values
    test_labels = test_df["label"].values

    all_metrics: list[EvalMetrics] = []
    best_metrics: EvalMetrics | None = None
    best_wrapper: TransformerWrapper | None = None

    for model_name, hf_name in TRANSFORMER_MODELS.items():
        print(f"\n  [{len(all_metrics)+1}/{len(TRANSFORMER_MODELS)}] Training {model_name}...")
        try:
            met, wrapper = _train_single_transformer(
                hf_name, model_name, train_texts, train_labels,
                test_texts, test_labels, device,
            )
            all_metrics.append(met)
            print(f"  {ram_report('')}")

            if best_metrics is None or met.spam_f1 > best_metrics.spam_f1:
                best_metrics = met
                best_wrapper = wrapper
        except Exception as exc:
            print(f"  FAILED: {exc}")
            import traceback
            traceback.print_exc()

    features_config = {
        "tokenizer": TRANSFORMER_MODELS,
        "max_length": MAX_LENGTH,
        "training": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "warmup_ratio": WARMUP_RATIO,
            "weight_decay": WEIGHT_DECAY,
        },
        "meta_feature_names": META_FEATURE_NAMES,
    }

    return all_metrics, best_metrics, features_config, best_wrapper


def retrain_on_full_dataset(
    hf_name: str,
    all_texts: list[str],
    all_labels: np.ndarray,
    device: str = "cuda",
) -> TransformerWrapper:
    """Fine-tune transformer from scratch on the full 342,178-email dataset."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers import get_linear_schedule_with_warmup, AdamW

    print(f"\n  [FULL DATASET RETRAINING] Loading fresh {hf_name}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForSequenceClassification.from_pretrained(hf_name, num_labels=2)
    model.to(device)

    print(f"  Tokenizing {len(all_texts):,} texts...")
    enc = tokenizer(
        all_texts, truncation=True, padding=True,
        max_length=MAX_LENGTH, return_tensors="pt",
    )
    dataset = TensorDataset(
        enc["input_ids"], enc["attention_mask"],
        torch.tensor(all_labels, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    full_epochs = 2
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE * 0.5, weight_decay=WEIGHT_DECAY)
    total_steps = len(loader) * full_epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    print(f"  Fine-tuning {full_epochs} epochs on ALL {len(all_texts):,} emails ({total_steps} steps)...")
    model.train()
    for epoch in range(full_epochs):
        total_loss = 0.0
        for step, batch in enumerate(loader):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            if (step + 1) % 1000 == 0:
                print(f"    Step {step+1}/{len(loader)}, Loss: {loss.item():.4f}")
        print(f"    Epoch {epoch+1}/{full_epochs} done, avg loss: {total_loss / len(loader):.4f}")
        print(f"    {ram_report('Full retrain')}")

    model.eval()
    return TransformerWrapper(model=model, tokenizer=tokenizer, model_name=hf_name, device=device)


def build_transformer_vectorizer_bundle(
    wrapper: TransformerWrapper,
    meta_feature_names: list[str],
) -> dict[str, Any]:
    return {
        "version": 3,
        "model_type": "transformer",
        "tokenizer": wrapper.tokenizer,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "meta_feature_names": meta_feature_names,
    }
