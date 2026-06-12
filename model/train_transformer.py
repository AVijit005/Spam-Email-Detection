"""Track B — Maximum-Performance Transformer Pipeline.

Fine-tunes DistilBERT, RoBERTa, and DeBERTa-v3 on the same holdout split.
Produces a TransformerWrapper compatible with the prediction API.
"""

from __future__ import annotations

import os
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
GRADIENT_ACCUMULATION_STEPS = 1
EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

CHECKPOINT_DIR = CURRENT_DIR / "transformer_checkpoints"


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


def _compute_spam_f1(model, tokenizer, texts: list[str], labels: np.ndarray,
                     device: str) -> float:
    """Compute spam F1 on a validation set without loading sklearn."""
    import torch
    from sklearn.metrics import f1_score, classification_report as cr

    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            enc = tokenizer(batch, truncation=True, padding=True,
                            max_length=MAX_LENGTH, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.append(preds)
    predictions = np.concatenate(all_preds)
    return float(f1_score(labels, predictions, pos_label=1))


def _save_checkpoint(model, tokenizer, optimizer, epoch: int, model_name: str) -> None:
    import torch
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "_").replace("-", "_")
    path = CHECKPOINT_DIR / f"{safe_name}_epoch{epoch}.pt"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    print(f"    Checkpoint saved → {path}")


def _load_latest_checkpoint(model_name: str):
    import torch
    safe_name = model_name.replace("/", "_").replace("-", "_")
    if not CHECKPOINT_DIR.exists():
        return None, 0
    checkpoints = sorted(CHECKPOINT_DIR.glob(f"{safe_name}_epoch*.pt"))
    if not checkpoints:
        return None, 0
    latest = checkpoints[-1]
    data = torch.load(latest, map_location="cpu", weights_only=False)
    print(f"    Resuming from checkpoint: {latest}")
    return data, data.get("epoch", 0)


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
    import torch.cuda.amp as amp
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers import get_linear_schedule_with_warmup, AdamW

    print(f"\n  Loading {model_name}: {hf_name}")

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForSequenceClassification.from_pretrained(hf_name, num_labels=2)
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
    scaler = amp.GradScaler(enabled=(device != "cpu"))

    total_steps = len(train_loader) * EPOCHS // GRADIENT_ACCUMULATION_STEPS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    resume_data, start_epoch = _load_latest_checkpoint(model_name)
    if resume_data is not None:
        model.load_state_dict(resume_data["model_state_dict"])
        optimizer.load_state_dict(resume_data["optimizer_state_dict"])
        start_epoch = resume_data["epoch"]

    print(f"  Mixed precision: {scaler.is_enabled()}")
    print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Fine-tuning {EPOCHS} epochs ({total_steps} effective steps)...")

    best_spam_f1 = -1.0
    best_model_state = None

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            with amp.autocast(device_type=device if device != "cpu" else "cpu",
                              enabled=(device != "cpu")):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            if (step + 1) % 500 == 0:
                print(f"    Epoch {epoch+1}/{EPOCHS}, Step {step+1}/{len(train_loader)}, Loss: {loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}")

        # Handle any remaining gradient accumulation
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        avg_loss = total_loss / len(train_loader)
        print(f"    Epoch {epoch+1}/{EPOCHS} done, avg loss: {avg_loss:.4f}")

        val_f1 = _compute_spam_f1(model, tokenizer, test_texts, test_labels, device)
        print(f"    Validation Spam F1: {val_f1:.4f}")

        if val_f1 > best_spam_f1:
            best_spam_f1 = val_f1
            import copy
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"    ** New best F1: {val_f1:.4f} **")

        _save_checkpoint(model, tokenizer, optimizer, epoch + 1, model_name)
        print(f"    {ram_report('')}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"    Loaded best model (F1={best_spam_f1:.4f})")

    model.eval()
    wrapper = TransformerWrapper(model=model, tokenizer=tokenizer, model_name=model_name, device=device)

    import time as _time
    t0 = _time.perf_counter()
    predictions = wrapper.predict(test_texts)
    eval_time = _time.perf_counter() - t0

    from sklearn.metrics import (
        accuracy_score, classification_report, roc_auc_score, confusion_matrix,
    )
    report = classification_report(
        test_labels, predictions, target_names=["Ham", "Spam"],
        output_dict=True, zero_division=0,
    )

    probs = wrapper.predict_proba(test_texts)[:, 1]

    met = EvalMetrics(
        model_name=model_name,
        track="transformer",
        accuracy=float(accuracy_score(test_labels, predictions)),
        spam_precision=float(report["Spam"]["precision"]),
        spam_recall=float(report["Spam"]["recall"]),
        spam_f1=float(report["Spam"]["f1-score"]),
        roc_auc=float(roc_auc_score(test_labels, probs)),
        train_time_seconds=eval_time,
        support=int(test_labels.sum()),
        confusion_matrix=confusion_matrix(test_labels, predictions).tolist(),
    )

    print(f"\n--- [transformer] {model_name} ---")
    print(f"Accuracy        : {met.accuracy:.4f}")
    print(f"Spam F1         : {met.spam_f1:.4f}")
    print(f"Spam Precision  : {met.spam_precision:.4f}")
    print(f"Spam Recall     : {met.spam_recall:.4f}")
    print(f"ROC-AUC         : {met.roc_auc:.4f}")
    print(f"Eval time       : {met.train_time_seconds:.1f}s")
    print("Confusion matrix:")
    print(met.confusion_matrix)

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
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "mixed_precision": True,
        },
        "meta_feature_names": META_FEATURE_NAMES,
    }

    return all_metrics, best_metrics, features_config, best_wrapper


def retrain_on_full_dataset(
    hf_name: str,
    all_texts: list[str],
    all_labels: np.ndarray,
    winner_wrapper: TransformerWrapper | None = None,
    device: str = "cuda",
) -> TransformerWrapper:
    """Fine-tune the TRANSFORMER_MODELS winner on the full 342,178-email dataset.

    If winner_wrapper is provided, continues fine-tuning its model weights
    (warm start from 3-epoch evaluation model). Otherwise trains from scratch.
    """
    import torch
    import torch.cuda.amp as amp
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers import get_linear_schedule_with_warmup, AdamW

    tokenizer = AutoTokenizer.from_pretrained(hf_name)

    if winner_wrapper is not None and winner_wrapper.model is not None:
        print(f"\n  [FULL DATASET RETRAINING] Continuing fine-tuning from evaluation winner: {hf_name}")
        model = winner_wrapper.model
        model.to(device)
    else:
        print(f"\n  [FULL DATASET RETRAINING] Loading fresh {hf_name}...")
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
    scaler = amp.GradScaler(enabled=(device != "cpu"))
    total_steps = len(loader) * full_epochs // GRADIENT_ACCUMULATION_STEPS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    print(f"  Mixed precision: {scaler.is_enabled()}")
    print(f"  Fine-tuning {full_epochs} epochs on ALL {len(all_texts):,} emails ({total_steps} effective steps)...")
    model.train()

    for epoch in range(full_epochs):
        total_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(loader):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            with amp.autocast(device_type=device if device != "cpu" else "cpu",
                              enabled=(device != "cpu")):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            if (step + 1) % 1000 == 0:
                print(f"    Step {step+1}/{len(loader)}, Loss: {loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}")
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
