"""Weighted Late-Fusion Ensemble — Classical + Transformer.

XGBoost handles keyword/phrase spam via TF-IDF bigrams.
DeBERTa-v3 handles sophisticated phishing via contextual understanding.
Late fusion averages probabilities with equal weight — no learned
meta-learner, avoiding overfitting on a single holdout set.

Architecture:
  p_spam = 0.50 × p_classical + 0.50 × p_transformer
  label = Spam if p_spam ≥ 0.55 else Not Spam

Falls back gracefully to classical-only if the transformer is unavailable.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import scipy.sparse as sp


warnings.filterwarnings("ignore", category=FutureWarning)


class EnsemblePredictor:
    def __init__(
        self,
        classical_model: Any,
        classical_vectorizer_bundle: dict[str, Any],
        transformer_model: Any | None = None,
        transformer_tokenizer: Any | None = None,
        transformer_device: str = "cpu",
        fusion_weight: float = 0.50,
        transformer_max_length: int = 512,
    ):
        self.classical_model = classical_model
        self.classical_vectorizer_bundle = classical_vectorizer_bundle
        self.transformer_model = transformer_model
        self.transformer_tokenizer = transformer_tokenizer
        self.transformer_device = transformer_device
        self.fusion_weight = fusion_weight
        self.transformer_max_length = transformer_max_length

    @property
    def has_transformer(self) -> bool:
        return self.transformer_model is not None and self.transformer_tokenizer is not None

    def _transformer_proba(self, texts: list[str]) -> np.ndarray:
        import torch
        was_training = self.transformer_model.training
        self.transformer_model.eval()
        device = next(self.transformer_model.parameters()).device
        probs_list = []
        for text in texts:
            enc = self.transformer_tokenizer(
                text, truncation=True, padding="max_length",
                max_length=self.transformer_max_length, return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            with torch.no_grad():
                with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", enabled=device.type == "cuda"):
                    outputs = self.transformer_model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            probs_list.append(probs)
        if was_training:
            self.transformer_model.train()
        return np.vstack(probs_list)

    def transformer_proba(self, texts: list[str]) -> np.ndarray:
        return self._transformer_proba(texts)

    def predict_proba(self, features: sp.csr_matrix, raw_texts: list[str]) -> np.ndarray:
        p_classical = self.classical_model.predict_proba(features)
        if not self.has_transformer:
            return p_classical
        p_transformer = self._transformer_proba(raw_texts)
        p_spam = (
            self.fusion_weight * p_classical[:, 1]
            + (1 - self.fusion_weight) * p_transformer[:, 1]
        )
        p_ham = 1.0 - p_spam
        return np.column_stack([p_ham, p_spam])

    def predict(self, features: sp.csr_matrix, raw_texts: list[str]) -> np.ndarray:
        return (self.predict_proba(features, raw_texts)[:, 1] >= 0.55).astype(np.int32)


def grid_search_fusion_weight(
    classical_probs: np.ndarray,
    transformer_probs: np.ndarray,
    y_true: np.ndarray,
    n_steps: int = 21,
) -> dict[str, Any]:
    best_f1 = 0.0
    best_weight = 0.50
    results = []
    for w in np.linspace(0.0, 1.0, n_steps):
        fused = w * classical_probs[:, 1] + (1 - w) * transformer_probs[:, 1]
        from sklearn.metrics import f1_score
        f1 = f1_score(y_true, fused >= 0.55, pos_label=1)
        results.append((w, f1))
        if f1 > best_f1:
            best_f1 = f1
            best_weight = w

    print(f"\n  Fusion weight grid search ({n_steps} steps):")
    for w, f1_score_val in results:
        marker = "<-" if w == best_weight else "  "
        print(f"    {marker} w={w:.2f} → Spam F1={f1_score_val:.4f}")

    return {"best_weight": round(best_weight, 4), "best_f1": round(best_f1, 4)}
