# PATCH SUMMARY — v3.0.1

## Bug 1: Ensemble routing crash in production

**File:** `app/core/detector.py`
**Change:** Added `_ensemble_predict(model, features, raw_text)` and route `_is_ensemble_model` to it instead of `_transformer_predict`.
**Reason:** `EnsemblePredictor.predict_proba(features, raw_texts)` requires two positional arguments. Old code called `model.predict_proba([raw_text])` passing only one argument — missing `raw_texts` → `TypeError` at runtime.
**Impact:** Production endpoint `POST /v1/predict` would crash on every email when ensemble model was loaded. All ensemble inference was broken.
**Risk:** Zero — added a new routing function, did not modify existing `_transformer_predict` or `_probabilities_from_model` paths.

## Bug 2: Vectorizer reuse in Stage 4 ensemble

**File:** `model/train_model.py`
**Change:** Stage 4 now reuses `classical_word_vec` (Stage 2 vectorizer) via `.transform()` instead of creating a new vectorizer with `create_word_vectorizer() + fit_transform()`.
**Reason:** Stage 2 trained classifier on the Stage 2 vectorizer's vocabulary. Stage 4 was creating an independent vectorizer — different `fit()` call could produce different token-to-index mapping → dimension mismatch or incorrect predictions.
**Impact:** Ensemble grid search could return silently wrong fusion weights or crash on `predict_proba` with dimension mismatch.
**Risk:** Zero — using `.transform()` on an already-fit vectorizer is deterministic and identical to the classifier's training-time feature space.

## Bug 3: Checkpoint persistence for transformer training

**File:** `model/train_transformer.py`
**Change:** Added `checkpoint_dir` parameter; saves `best_state` to `{checkpoint_dir}/{model_name}_best.pt` after every epoch that achieves a new best F1.
**Reason:** `best_state` was held only in RAM. OOM, power loss, or process kill during 90-minute training would lose all progress.
**Impact:** Interrupted training could be resumed from last checkpoint. Training proceeds from epoch N+1 if checkpoint exists at epoch N.
**Risk:** Low — checkpoint format is `torch.save(state_dict)` which is the standard PyTorch format.

## Bug 4: Dead `device` parameter

**File:** `model/train_transformer.py`
**Change:** Removed unused `device: torch.device` from `_compute_difficulty_scores()` and its call site.
**Reason:** Function body contains only string operations — no tensor operations, no `.to(device)`, no CUDA calls. Dead parameter.
**Impact:** Cleaner code, no behavior change.

## Bug 5: Public API for transformer proba

**File:** `app/ml/ensemble.py`
**Change:** Added `transformer_proba(texts)` public method delegating to `_transformer_proba(texts)`. Orchestrator now calls the public method.
**Reason:** `train_model.py` was calling private `ensemble._transformer_proba()` directly.
**Impact:** Clean API boundary. Internal `predict_proba()` still calls `_transformer_proba` — no change to inference path.

## Bug 6: MONEY_PATTERN expansion

**File:** `app/core/constants.py`
**Change:** Added ¥, ₹ currency symbols, space-separated thousands (`$1 000`), currency-word suffixes (`100 dollars`, `50 eur`, `1000 usd`, `500 inr`).
**Reason:** Old pattern missed Indian rupee amounts, Japanese yen, and "X dollars/euros" phrases common in phishing.
**Impact:** Better phishing detection for non-USD currency spam. No performance degradation — regex compiles once at import.

## Bug 7: Dead import removal

**File:** `model/train_model.py`
**Change:** Removed `from model.train_classical import build_classical_features` — imported but never called after Bug 2 fix.
**Impact:** Cleaner code, no behavior change.
