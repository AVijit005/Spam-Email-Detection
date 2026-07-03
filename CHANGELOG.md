# Changelog

All notable changes to the Spam Email Detection project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.1.0] — 2026-06-16

### Added

- **Full Kaggle training execution** — Trained on the complete 342,178-row dataset using dual T4 GPUs. Replaced the LogisticRegression baseline (2,605 rows, 92.2% F1) with the full ensemble (XGBoost + DeBERTa-v3).
- **Final model artifacts** — Ensemble model achieving 99.22% spam F1: XGBoost classical branch (98.33% F1, 99.86% ROC-AUC), DeBERTa-v3 transformer branch (99.13% F1, 99.95% ROC-AUC), fusion weight w=0.35.
- **DeBERTa-v3 checkpoints** — `checkpoints/DeBERTa-v3_best.pt` and `checkpoints/DeBERTa-v3_checkpoint.pt` with resume capability.
- **Token cache** — Pre-tokenized train/test splits saved as safetensors for faster re-training.
- **SHA-256 integrity files** — All model artifacts include `.sha256` companion files.

### Changed

- `spam_model.pkl`: 118 KB LogisticRegression → 1.35 MB XGBoost (25,000 features + 32 meta-features)
- `vectorizer.pkl`: 10,000 word + 5,000 char TF-IDF → 25,000 word unigram/bigram TF-IDF
- `model_metadata.json`: Updated with full training statistics, candidate comparisons, and ensemble config
- `README.md`: Replaced "Expected v3.0 Performance" section with actual Kaggle results
- `TECHNICAL_REPORT.md`: Populated all "Pending final run" placeholders with final metrics

---

## [3.0.1] — 2026-04-03

### Fixed

- **Ensemble routing crash** — `EnsemblePredictor.predict_proba()` was called with a single positional argument instead of two (`features`, `raw_texts`), causing `TypeError` on every production prediction request. Added `_ensemble_predict()` routing function with correct argument passing.
- **Stage 4 vectorizer mismatch** — Ensemble grid search created an independent TF-IDF vectorizer instead of reusing the Stage 2 vectorizer with `.transform()`, causing potential dimension mismatches or silently wrong fusion weights.
- **Transformer checkpoint persistence** — `best_state` was held only in RAM during training. OOM, power loss, or process kill during a 90-minute training run would lose all progress. Now saves `best_state` to `checkpoints/{model}_best.pt` after every best-F1 epoch.
- **Missing public `transformer_proba()` API** — `train_model.py` was calling the private `_transformer_proba()` method on `EnsemblePredictor`. Added a public `transformer_proba()` method delegating internally.
- **`MONEY_PATTERN` incomplete** — Missing ₹ (rupee), ¥ (yen), space-separated thousands (`$1 000`), and currency-word suffixes (`100 dollars`, `50 eur`, `1000 usd`). Expanded regex.
- **Dead `device` parameter** — `_compute_difficulty_scores()` had an unused `device: torch.device` parameter. Removed.
- **Dead import** — `from model.train_classical import build_classical_features` was imported but never called after the Stage 4 vectorizer fix. Removed.

### Security

- **SHA-256 verification** — Fixed `hashlib.compare_digest` → `hmac.compare_digest`. `hashlib.compare_digest` does not exist; SHA-256 integrity verification had never actually worked.
- **Rate limiting** — Fixed `SlowAPIMiddleware` not being registered in the FastAPI app. Rate limiting had never been enforced.

---

## [3.0.0] — 2026-03-28

### Added

- **Dual-track training architecture** — 6-stage training orchestrator (`model/train_model.py`):
  - **Stage 1**: Load & preprocess CSV (supports 342,178-row Kaggle dataset)
  - **Stage 2**: Track A — Classical ML candidates (SGDClassifier, XGBoost, LightGBM) with Optuna hyperparameter optimization (30 trials, 20-min timeout)
  - **Stage 3**: Track B — Transformer fine-tuning (DeBERTa-v3 as primary; also RoBERTa, ELECTRA, ModernBERT, DistilBERT, BERT) with focal loss, FGM adversarial training, and curriculum learning
  - **Stage 4**: Ensemble fusion — grid search for optimal fusion weight (21 steps, 0.0–1.0) optimizing spam F1
  - **Stage 5**: Retrain winner on 100% dataset
  - **Stage 6**: Export artifacts with SHA-256 integrity hashes
- **`EnsemblePredictor`** (`app/ml/ensemble.py`) — weighted late-fusion: `p_spam = w · p_classical + (1-w) · p_transformer`. Graceful fallback to classical-only if transformer unavailable.
- **Kaggle GPU training support** — auto-detection of Kaggle input directories via `KAGGLE_INPUT_DIR` and `/kaggle/input/`. Multi-GPU DDP support via `torchrun`. CLI flags: `--competition`, `--csv-path`, `--output-dir`.
- **32 meta-features** — expanded from 16. Added: URL domain analysis, HTML/hidden content detection, Unicode obfuscation, homograph detection, Flesch reading ease, type-token ratio, imperative verb ratio, credential harvesting hits, attachment indicators.
- **Transformer checkpoint system** — best-F1 checkpoints saved to disk; emergency checkpoint on SIGTERM/SIGINT; safetensors token caching to avoid re-tokenization on rerun.
- **SHA-256 integrity for all artifacts** — sidecar `.sha256` files for `spam_model.pkl`, `vectorizer.pkl`, and `transformer_model.pt`.
- **Model metadata export** — comprehensive `model_metadata.json` with training config, metrics, timestamps, and feedback stats.
- **Training and validation guides** (`TRAINING_GUIDE.md`, `VALIDATION_GUIDE.md`)

### Enhanced

- **Detection engine** — ML layer now routes to ensemble (XGBoost + DeBERTa-v3) when both artifacts are present, transformer-only, or classical-only, depending on available artifacts.
- **Model registry** — `save_model()` and `load_model()` support ensemble artifacts and SHA-256 verification.

---

## [2.0.0] — 2026-02-15

### Added

- **Chrome Extension** (Manifest V3) — Gmail integration with DOM parsing, auto-scan on email view, overlay banners with confidence and explanations, popup for manual scanning, options page for configuration.
- **5-layer detection pipeline**: Whitelist → Trusted Service Catalog → Rule-Based Spam → Benign Context Guard → ML Model.
- **User feedback loop** — `POST /v1/feedback` with JSONL file storage (default) and optional MySQL storage. PII redaction at API boundary.
- **Retraining** — `POST /v1/retrain` with concurrency lock, subprocess-based training, and automatic model reload.
- **Docker deployment** — multi-stage Dockerfile with non-root user, Docker Compose with optional MySQL profile, Gunicorn + Uvicorn, health checks.
- **Security hardening** — API key authentication (`X-API-Key` header), rate limiting (60 req/min via SlowAPI), CORS protection (origin regex), SHA-256 model integrity verification.
- **PII redaction** — 5 patterns: email addresses, phone numbers, IP addresses, SSNs, credit card numbers.
- **Explanation engine** — per-prediction explanations showing top contributing features from model coefficients.
- **Schema validation** — Pydantic models for all request/response types with max-length enforcement.

---

## [1.0.0] — 2026-01-10

### Added

- **Initial release** — single LogisticRegression model with TF-IDF vectorization (word + character n-grams).
- **16 meta-features**: URL count, caps ratio, exclamation/question count, money count, phone count, word count, digit ratio, spam phrase hits, urgency/account/CTA keyword hits, and more.
- **Basic FastAPI backend** — `POST /predict` endpoint, health check.
- **Training pipeline** — 80/20 stratified split, holdout evaluation, full-dataset retrain.
- **Model metadata** — JSON export with training config and metrics.
- **20 backend tests** covering core utilities and API endpoints.
