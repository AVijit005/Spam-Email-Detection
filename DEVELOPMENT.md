# Development Notes (v3.1)

## Where The Project Stands

The project is a production-grade Gmail spam and phishing detector with:

- Chrome extension: Gmail DOM extraction, banner overlays, popup scanner, options page, feedback
- FastAPI backend: 5-layer detection pipeline, dual-track ensemble inference (XGBoost + DeBERTa-v3)
- ML pipeline: 6-stage training orchestrator (Load → Classical → Transformer → Ensemble → Retrain → Export)
- Docker deployment with SHA-256 model integrity, PII redaction, rate limiting, CORS

## Detection Pipeline

1. **User whitelist** — trusted sender domains configured in extension options
2. **Trusted service catalog** — curated list of financial/service provider domains
3. **Phishing/spam rules** — keyword patterns, urgency signals, credential harvesting
4. **Benign context guard** — conversational language, personal references, meeting context
5. **ML ensemble** — XGBoost (TF-IDF + 32 meta-features) + DeBERTa-v3 (contextual) via weighted late fusion

## Model Architecture

The production deployment uses an **EnsemblePredictor** with:

- **XGBoost** (classical track): 25,000 TF-IDF word unigrams/bigrams + 32 engineered meta-features
- **DeBERTa-v3** (transformer track): `microsoft/deberta-v3-base` fine-tuned with focal loss (γ=2.0), FGM adversarial training (ε=0.5), curriculum learning
- **Fusion**: Weighted late fusion with w=0.35 (grid-searched optimal), gracefully degrades to XGBoost-only if transformer unavailable

Trained on 342,178 emails (Kaggle GPU). Ensemble F1: 99.22%.

## Training Pipeline

6-stage orchestrator (`model/train_model.py`):

1. Load & preprocess CSV (token replacement, parallel processing)
2. Track A — Classical ML (SGDClassifier, XGBoost, LightGBM) with optional Optuna HPO
3. Track B — Transformer fine-tuning (DeBERTa-v3) with focal loss + FGM + curriculum learning
4. Ensemble fusion — grid search for optimal fusion weight
5. Retrain winner on full dataset
6. Export artifacts with SHA-256 integrity hashes

Supports: Kaggle auto-detection, multi-GPU DDP, checkpoint resume, VRAM-probed batch sizing.

## Extension Architecture

- Manifest V3 with service worker background
- Content script with MutationObserver for Gmail DOM changes
- Banner injection with feedback buttons
- Popup: paste-and-analyze, Gmail extraction, scan history, settings
- Options: backend URL, API key, auto-scan toggle, history limit
- API key-aware — sends `X-API-Key` header when configured

## Configuration

All settings use `SPAM_` prefixed environment variables (pydantic-settings).

Key settings:
- `SPAM_ENABLE_TRANSFORMER` — toggle ensemble vs XGBoost-only (default: true)
- `SPAM_TRANSFORMER_DEVICE` — `cpu` or `cuda` (default: cpu)
- `SPAM_TRANSFORMER_MODEL_NAME` — HuggingFace model ID (default: microsoft/deberta-v3-base)
- `SPAM_MODEL_PATH`, `SPAM_TRANSFORMER_MODEL_PATH`, `SPAM_TRANSFORMER_TOKENIZER_PATH` — artifact paths

## Current Quality

- Backend unit tests: 185 passing
- Integration tests: 26 passing
- Legacy tests: 20 passing
- SHA-256 integrity verified for all model artifacts
- Known limitations documented in `KNOWN_ISSUES.md`

## Remaining Opportunities

- No extension automated tests (DOM parsing, banner injection)
- Gmail DOM selectors may break on UI updates
- Single Gunicorn worker (acceptable for current ensemble memory footprint)
- No CI/CD pipeline
- Retraining is user-triggered, not scheduled
