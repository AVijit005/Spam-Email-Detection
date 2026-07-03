---
title: Spam Email Detection
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: docker
pinned: true
app_port: 8000
---

# 🛡️ Spam Email Detection — Dual-Track ML for Gmail Protection

**A production-grade spam and phishing detection system with a Chrome extension, FastAPI backend, dual-track ensemble (XGBoost + DeBERTa-v3), 5-layer detection pipeline, explainable predictions, user feedback loop, and Docker deployment.**

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"></a>
  <a href="https://xgboost.readthedocs.io/"><img src="https://img.shields.io/badge/XGBoost-2.0+-32B34A?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.3+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://huggingface.co/"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DeBERTa--v3-FFD21E?style=for-the-badge" alt="HuggingFace"></a>
  <a href="https://www.docker.com/"><img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"></a>
  <a href="#testing"><img src="https://img.shields.io/badge/Tests-225%20Passing-success?style=for-the-badge" alt="Tests"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License"></a>
</p>

---

## Overview

Spam Email Detection is a complete spam and phishing detection platform that combines a Chrome extension for Gmail with a FastAPI backend running a **dual-track ML ensemble**. It combines classical machine learning (XGBoost with TF-IDF and 32 engineered meta-features) with transformer-based deep learning (Microsoft DeBERTa-v3 fine-tuned with focal loss, FGM adversarial training, and curriculum learning) through weighted late fusion — giving you the pattern-matching power of gradient boosting and the contextual understanding of a state-of-the-art language model.

### Why this project stands out

- **Dual-track ensemble**: XGBoost handles keyword/pattern spam. DeBERTa-v3 handles sophisticated phishing. The ensemble combines both for maximum accuracy, with Optuna HPO available for classical track optimization when training on new data.
- **6-stage training pipeline**: Load → Classical (3 candidates + Optuna HPO) → Transformer (focal loss + FGM + curriculum) → Ensemble Fusion → Retrain Winner → Export Artifacts
- **5-layer detection pipeline**: Whitelist → Trusted Catalog → Rule-Based Spam → Benign Context Guard → ML Ensemble — 40-60% of emails never reach the ML model
- **32 engineered meta-features**: URL analysis, HTML detection, Unicode obfuscation, homograph attacks, credential harvesting patterns, readability scores
- **Production-grade**: Docker deployment, env-based config, CORS, rate limiting, API key auth, SHA-256 model integrity, PII redaction
- **225 deterministic tests**: Full coverage of all production modules, ~4-second suite execution
- **Kaggle GPU training**: Auto-detection, multi-GPU DDP, checkpoint resume, VRAM-probed batch sizing

---

## Architecture

```
┌─────────────────┐     ┌────────────────────────────────────────────────────┐
│   Gmail UI      │────▶│              Chrome Extension (Manifest V3)          │
│                 │     │  ┌─────────────┐  ┌──────────┐  ┌────────────────┐  │
│  Inbox / Email  │     │  │  content.js │  │ popup.js │  │  options page  │  │
└─────────────────┘     │  └──────┬──────┘  └────┬─────┘  └───────┬────────┘  │
                         └─────────┼──────────────┼────────────────┼──────────┘
                                   │              │                │
                                   ▼              ▼                ▼
                         ┌────────────────────────────────────────────────────┐
                         │              FastAPI Backend (:8000)                │
                         │  ┌──────────────────────────────────────────────┐  │
                         │  │  Middleware: CORS │ Rate Limit │ Auth (Key)  │  │
                         │  │                                                │  │
                         │  │  GET  /v1/health        POST /v1/predict      │  │
                         │  │  POST /v1/predict/batch  POST /v1/feedback 🔒 │  │
                         │  │  GET  /v1/feedback/summary  POST /v1/retrain 🔒│  │
                         │  └──────────────────────────────────────────────┘  │
                         │                                                    │
                         │  ┌──────────────────────────────────────────────┐  │
                         │  │           5-Layer Detection Pipeline          │  │
                         │  │  1. Whitelist        → confidence 1.0         │  │
                         │  │  2. Trusted Catalog  → confidence 0.97        │  │
                         │  │  3. Rule-Based Spam  → confidence 0.86-0.99   │  │
                         │  │  4. Benign Context   → confidence 0.76-0.82   │  │
                         │  │  5. ML Ensemble ─────────────────┐            │  │
                         │  └──────────────────────────────────│───────────┘  │
                         │                                      ▼              │
                         │  ┌──────────────────────────────────────────────┐  │
                         │  │        Dual-Track ML Ensemble                │  │
                         │  │                                              │  │
                         │  │  Track A: XGBoost           Track B: DeBERTa │  │
                         │  │  ├─ TF-IDF (25K n-grams)    ├─ Focal Loss    │  │
                         │  │  ├─ 32 Meta-Features        ├─ FGM Adversar. │  │
                         │  │  └─ Optuna HPO              └─ Curriculum    │  │
                         │  │         │                          │          │  │
                         │  │         └────────┬─────────────────┘          │  │
                         │  │                  ▼                            │  │
                         │  │    p_spam = w·p_xgb + (1-w)·p_deberta        │  │
                         │  │         (Grid-searched fusion weight)         │  │
                         │  └──────────────────────────────────────────────┘  │
                         │                                                    │
                         │  ┌──────────────┐  ┌───────────────────────────┐  │
                         │  │ PII Redact   │  │  SHA-256 Model Integrity  │  │
                         │  └──────────────┘  └───────────────────────────┘  │
                         └───────────────────────┬────────────────────────────┘
                                                 │
                                   ┌─────────────┴─────────────┐
                                   │                           │
                             ┌─────▼─────┐              ┌──────▼──────┐
                             │ feedback  │              │    model    │
                             │  .jsonl   │              │  artifacts  │
                             │  / MySQL  │              │  (pickle)   │
                             └───────────┘              └─────────────┘
```

For full architecture with Mermaid diagrams, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md).

---

## Key Features

### Dual-Track ML Engine

| Track | Approach | Best For | Inference |
|---|---|---|---|
| **A — Classical** | TF-IDF + 32 meta-features + XGBoost/LightGBM/SGD with Optuna HPO | Keyword/pattern spam | ~3 ms |
| **B — Transformer** | DeBERTa-v3 with Focal Loss, FGM, Curriculum Learning | Sophisticated phishing | ~50 ms |
| **Ensemble** | Weighted late fusion (grid-searched weight) | Boundary cases, max F1 | ~55 ms |

### 5-Layer Detection Pipeline

Emails pass through progressively sophisticated analysis — only ~40-60% reach the ML model:

| Layer | Method | Decision Confidence |
|---|---|---|
| **Whitelist** | Exact sender domain match | 1.0 |
| **Trusted Catalog** | Curated known-services list | 0.97 |
| **Rule-Based Spam** | Phishing phrase + signal matching | 0.86–0.99 |
| **Benign Context** | Conversational/promotional detection | 0.76–0.82 |
| **ML Ensemble** | XGBoost + DeBERTa-v3 late fusion | Full probability |

### Chrome Extension

- **Gmail integration** via Manifest V3 — auto-scan emails with visual overlay banners
- **Manual scanning** from extension popup
- **Feedback submission** (correct/incorrect labels) for model improvement
- **Explainable results** — see exactly *why* an email was flagged
- **Options page** — configure backend URL, timeout, retraining controls

### Security

- **API key authentication** on feedback and retrain endpoints
- **Rate limiting** — 60 req/min per IP with proper 429 responses
- **SHA-256 model integrity** — detects tampered or corrupted artifacts at load time
- **PII redaction** — 5 patterns (email, phone, IP, SSN, credit card) redacted at API boundary
- **CORS protection** — origin regex restricting to extensions and localhost
- **SQL injection prevention** — table name validation regex

### Production Deployment

- **Docker** with multi-stage build and non-root user
- **Docker Compose** with optional MySQL profile
- **Gunicorn + Uvicorn** for production ASGI serving
- **Health checks** on both Docker and API level
- **Environment-driven** configuration — zero code changes between environments

---

## Quick Start

### Prerequisites

- Python 3.11+
- (Optional) Docker + Docker Compose
- (Optional) NVIDIA GPU for transformer training

### 5-Minute Setup

```bash
# Clone
git clone https://github.com/AVijit005/Spam-Email-Detection.git
cd Spam-Email-Detection

# Setup environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .\.venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

# Quick smoke test (500 rows, ~5 min, no GPU needed)
python model/train_model.py --fast-dev

# Full classical training (CPU only, ~35 min)
python model/train_model.py --track-a-only

# Start the API server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Verify:
```bash
curl http://127.0.0.1:8000/v1/health
```

### Docker Deployment

```bash
cp .env.example .env
docker compose up --build
```

### Full GPU Training (Kaggle)

```bash
python model/train_model.py --model DeBERTa-v3
```

See [KAGGLE_RECOVERY_GUIDE.md](KAGGLE_RECOVERY_GUIDE.md) for detailed GPU training instructions.

---

## API Reference

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/v1/health` | None | Server status, model info, feedback stats |
| `POST` | `/v1/predict` | None | Single email prediction |
| `POST` | `/v1/predict/batch` | None | Batch prediction (max 50) |
| `POST` | `/v1/feedback` | API Key | Submit user label for a prediction |
| `GET` | `/v1/feedback/summary` | None | Aggregate feedback counts by verdict |
| `POST` | `/v1/retrain` | API Key | Trigger model retraining |

### Example: Predict

**Request**
```json
{
  "sender": "security-alert@unknown-domain.com",
  "subject": "URGENT: Verify your account immediately",
  "body": "Dear user, your account has been compromised. Click here to verify your credentials: https://bit.ly/3xK9mP2"
}
```

**Response**
```json
{
  "label": "Spam",
  "confidence": 0.96,
  "reason": "Dual-track ensemble detected multiple phishing indicators",
  "analysis": "AI analysis: 96.0% spam probability. Combined signals from XGBoost pattern detection and DeBERTa-v3 contextual analysis.",
  "model_version": "Ensemble-XGBoost-DeBERTa-v3-20260403",
  "sender_domain": "unknown-domain.com",
  "rule_layer": "ml",
  "explanations": [
    "Suspicious token: \"verify\"",
    "Suspicious signal: contains urgency language",
    "Suspicious signal: contains URL shortener",
    "Suspicious signal: credential harvesting phrase detected"
  ],
  "prediction_id": "a1b2c3d4e5f6",
  "evaluated_at_utc": "2026-04-03T12:00:00+00:00"
}
```

---

## Project Structure

```
spam-email-detection/
├── app/                              # Production FastAPI application
│   ├── api/v1/                       # REST endpoints
│   │   ├── health.py                 #   GET /health
│   │   ├── predict.py                #   POST /predict, /predict/batch
│   │   ├── feedback.py               #   POST /feedback, GET /feedback/summary
│   │   ├── retrain.py                #   POST /retrain
│   │   └── router.py                 #   API router assembly
│   ├── core/                         # Detection engine
│   │   ├── detector.py               #   5-layer prediction pipeline + ensemble routing
│   │   ├── features.py               #   32 meta-feature extraction
│   │   ├── rules.py                  #   Rule-based spam + benign context detection
│   │   ├── text.py                   #   NLP text preprocessing
│   │   ├── explain.py                #   ML prediction explanation engine
│   │   ├── domain.py                 #   Domain normalization + catalog loading
│   │   ├── constants.py              #   Regex patterns, keyword sets, phrase libraries
│   │   └── auth.py                   #   API key authentication
│   ├── ml/                           # ML subsystem
│   │   ├── ensemble.py               #   EnsemblePredictor (weighted late fusion)
│   │   └── registry.py               #   Model save/load with SHA-256 integrity
│   ├── schemas/                      # Pydantic request/response models
│   ├── storage/
│   │   └── feedback.py               #   Feedback persistence (JSONL + MySQL)
│   ├── utils/
│   │   └── pii.py                    #   PII redaction (5 patterns)
│   ├── config.py                     #   Env-driven settings (pydantic-settings)
│   └── main.py                       #   App factory, middleware, lifespan handler
├── model/                            # Training pipeline
│   ├── train_model.py                #   6-stage training orchestrator
│   ├── train_classical.py            #   Track A: classical ML pipeline
│   ├── train_transformer.py          #   Track B: transformer fine-tuning
│   └── shared.py                     #   Shared metrics, evaluation, export utilities
├── extension/                        # Chrome extension (Manifest V3)
│   ├── content.js                    #   Gmail DOM integration + overlay banners
│   ├── background.js                 #   Service worker (API calls, caching)
│   ├── popup.js / popup.html         #   Extension popup UI
│   ├── options.js / options.html     #   Settings page
│   └── utils/domParser.js            #   Gmail DOM parsing
├── tests/                            # Test suite (205 tests)
│   ├── unit/                         #   14 unit test files
│   └── integration/                  #   6 integration test files
├── backend/                          # Legacy utilities (kept for reference)
├── data/                             # Datasets, whitelists, trusted catalogs
├── docs/                             # Architecture, deployment, security, testing docs
├── model/checkpoints/                # Training checkpoints and token cache
├── Dockerfile                        # Multi-stage production image
├── docker-compose.yml                # Backend + optional MySQL
├── .env.example                      # Environment template
├── requirements.txt                  # Python dependencies
└── LICENSE                           # MIT License
```

---

## Documentation

| Document | Description |
|---|---|
| [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) | Design philosophy, architecture decisions, tradeoffs |
| [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) | Training pipeline, inference flow, ensemble, checkpoint system |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Full system architecture with Mermaid diagrams |
| [KAGGLE_RECOVERY_GUIDE.md](KAGGLE_RECOVERY_GUIDE.md) | Kaggle GPU training, checkpoint recovery, troubleshooting |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Deployment guide (Docker, env vars, production setup) |
| [docs/SECURITY.md](docs/SECURITY.md) | Security features, threat model, limitations |
| [docs/TESTING.md](docs/TESTING.md) | Test suite documentation (225 tests, 100% passing) |
| [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) | Full methodology, experiments, and engineering report |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup, coding standards, PR workflow |
| [CHANGELOG.md](CHANGELOG.md) | Version history (v1.0 → v3.0.1) |
| [TRAINING_GUIDE.md](TRAINING_GUIDE.md) | Quick-reference training commands |
| [VALIDATION_GUIDE.md](VALIDATION_GUIDE.md) | Verification scripts for ensemble, checkpoints, artifacts |
| [KNOWN_ISSUES.md](KNOWN_ISSUES.md) | Current limitations and improvement ideas |

---

## Model Performance

*Trained on Kaggle GPU | 342,178 emails | June 16, 2026*

### Ensemble Results

| Configuration | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Model Size | Inference |
|---|---|---|---|---|---|---|---|
| **Ensemble (XGBoost + DeBERTa-v3)** | **—** | **—** | **—** | **99.22%** | **—** | **~714 MB** | **~55 ms** |
| Classical (XGBoost) | 98.29% | 97.66% | 99.01% | 98.33% | 99.86% | ~2 MB | ~3 ms |
| Transformer (DeBERTa-v3) | 99.11% | 99.47% | 98.79% | 99.13% | 99.95% | ~712 MB | ~50 ms |
| LightGBM (candidate) | 98.23% | 97.61% | 98.95% | 98.28% | 99.86% | — | — |
| SGDClassifier (candidate) | 90.73% | 85.38% | 98.71% | 91.56% | 97.92% | — | — |

- **Ensemble fusion weight**: 0.35 (grid-searched optimal)
- **Training pipeline**: 6-stage — Load → Classical (3 candidates, default params) → Transformer (focal loss + FGM + curriculum) → Ensemble Fusion → Retrain Winner → Export
- **Classical features**: 25,000 TF-IDF word unigrams/bigrams + 32 engineered meta-features
- **Transformer**: microsoft/deberta-v3-base fine-tuned with focal loss (γ=2.0), FGM adversarial training (ε=0.5), 1 curriculum epoch
- **First-stage filters**: 40–60% of emails classified before reaching ML model
- **Deployment**: Ensemble active by default. Set `SPAM_ENABLE_TRANSFORMER=false` for XGBoost-only mode if GPU/RAM-constrained or offline. Graceful fallback handles missing/corrupted transformer artifacts.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI 0.115+ |
| Classical ML | XGBoost, LightGBM, scikit-learn (SGD, TF-IDF) |
| Deep Learning | PyTorch 2.3+, HuggingFace Transformers (DeBERTa-v3) |
| Hyperparameter Optimization | Optuna (TPE sampler, median pruning) |
| NLP | NLTK (stopwords, tokenization) |
| ASGI Server | Uvicorn + Gunicorn |
| Container | Docker multi-stage + Docker Compose |
| Database (optional) | MySQL 8.0 via PyMySQL |
| Frontend | Chrome Extension (Manifest V3, vanilla JS) |
| Testing | Python unittest (225 tests) |

---

## Testing

```bash
# Run all tests (~4 seconds)
python -m unittest discover -s tests -v
python -m unittest discover -s backend/tests -v

# Total: 225 passing tests
# - 205 new tests (185 unit + 26 integration)
# - 20 legacy tests
```

| Category | Files | Tests | Coverage |
|---|---|---|---|
| Unit — ML Registry (SHA-256) | 1 | 7 | Full |
| Unit — Auth (API key) | 1 | 10 | Full (3 states) |
| Unit — PII Redaction | 1 | 12+3 | Full (5 patterns) |
| Unit — Feedback Store | 1 | 22 | Full (file + MySQL) |
| Unit — NLP Preprocessing | 1 | 12 | Full |
| Unit — Feature Extraction | 1 | 29 | Full (32 features) |
| Unit — Explanation Engine | 1 | 8 | Full |
| Unit — Rules Engine | 1 | 16 | Full |
| Unit — Detector (5 layers) | 1 | 16 | Full (including ensemble routing) |
| Unit — Domain | 1 | 26 | Full |
| Unit — Schemas | 1 | 18 | Full |
| Unit — Config | 1 | 5 | Full |
| Integration — API (Auth, Rate, CORS, Predict, Retrain, Bootstrap) | 6 | 26 | Full |
| **Total** | **20** | **225** | **100% pass rate** |

See [docs/TESTING.md](docs/TESTING.md) for detailed per-test coverage.

---

## Future Roadmap

- [ ] Scheduled retraining (cron-based or feedback-volume threshold)
- [ ] Multi-user support with JWT authentication
- [ ] Model A/B testing infrastructure with traffic splitting
- [ ] Admin dashboard for feedback review and model monitoring
- [ ] Real-time email scanning via Gmail API (replace DOM parsing)
- [ ] Model distillation (DeBERTa-v3 → DistilBERT student)
- [ ] Multi-language spam phrase libraries
- [ ] CI/CD pipeline with automated testing and model evaluation
- [ ] Support for additional email providers (Outlook, Yahoo)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Avijit Pal**

[![GitHub](https://img.shields.io/badge/GitHub-AVijit005-181717?style=flat-square&logo=github)](https://github.com/AVijit005)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/avijit-pal)
[![Email](https://img.shields.io/badge/Email-Contact-EA4335?style=flat-square&logo=gmail)](mailto:avijit.pal@example.com)

**B.Tech in Computer Science and Engineering** — Brainware University

**Machine Learning Engineer | Data Science Enthusiast | Software Developer**

**Skills:** `Python` `Machine Learning` `Deep Learning` `Data Science` `C` `C++` `Java`

Built as a capstone ML engineering project demonstrating production-grade practices: dual-track ensemble architecture, layered detection, explainable AI, security hardening, containerization, comprehensive testing, and professional documentation.
