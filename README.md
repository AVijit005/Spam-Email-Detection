# 🛡️ Spam Email Detection — ML-Powered Gmail Protection

**A production-grade spam and phishing detection system with a Chrome extension, FastAPI backend, layered ML detection, explainable predictions, user feedback loop, retraining pipeline, and optional MySQL persistence.**

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-225%20passing-brightgreen.svg)](#testing)
[![Coverage](https://img.shields.io/badge/coverage-full%20production%20module%20coverage-success.svg)](#testing)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## Overview

Spam Email Detection is a complete spam and phishing detection platform that combines a Chrome extension for Gmail with a FastAPI backend running a layered detection engine. The system provides explainable predictions, captures user feedback, and supports retraining — making it suitable as both a real-world tool and a portfolio project demonstrating modern ML engineering practices.

### Why this project stands out

- **5-layer detection pipeline**: Whitelist → Trusted Service Catalog → Rule-Based Spam Detection → Benign Context Guard → Machine Learning Classification
- **Explainable AI**: Every prediction includes explanations showing which tokens and signals influenced the decision
- **Production-grade**: Docker deployment, env-based config, CORS protection, rate limiting, API key authentication, SHA-256 model integrity verification
- **PII redaction**: Emails, phone numbers, IPs, SSNs, and credit card numbers are automatically redacted at the API boundary
- **225 passing tests**: Full coverage of all production modules including integration tests that verify the real bootstrap flow with on-disk model artifacts

---

## Key Features

### Detection Engine
- **5-layer classification pipeline** with progressive confidence scoring
- **TF-IDF vectorization** with word, character, and meta-feature extraction
- **Logistic Regression** model for high accuracy with interpretability
- **Rule-based phishing detection** using curated phrase and keyword matching
- **Benign context guard** that identifies conversational and low-risk promotional emails

### Chrome Extension
- **Gmail integration** via Manifest V3
- **Auto-scan** incoming emails with visual overlay banners
- **Manual scan** from the extension popup
- **Feedback submission** to correct incorrect predictions
- **Options page** with backend URL configuration, timeout, history, and retraining controls

### Security
- **API key authentication** on feedback and retrain endpoints
- **Rate limiting** (60 req/min) with proper 429 responses
- **SHA-256 model integrity verification** — detects tampered model files
- **PII redaction** at the API entry points (email, phone, IP, SSN, credit card)
- **CORS protection** with origin regex (localhost, extension IDs, HTTPS)
- **SQL injection prevention** via table name validation regex

### Feedback & Retraining
- **Feedback loop** with JSONL file storage (default) or MySQL (optional)
- **Feedback-aware retraining** that incorporates user-reviewed samples
- **Retrain concurrency lock** to prevent overlapping training jobs
- **Retrain timeout** with graceful 500 error on training failure
- **Feedback summary API** with per-verdict counts

### Deployment
- **Docker** with multi-stage build and non-root user
- **Docker Compose** with optional MySQL profile
- **Gunicorn + Uvicorn** for production ASGI serving
- **Health check** endpoint and Docker HEALTHCHECK
- **Environment-driven** configuration via `.env` file

---

## Architecture Overview

```
┌──────────────┐     ┌─────────────────────────────────────────────┐
│  Gmail UI    │────▶│              Chrome Extension               │
│              │     │  ┌──────────┐  ┌────────┐  ┌─────────────┐ │
│  Inbox View  │     │  │ content  │  │ popup  │  │  options    │ │
│  Email View  │     │  │   .js    │  │  .js   │  │   page      │ │
└──────────────┘     │  └────┬─────┘  └───┬────┘  └──────┬──────┘ │
                     └───────┼────────────┼───────────────┼────────┘
                             │            │               │
                             ▼            ▼               ▼
                     ┌─────────────────────────────────────────────┐
                     │           FastAPI Backend (:8000)            │
                     │  ┌───────────────────────────────────────┐  │
                     │  │              Middleware               │  │
                     │  │  CORS │ Rate Limit │ Auth (API Key)   │  │
                     │  │                                       │  │
                     │  │       /v1/health    (GET)             │  │
                     │  │       /v1/predict   (POST)            │  │
                     │  │       /v1/predict/batch (POST)        │  │
                     │  │       /v1/feedback  (POST) 🔒         │  │
                     │  │       /v1/feedback/summary (GET)      │  │
                     │  │       /v1/retrain   (POST) 🔒         │  │
                     │  └───────────────────────────────────────┘  │
                     │                                             │
                     │  ┌───────────────────────────────────────┐  │
                     │  │        Detection Pipeline             │  │
                     │  │  1. Whitelist      (user domains)     │  │
                     │  │  2. Trusted Catalog (built-in)        │  │
                     │  │  3. Rule-Based     (phishing signals) │  │
                     │  │  4. Benign Context (conversation)     │  │
                     │  │  5. ML Model       (Logistic Regr)    │  │
                     │  └───────────────────────────────────────┘  │
                     │                                             │
                     │  ┌─────────────┐  ┌──────────────────────┐  │
                     │  │ PII Redact  │  │  SHA-256 Integrity   │  │
                     │  └─────────────┘  └──────────────────────┘  │
                     └──────────────────┬──────────────────────────┘
                                        │
                              ┌─────────┴─────────┐
                              │                   │
                        ┌─────▼─────┐      ┌──────▼──────┐
                        │ feedback  │      │    model    │
                        │  .jsonl   │      │  artifacts  │
                        │  (file)   │      │  (pickle)   │
                        └───────────┘      └─────────────┘
                              │
                        ┌─────▼─────┐
                        │   MySQL   │
                        │ (optional)│
                        └───────────┘
```

For detailed architecture with Mermaid diagrams, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI 0.115+ |
| ML | scikit-learn (LogisticRegression, TfidfVectorizer) |
| NLP | NLTK (stopwords, lemmatization, WordNet) |
| ASGI Server | Uvicorn + Gunicorn |
| Container | Docker + Docker Compose |
| DB (optional) | MySQL 8.0 via PyMySQL |
| Frontend | Chrome Extension (Manifest V3, vanilla JS) |
| Testing | Python unittest (225 tests) |

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/v1/health` | None | Server health, model status, feedback stats |
| `POST` | `/v1/predict` | None | Single email prediction |
| `POST` | `/v1/predict/batch` | None | Batch prediction (max 50) |
| `POST` | `/v1/feedback` | API Key | Submit user label for a prediction |
| `GET` | `/v1/feedback/summary` | None | Aggregate feedback counts |
| `POST` | `/v1/retrain` | API Key | Trigger model retraining |

### Example: Predict

Request:
```json
{
  "sender": "alerts@example.com",
  "subject": "Security alert",
  "body": "Click here to verify your account immediately."
}
```

Response:
```json
{
  "label": "Spam",
  "confidence": 0.92,
  "reason": "Machine learning model detected suspicious patterns",
  "analysis": "AI analysis: 92.0% spam probability based on text and metadata.",
  "model_version": "LogisticRegression-20260403",
  "sender_domain": "example.com",
  "rule_layer": "ml",
  "explanations": [
    "Suspicious token: \"verify\"",
    "Suspicious token: \"click\"",
    "Suspicious signal: contains urgency language",
    "Suspicious signal: contains calls to action"
  ],
  "prediction_id": "a1b2c3d4e5f6",
  "evaluated_at_utc": "2026-04-03T12:00:00+00:00"
}
```

---

## Installation

### Prerequisites

- Python 3.11+
- (Optional) Docker + Docker Compose
- (Optional) MySQL 8.0 for feedback storage

### Quick Start (Local)

```bash
# Clone
git clone https://github.com/your-username/spam-email-detection.git
cd spam-email-detection

# Setup virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .\.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r backend/requirements.txt

# Train the model
python model/train_model.py

# Verify model integrity
python backend/verify_model.py

# Start the server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Docker

```bash
# Build and run backend only
docker compose up --build

# With MySQL
docker compose --profile mysql up --build
```

Health check:
```bash
curl http://127.0.0.1:8000/v1/health
```

---

## Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Default | Description |
|---|---|---|
| `SPAM_API_HOST` | `0.0.0.0` | Server bind address |
| `SPAM_API_PORT` | `8000` | Server port |
| `SPAM_LOG_LEVEL` | `info` | Logging level |
| `SPAM_TRAIN_ON_START` | `false` | Train model on startup |
| `SPAM_RETRAIN_TIMEOUT_SECONDS` | `900` | Retrain timeout |
| `SPAM_SPAM_THRESHOLD` | `0.55` | ML spam classification threshold |
| `SPAM_FEEDBACK_BACKEND` | `file` | `file` or `mysql` |
| `SPAM_DB_HOST` | — | MySQL host |
| `SPAM_DB_USER` | — | MySQL user |
| `SPAM_DB_PASSWORD` | — | MySQL password |
| `SPAM_DB_NAME` | `spam_detector` | MySQL database |
| `SPAM_API_KEY` | `""` | API key for secured endpoints |

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for full deployment guide.

---

## Chrome Extension Setup

1. Open `chrome://extensions/`
2. Enable **Developer mode**
3. Click **Load unpacked** and select the `extension/` folder
4. Open the extension **Options** page
5. Set the backend URL to `http://127.0.0.1:8000` (local) or your deployed HTTPS URL
6. Click **Check Backend** to verify connectivity
7. Open Gmail — emails are auto-scanned with prediction banners

---

## Testing

```bash
# Run all tests
python -m unittest discover -s tests
python -m unittest discover -s backend/tests

# Total: 225 passing tests
# - 205 new tests (unit + integration)
# - 20 legacy tests
```

| Category | Tests | Coverage |
|---|---|---|
| Unit: Registry (SHA-256) | 7 | Full — save, load, verify, tamper, missing |
| Unit: Auth (API key) | 10 | Full — all 3 states + integration |
| Unit: PII Redaction | 12+3 | Full — all 5 patterns + idempotency |
| Unit: Feedback Store | 22 | Full — file + MySQL (mock) |
| Unit: NLP (text preprocessing) | 12 | Full — tokenization, lemmatization, stopwords |
| Unit: Feature Extraction | 29 | Full — all 16 meta features |
| Unit: Explanation Engine | 8 | Full — spam/NSP/edge cases |
| Unit: Rules Engine | 16 | Full — spam, benign, trusted domain |
| Unit: Detector | 16 | Full — all 5 detection layers |
| Unit: Schemas | 18 | Full — max-length, required, defaults |
| Unit: Config | 5 | Full — env vars, defaults, booleans |
| Unit: Domain | 26 | Full — normalize, catalog, whitelist, edges |
| Integration: API Auth | 5 | Full — secured + unsecured paths |
| Integration: Rate Limit | 2 | Full — 429 enforcement verified |
| Integration: API Predict | 2 | Full — 500 on missing model |
| Integration: API Retrain | 4 | Full — 409, timeout, failure, success |
| Integration: CORS | 10 | Full — allow, block, preflight, methods |
| Integration: Bootstrap | 3 | Full — real artifacts, load_resources |

See [docs/TESTING.md](docs/TESTING.md) for detailed test documentation.

---

## Machine Learning Pipeline

The model training pipeline (`model/train_model.py`):

1. **Load** spam dataset from `data/spam.csv`
2. **Split** 80/20 stratify before vectorization (no leakage)
3. **Vectorize** with `TfidfVectorizer` for word and character n-grams
4. **Extract** 16 meta-features (URL count, urgency keywords, caps ratio, etc.)
5. **Train** `LogisticRegression` on the combined feature matrix
6. **Evaluate** on holdout set with accuracy, F1, and confusion matrix
7. **Load** feedback dataset (recent user-labeled samples) with duplicate collapsing
8. **Retrain** final model on full dataset + feedback samples
9. **Save** model, vectorizer, and metadata with SHA-256 integrity hashes

Model metrics (last verified run):
- Holdout accuracy: **97.5%**
- Spam F1 score: **92.2%**

---

## Security Features

Documented in [docs/SECURITY.md](docs/SECURITY.md):

| Feature | Implementation |
|---|---|
| API Authentication | `X-API-Key` header on feedback and retrain endpoints |
| Rate Limiting | 60 req/min via SlowAPI with `SlowAPIMiddleware` |
| Model Integrity | SHA-256 hashing with `hmac.compare_digest` |
| PII Redaction | Regex-based redaction of 5 PII patterns at API boundary |
| CORS Protection | Origin regex: localhost, extension IDs, HTTPS only |
| SQL Protection | Table name validation regex `^[a-zA-Z_][a-zA-Z0-9_]*$` |

---

## Project Structure

```
spam-email-detection/
├── app/                              # Production FastAPI application
│   ├── api/v1/
│   │   ├── feedback.py               # Feedback storage endpoint
│   │   ├── health.py                 # Health check endpoint
│   │   ├── predict.py                # Prediction endpoint
│   │   ├── retrain.py                # Retraining endpoint
│   │   └── router.py                 # API router assembly
│   ├── core/
│   │   ├── auth.py                   # API key authentication
│   │   ├── constants.py              # Keywords, patterns, phrases
│   │   ├── detector.py               # 5-layer prediction engine
│   │   ├── domain.py                 # Domain normalization & loading
│   │   ├── explain.py                # ML prediction explanations
│   │   ├── features.py               # Meta-feature extraction
│   │   ├── rules.py                  # Rule-based & benign detection
│   │   └── text.py                   # NLP text preprocessing
│   ├── ml/
│   │   └── registry.py               # Model save/load with SHA-256
│   ├── schemas/                      # Pydantic request/response models
│   ├── storage/
│   │   └── feedback.py               # File + MySQL feedback storage
│   ├── utils/
│   │   └── pii.py                    # PII redaction patterns
│   ├── config.py                     # Env-driven settings
│   └── main.py                       # App factory, middleware, lifespan
├── backend/                          # Legacy utilities (transitional)
│   ├── feedback_store.py             # Feedback backend resolver
│   ├── run_server.py                 # Legacy entrypoint → app.main:app
│   ├── runtime_config.py             # Runtime configuration
│   ├── spam_detector_core.py         # Core detection utilities
│   ├── tests/                        # Legacy backend tests (20)
│   └── verify_model.py               # Model integrity verification
├── model/
│   └── train_model.py                # Training pipeline
├── extension/                        # Chrome extension (Manifest V3)
│   ├── content.js                    # Gmail DOM integration
│   ├── popup.js / popup.html         # Extension popup UI
│   ├── options.js / options.html     # Extension settings page
│   ├── background.js                 # Service worker
│   ├── utils/domParser.js            # Gmail DOM parsing
│   └── assets/                       # Extension icons
├── data/
│   ├── spam.csv                      # Training dataset
│   ├── trusted_domains.csv           # Trusted service catalog
│   └── whitelist.csv                 # User whitelist
├── tests/                            # New test suite (205 tests)
│   ├── unit/                         # 14 unit test files
│   └── integration/                  # 6 integration test files
├── docs/                             # Documentation
│   ├── ARCHITECTURE.md
│   ├── DEPLOYMENT.md
│   ├── SECURITY.md
│   └── TESTING.md
├── Dockerfile                        # Multi-stage production image
├── docker-compose.yml                # Backend + optional MySQL
├── .env.example                      # Environment template
└── requirements.txt                  # Python dependencies
```

---

## Future Roadmap

- [ ] Scheduled retraining (cron-based or background task)
- [ ] Multi-user support with JWT authentication
- [ ] Admin dashboard for feedback review
- [ ] Model A/B testing infrastructure
- [ ] Real-time email scanning via Gmail API
- [ ] Support for additional email providers (Outlook, Yahoo)
- [ ] CI/CD pipeline with automated model evaluation

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

Built as a capstone ML engineering project demonstrating production-grade practices: layered detection, explainability, security hardening, containerization, testing, and documentation.
