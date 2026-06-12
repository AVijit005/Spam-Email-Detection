# Backend

## Overview

The backend is a FastAPI service that:

- serves health, prediction, feedback, and retraining endpoints
- runs layered spam detection logic
- stores reviewed feedback in JSONL or MySQL
- retrains the saved model from the base dataset plus reviewed feedback

## Core Files

- `app.py.legacy`: Archived legacy API module (replaced by `app/main.py` in production)
- `feedback_store.py`: JSONL/MySQL feedback persistence
- `run_server.py`: deployment-friendly startup entrypoint
- `runtime_config.py`: env-driven runtime configuration
- `spam_detector_core.py`: shared detection logic and prediction explanations
- `model/train_model.py`: training and feedback-aware retraining pipeline
- `verify_model.py`: end-to-end verifier against saved artefacts

## Run Locally

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\.venv\Scripts\python.exe backend\model\train_model.py
.\.venv\Scripts\python.exe backend\run_server.py
```

Windows shortcut:

```powershell
backend\run.bat
```

## Runtime Configuration

Supported env vars:

```powershell
$env:SPAM_API_HOST = "127.0.0.1"
$env:SPAM_API_PORT = "8000"
$env:SPAM_LOG_LEVEL = "info"
$env:SPAM_RETRAIN_TIMEOUT_SECONDS = "900"
$env:SPAM_TRAIN_ON_START = "false"
$env:SPAM_BOOTSTRAP_MODEL_IF_MISSING = "true"
```

`backend/run_server.py` can train automatically when model artefacts are missing.

## Feedback Storage

### File Mode

Default file:

```text
backend/data/feedback.jsonl
```

### MySQL Mode

```powershell
$env:SPAM_FEEDBACK_BACKEND = "mysql"
$env:SPAM_DB_HOST = "127.0.0.1"
$env:SPAM_DB_PORT = "3306"
$env:SPAM_DB_USER = "root"
$env:SPAM_DB_PASSWORD = ""
$env:SPAM_DB_NAME = "spam_detector"
$env:SPAM_DB_TABLE = "feedback_entries"
```

If `SPAM_FEEDBACK_BACKEND` is `auto`, MySQL is used when DB vars are present; otherwise the backend falls back to file storage.

## API Surface

### `GET /health`

Reports:

- model readiness
- active feedback backend
- feedback counts
- feedback rows consumed into training
- trained timestamp
- model version
- spam threshold

### `POST /predict`

Single-email prediction endpoint.

### `POST /predict/batch`

Batch prediction endpoint.

### `POST /feedback`

Stores reviewed labels and verdicts.

### `GET /feedback/summary`

Returns feedback totals.

### `POST /retrain`

Runs the training pipeline again, consumes reviewed feedback from the configured store, saves artefacts, and reloads them into the running API.

## Deployment

Container entrypoint:

```text
python backend/run_server.py
```

Root deployment assets:

- [Dockerfile](/D:/ml/sic-final-project/Dockerfile)
- [docker-compose.yml](/D:/ml/sic-final-project/docker-compose.yml)
- [.env.example](/D:/ml/sic-final-project/.env.example)

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s backend\tests
.\.venv\Scripts\python.exe backend\verify_model.py
```
