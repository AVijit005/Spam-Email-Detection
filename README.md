# Spam Email Detection

Spam and phishing detection for Gmail, built as a Chrome extension plus a FastAPI backend with explainable predictions, user feedback capture, retraining, and optional MySQL-backed feedback storage.

## Current Project State

- Layered detection: whitelist, trusted-service catalog, rule-based phishing checks, benign-context rules, and ML classification
- Explainable predictions in both the backend response and extension UI
- Feedback loop with `/feedback`, `/feedback/summary`, and `/retrain`
- Feedback-aware retraining from either local JSONL storage or MySQL
- Deployment-ready backend startup with env-based config, Docker, and Docker Compose
- Extension options page for backend URL, timeout, history, auto-scan, and retraining

## Architecture

```text
Gmail UI
  -> Chrome extension
     -> FastAPI backend
        -> layered spam detector
        -> feedback store (JSONL or MySQL)
        -> retraining pipeline
```

## Project Structure

```text
sic-final-project/
├── backend/
│   ├── data/
│   │   ├── spam.csv
│   │   ├── trusted_domains.csv
│   │   └── whitelist.csv
│   ├── model/
│   │   └── train_model.py
│   ├── tests/
│   ├── app.py
│   ├── feedback_store.py
│   ├── run_server.py
│   ├── runtime_config.py
│   ├── spam_detector_core.py
│   ├── verify_model.py
│   └── requirements.txt
├── extension/
│   ├── background.js
│   ├── content.css
│   ├── content.js
│   ├── manifest.json
│   ├── options.css
│   ├── options.html
│   ├── options.js
│   ├── popup.css
│   ├── popup.html
│   └── popup.js
├── .env.example
├── docker-compose.yml
├── Dockerfile
└── README.md
```

## Local Backend Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\.venv\Scripts\python.exe backend\model\train_model.py
.\.venv\Scripts\python.exe backend\verify_model.py
.\.venv\Scripts\python.exe backend\run_server.py
```

Windows shortcut:

```powershell
backend\run.bat
```

## VS Code Run / Debug

Workspace files were added in [.vscode/launch.json](/D:/ml/sic-final-project/.vscode/launch.json), [.vscode/tasks.json](/D:/ml/sic-final-project/.vscode/tasks.json), and [.vscode/settings.json](/D:/ml/sic-final-project/.vscode/settings.json).

In VS Code:

1. Open the project folder.
2. Open `Run and Debug`.
3. Choose `Backend: Run Server (MySQL)`.
4. Enter your MySQL password when prompted.
5. Start debugging.

Useful tasks:

- `Install Backend Requirements`
- `Train Model (MySQL Feedback)`
- `Verify Model`
- `Health Check`
- `Setup Backend (Install + Train + Verify)`

Default backend URL:

```text
http://127.0.0.1:8000
```

## Run Process

### Option 1: Run In VS Code

1. Open the project folder in VS Code.
2. Press `Ctrl+Shift+P`.
3. Run `Tasks: Run Task`.
4. Choose `Setup Backend (Install + Train + Verify)`.
5. Enter your MySQL password when prompted.
6. Open `Run and Debug`.
7. Select `Backend: Run Server (MySQL)`.
8. Press `F5`.
9. Enter your MySQL password again when prompted.

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

You should see:

- `status: ok`
- `model_loaded: true`
- `feedback_backend: mysql`

### Option 2: Run In PowerShell

```powershell
cd D:\ml\sic-final-project

.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt

$env:SPAM_FEEDBACK_BACKEND = "mysql"
$env:SPAM_DB_HOST = "127.0.0.1"
$env:SPAM_DB_PORT = "3306"
$env:SPAM_DB_USER = "root"
$env:SPAM_DB_PASSWORD = "your-password"
$env:SPAM_DB_NAME = "spam_detector"
$env:SPAM_DB_TABLE = "feedback_entries"

.\.venv\Scripts\python.exe backend\model\train_model.py
.\.venv\Scripts\python.exe backend\run_server.py
```

Windows shortcut:

```powershell
backend\run.bat
```

### Load The Chrome Extension

1. Open `chrome://extensions/`.
2. Enable Developer mode.
3. Click `Load unpacked`.
4. Select the `extension` folder.
5. Open the extension options page.
6. Set backend URL to `http://127.0.0.1:8000`.
7. Click `Check Backend`.

### Use The Project

1. Open Gmail.
2. Open an email.
3. Let auto-scan run or use the popup manually.
4. Review the prediction banner and explanations.
5. If a prediction is wrong, submit feedback.
6. Use `Retrain From Feedback` from the extension options page when needed.

## Deployment Readiness

The backend now supports:

- env-driven host, port, log level, retraining timeout, and startup behavior
- automatic model bootstrap when artefacts are missing
- Docker image startup through [backend/run_server.py](/D:/ml/sic-final-project/backend/run_server.py)
- Docker Compose for backend-only or backend-plus-MySQL flows
- remote HTTPS backend URLs from the extension

### Environment Variables

Key backend env vars:

```powershell
$env:SPAM_API_HOST = "0.0.0.0"
$env:SPAM_API_PORT = "8000"
$env:SPAM_LOG_LEVEL = "info"
$env:SPAM_BOOTSTRAP_MODEL_IF_MISSING = "true"
$env:SPAM_TRAIN_ON_START = "false"
$env:SPAM_RETRAIN_TIMEOUT_SECONDS = "900"
```

### Docker

Build and run:

```powershell
docker compose up --build
```

This starts the backend container and exposes it on port `8000` by default.

To include MySQL too:

```powershell
docker compose --profile mysql up --build
```

Use [.env.example](/D:/ml/sic-final-project/.env.example) as your starting point for deployment configuration.

If the backend container should use the Compose MySQL service, set these in your `.env`:

```text
SPAM_FEEDBACK_BACKEND=mysql
SPAM_DB_HOST=mysql
SPAM_DB_PORT=3306
SPAM_DB_USER=root
SPAM_DB_PASSWORD=root
SPAM_DB_NAME=spam_detector
```

## Feedback Storage

By default, reviewed feedback is stored in:

```text
backend/data/feedback.jsonl
```

Optional MySQL-backed feedback storage:

```powershell
$env:SPAM_FEEDBACK_BACKEND = "mysql"
$env:SPAM_DB_HOST = "127.0.0.1"
$env:SPAM_DB_PORT = "3306"
$env:SPAM_DB_USER = "root"
$env:SPAM_DB_PASSWORD = "your-password"
$env:SPAM_DB_NAME = "spam_detector"
$env:SPAM_DB_TABLE = "feedback_entries"
```

If `SPAM_FEEDBACK_BACKEND` is `auto`, the backend uses MySQL when DB variables are present and otherwise falls back to JSONL.

## Extension Setup

1. Open `chrome://extensions/`.
2. Enable Developer mode.
3. Click Load unpacked.
4. Select the `extension` folder.
5. Open the extension options page.
6. Point the extension at either:
   - `http://127.0.0.1:8000` for local use
   - `https://your-domain.example` for a deployed backend

Rules:

- local development may use `http://localhost` or `http://127.0.0.1`
- deployed backends must use `https://`

## API

### `GET /health`

Reports:

- backend readiness
- model version and threshold
- active feedback backend
- feedback count
- feedback rows already consumed into training
- trusted-domain and whitelist counts

### `POST /predict`

```json
{
  "sender": "alerts@example.com",
  "subject": "Security alert",
  "body": "Click here to verify your account."
}
```

Response includes:

- label, confidence, reason, analysis
- rule layer and sender domain
- explanation cues and signals
- prediction ID and evaluation timestamp

### `POST /predict/batch`

```json
{
  "emails": [
    {
      "sender": "alerts@example.com",
      "subject": "Security alert",
      "body": "Click here to verify your account."
    }
  ]
}
```

### `POST /feedback`

Stores the user-reviewed label for a prediction.

### `GET /feedback/summary`

Returns aggregate feedback counts.

### `POST /retrain`

Retrains the backend from:

- the base spam dataset
- all valid reviewed feedback rows from the configured feedback store

The running API reloads the new artefacts after retraining completes.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s backend\tests
.\.venv\Scripts\python.exe backend\verify_model.py
```

Current verified state:

- backend tests: `15/15` passed
- verifier scenarios: `6/6` passed
- saved model: `LogisticRegression`
- holdout accuracy: `0.9750`
- spam F1: `0.9222`

## Notes

- `backend/data/whitelist.csv` is the only source of automatic `whitelisted` decisions.
- `backend/data/trusted_domains.csv` is a curated trusted-service catalog, not a bypass whitelist.
- model artefacts are generated locally and ignored by git.
- feedback is now automatically consumed by retraining from either JSONL or MySQL.
