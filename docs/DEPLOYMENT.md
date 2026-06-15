# Deployment Guide

## Quick Start

### Local Python (5 minutes)

```bash
git clone <repo-url> spam-email-detection
cd spam-email-detection

python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .\.venv\Scripts\activate       # Windows

pip install -r backend/requirements.txt
python model/train_model.py
python backend/verify_model.py
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Verify:
```bash
curl http://127.0.0.1:8000/v1/health
# {"status":"ok","model_loaded":true,"model_version":"Ensemble-XGBoost-DeBERTa-v3",...}
```

## Docker Deployment

### Backend Only

```bash
cp .env.example .env
docker compose up --build
```

Exposes backend on port 8000. Data and model directories are volume-mounted from `./data` and `./model`.

### Backend + MySQL

```bash
cp .env.example .env
# Edit .env: set SPAM_FEEDBACK_BACKEND=mysql and MySQL credentials
docker compose --profile mysql up --build
```

Starts both `backend` and `mysql` containers. MySQL data persists in a named volume `mysql_data`.

### Docker Compose Services

| Service | Container | Port | Profile |
|---|---|---|---|
| `backend` | `spam-detector-backend` | 8000 | (default) |
| `mysql` | `spam-detector-mysql` | 3306 | `mysql` |

### Docker Health Checks

The backend container includes a `HEALTHCHECK` that queries `/v1/health` every 30 seconds.

The MySQL container uses `mysqladmin ping` with a 10-second interval and 10 retries before marking unhealthy.

### Production Considerations

**Use Gunicorn for production**:
```dockerfile
CMD ["gunicorn", "app.main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

**Set a strong API key**:
```
SPAM_API_KEY=<64+ character random string>
```

**Use HTTPS**: Place the backend behind nginx or a cloud load balancer with TLS termination. The Chrome extension requires HTTPS for non-localhost backends.

**Disable model bootstrap on production restarts**:
```
SPAM_BOOTSTRAP_MODEL_IF_MISSING=false
SPAM_TRAIN_ON_START=false
```

## Environment Variables

Copy `.env.example` to `.env` and configure:

### Core Server

| Variable | Default | Description |
|---|---|---|
| `SPAM_API_HOST` | `0.0.0.0` | Server bind address |
| `SPAM_API_PORT` | `8000` | Server port |
| `SPAM_LOG_LEVEL` | `info` | Logging level (`debug`, `info`, `warning`, `error`) |

### Model

| Variable | Default | Description |
|---|---|---|
| `SPAM_TRAIN_ON_START` | `false` | Run training on server startup |
| `SPAM_BOOTSTRAP_MODEL_IF_MISSING` | `true` | Auto-train if model artifacts missing |
| `SPAM_RETRAIN_TIMEOUT_SECONDS` | `900` | Max seconds for retrain subprocess |
| `SPAM_SPAM_THRESHOLD` | `0.55` | ML probability threshold for spam classification |

### Feedback Storage

| Variable | Default | Description |
|---|---|---|
| `SPAM_FEEDBACK_BACKEND` | `file` | `file`, `mysql`, or `auto` |
| `SPAM_DB_HOST` | — | MySQL hostname |
| `SPAM_DB_PORT` | `3306` | MySQL port |
| `SPAM_DB_USER` | — | MySQL username |
| `SPAM_DB_PASSWORD` | — | MySQL password |
| `SPAM_DB_NAME` | `spam_detector` | MySQL database name |
| `SPAM_DB_TABLE` | `feedback_entries` | MySQL table name |

### Security

| Variable | Default | Description |
|---|---|---|
| `SPAM_API_KEY` | `""` | API key for secured endpoints (empty = no auth) |
| `SPAM_JWT_SECRET_KEY` | `""` | Reserved for future JWT auth |

### MySQL Container

| Variable | Default | Description |
|---|---|---|
| `MYSQL_DATABASE` | `spam_detector` | Database to create |
| `MYSQL_ROOT_PASSWORD` | `change-me-in-production` | MySQL root password |
| `MYSQL_PORT_FORWARD` | `3306` | Host port for MySQL |

## Backend Modes

### `SPAM_FEEDBACK_BACKEND=file`
Feedback stored in `data/feedback.jsonl` as newline-delimited JSON. Zero infrastructure dependencies. Suitable for single-instance deployments.

### `SPAM_FEEDBACK_BACKEND=mysql`
Feedback stored in MySQL. Requires `SPAM_DB_HOST`, `SPAM_DB_USER`, and `SPAM_DB_NAME`. The backend auto-creates the feedback table on first write.

### `SPAM_FEEDBACK_BACKEND=auto`
Uses MySQL if `SPAM_DB_HOST`, `SPAM_DB_USER`, and `SPAM_DB_NAME` are all set; otherwise falls back to file. Recommended for flexible deployments.

## Chrome Extension Configuration

1. Install the extension from `chrome://extensions/` (Developer mode → Load unpacked → `extension/` folder)
2. Open extension Options page
3. Set **Backend URL**:
   - Local development: `http://127.0.0.1:8000` or `http://localhost:8000`
   - Production: `https://your-domain.com` (HTTPS required)
4. Click **Check Backend** to verify connectivity
5. Configure scan timeout, history, and auto-scan preferences

### CORS Rules

The backend accepts origins matching this regex:
```
^(chrome-extension://[a-z]{32,64}|moz-extension://[a-z0-9-]{8,64}|http://localhost(:\d+)?|http://127\.0\.0\.1(:\d+)?)$
```

- Chrome extension IDs are auto-detected
- `localhost` and `127.0.0.1` are allowed with any port for development
- Remote deployments **must use HTTPS** — HTTP origins are rejected by CORS

## Deployment Checklist

- [ ] `.env` configured from `.env.example`
- [ ] Model trained (`python model/train_model.py`)
- [ ] Model verified (`python backend/verify_model.py`)
- [ ] API key set (`SPAM_API_KEY`) if auth is desired
- [ ] Feedback backend configured (file or MySQL)
- [ ] If MySQL: database exists, credentials correct
- [ ] CORS origins validated (extension IDs, deployment domain)
- [ ] Health check passing: `curl /v1/health`
- [ ] Extension installed and pointed at the backend URL
- [ ] Predict and feedback tested end-to-end

## Troubleshooting

### Container exits immediately
Check Docker logs: `docker compose logs backend`. Common causes: missing `.env` file, port conflict (8000 already in use), or model artifacts missing with `SPAM_BOOTSTRAP_MODEL_IF_MISSING=false`.

### Model not loaded (health shows `model_loaded: false`)
Run training: `python model/train_model.py` or set `SPAM_BOOTSTRAP_MODEL_IF_MISSING=true`.

### MySQL connection refused
Ensure the MySQL container is healthy: `docker compose ps mysql`. Verify credentials in `.env` match the container environment variables.

### CORS errors in extension
Verify the backend URL in extension options matches the CORS origin regex. For remote backends, ensure HTTPS.

### 429 Too Many Requests
Rate limit of 60 req/min is enforced. Reduce scan frequency or increase the limit in `app/main.py`.

### 409 Retraining conflict
Another retraining job is in progress. Wait for it to complete or restart the backend to clear the lock.

### SHA-256 integrity error on startup
Model files have been tampered with or corrupted. Delete the `.sha256` sidecar files and re-verify, or retrain: `python model/train_model.py`.
