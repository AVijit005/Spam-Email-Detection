# Deployment

## Fast Path

### Local Python

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\.venv\Scripts\python.exe backend\run_server.py
```

### VS Code

Use the workspace files:

- [.vscode/launch.json](/D:/ml/sic-final-project/.vscode/launch.json)
- [.vscode/tasks.json](/D:/ml/sic-final-project/.vscode/tasks.json)

Recommended debug profile:

- `Backend: Run Server (MySQL)`

### Docker

```powershell
docker compose up --build
```

### Docker With MySQL

```powershell
docker compose --profile mysql up --build
```

If the backend container should use the Compose MySQL service, set these in `.env`:

```text
SPAM_FEEDBACK_BACKEND=mysql
SPAM_DB_HOST=mysql
SPAM_DB_PORT=3306
SPAM_DB_USER=root
SPAM_DB_PASSWORD=root
SPAM_DB_NAME=spam_detector
```

## Environment Variables

Core backend runtime:

```powershell
$env:SPAM_API_HOST = "0.0.0.0"
$env:SPAM_API_PORT = "8000"
$env:SPAM_LOG_LEVEL = "info"
$env:SPAM_BOOTSTRAP_MODEL_IF_MISSING = "true"
$env:SPAM_TRAIN_ON_START = "false"
$env:SPAM_RETRAIN_TIMEOUT_SECONDS = "900"
```

Feedback backend:

```powershell
$env:SPAM_FEEDBACK_BACKEND = "file"
```

Optional MySQL:

```powershell
$env:SPAM_FEEDBACK_BACKEND = "mysql"
$env:SPAM_DB_HOST = "127.0.0.1"
$env:SPAM_DB_PORT = "3306"
$env:SPAM_DB_USER = "root"
$env:SPAM_DB_PASSWORD = "your-password"
$env:SPAM_DB_NAME = "spam_detector"
$env:SPAM_DB_TABLE = "feedback_entries"
```

## Extension Configuration

The extension options page now supports:

- local backend URLs via `http://localhost` or `http://127.0.0.1`
- deployed backend URLs via `https://...`

Remote HTTP URLs are intentionally blocked. Use HTTPS for deployed backends.

## Startup Behavior

`backend/run_server.py` can bootstrap model artefacts automatically when they are missing.

That behavior is controlled by:

```powershell
$env:SPAM_BOOTSTRAP_MODEL_IF_MISSING = "true"
$env:SPAM_TRAIN_ON_START = "false"
```

## Health Check

After startup:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Check:

- `status`
- `model_loaded`
- `feedback_backend`
- `feedback_store_error`
