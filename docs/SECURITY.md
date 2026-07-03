# Security Documentation

## Overview

This document covers the security features, threat model, assumptions, and known limitations of the Spam Email Detection system.

## Security Features

### 1. API Authentication

Endpoints requiring authentication: `POST /v1/feedback`, `POST /v1/retrain`

**Implementation**: `app/core/auth.py`

```python
require_api_key(api_key: str | None = Security(_api_key_header)) -> None
```

- Token sent via `X-API-Key` header
- Configured via `SPAM_API_KEY` environment variable
- **Empty string = authentication disabled** (all requests pass through)
- **Non-empty string = strict comparison** — missing or wrong key returns 401
- Uses `fastapi.security.APIKeyHeader` with `auto_error=False`

**Status codes**:
| Scenario | Status |
|---|---|
| No API key configured (`SPAM_API_KEY=""`) | 200 (all requests allowed) |
| Valid key provided | 200 |
| No header sent | 401 |
| Wrong key sent | 401 |

**Test coverage**: 5 unit tests + 5 integration tests covering all 3 states.

### 2. Rate Limiting

**Implementation**: `slowapi` with `SlowAPIMiddleware`

```python
Limiter(key_func=get_remote_address, default_limits=["60/minute"])
```

- Global limit: 60 requests per minute per client IP
- Uses `X-Forwarded-For` or `REMOTE_ADDR` for client identification
- Returns HTTP 429 with standard rate limit headers when exceeded
- **Applies to all routes** — no per-route exemptions
- The limiter is stored on `app.state.limiter` and the exception handler is registered

**Test coverage**: 2 integration tests confirming 429 enforcement on predict and health endpoints.

### 3. Model Integrity Verification (SHA-256)

**Implementation**: `app/ml/registry.py`

```python
_verify_hash(path: Path, expected: str) -> None:
    actual = _compute_hash(path)
    if not hmac.compare_digest(actual, expected):
        raise ModelIntegrityError(...)
```

- Every `save_model()` call writes SHA-256 sidecar files (`.sha256`)
- Every `load_model()` call verifies the hash if a sidecar exists
- Uses `hmac.compare_digest` for constant-time comparison (timing-attack resistant)
- Loads model without verification if no sidecar exists (backward compatibility)
- Returns `None` if model or vectorizer files are missing (graceful degradation)

**Threat model**: Detects tampered or corrupted model files at startup. An attacker who can modify files on disk and recompute the hash would bypass this — it protects against accidental corruption and unsophisticated tampering, not an attacker with filesystem write access.

**Test coverage**: 7 unit tests — save creates hashes, load verifies matching hash, load rejects tampered hash, load returns None when files missing, load succeeds without sidecar.

### 4. PII Redaction

**Implementation**: `app/utils/pii.py`

Redacts 5 PII categories at the API boundary (in `predict_email`):

| Pattern | Regex | Replacement |
|---|---|---|
| Email addresses | `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b` | `[EMAIL]` |
| Phone numbers (US) | `\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b` | `[PHONE]` |
| IP addresses | `\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b` | `[IP]` |
| SSN (US) | `\b\d{3}-\d{2}-\d{4}\b` | `[SSN]` |
| Credit card numbers | `\b(?:\d[ -]*?){13,19}\b` | `[CCARD]` |

**Key properties**:
- Runs **before** the detection pipeline — redacted text never reaches the model or feedback store
- Applies to both `subject` and `body` fields
- Idempotent — running redaction twice produces the same result
- Preserves all non-PII text unchanged
- Non-string inputs return empty strings

**Test coverage**: 12 unit tests — each PII pattern individually, idempotency, empty/None handling, text preservation.

### 5. CORS Protection

**Implementation**: `fastapi.middleware.cors.CORSMiddleware`

```python
CORSMiddleware(
    allow_origins=[],
    allow_origin_regex=r"^(chrome-extension://[a-z]{32,64}|moz-extension://[a-z0-9-]{8,64}|http://localhost(:\d+)?|http://127\.0\.0\.1(:\d+)?)$",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

**Rules**:
- **Allowed origins**: Chrome extensions (32-64 char IDs), Firefox extensions (8-64 char IDs), localhost (any port), 127.0.0.1 (any port)
- **Disallowed**: All other origins, including arbitrary HTTPS origins and HTTP origins
- **Methods**: GET and POST only — PUT, DELETE, PATCH blocked
- **Credentials**: `Access-Control-Allow-Credentials` not sent (`allow_credentials=False`)
- **Headers**: All headers allowed (`*`) — permissive for extension development

**Test coverage**: 10 integration tests — valid origins allowed, invalid origins blocked, PUT/DELETE blocked, no credentials exposed, preflight behavior verified.

### 6. SQL Injection Prevention

**Implementation**: `app/storage/feedback.py`

```python
_TABLE_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
```

- Table names are validated against a strict regex before use in SQL queries
- Rejects any table name containing spaces, semicolons, dashes, or other special characters
- The `SPAM_DB_TABLE` environment variable is the only user-controlled SQL identifier
- All other SQL values use parameterized queries (`%(name)s` placeholders)

**Test coverage**: 6 unit tests — valid table names accepted, SQL injection attempts rejected (`feedback; DROP TABLE users;`), empty string rejected.

## Threat Model

### Assumptions

1. The server is deployed in a trusted environment (VPS, cloud VM, or local machine)
2. Model files on disk are protected by OS-level permissions
3. The `.env` file is not committed to version control (`.gitignore` enforced)
4. Network traffic between the extension and backend uses HTTPS in production
5. The Chrome extension is loaded in developer mode (no Chrome Web Store review)

### Defended Against

| Threat | Protection |
|---|---|
| Unauthenticated feedback/retrain | API key authentication |
| Brute-force API access | Rate limiting (60/min) |
| Corrupted model files | SHA-256 integrity verification |
| PII leakage to feedback/training | Redaction at API boundary |
| Cross-origin attacks from malicious sites | CORS origin restrictions |
| SQL injection via feedback table name | Table name validation regex |
| Timing attacks on hash comparison | `hmac.compare_digest` |

### Not Defended Against

| Threat | Reason |
|---|---|
| Attacker with filesystem write access | Can recompute SHA-256 hashes after tampering |
| Denial of Service (volumetric) | Rate limiting provides basic protection but not full DDoS mitigation |
| Man-in-the-middle attacks (HTTP) | Deployment should use HTTPS; HTTP is allowed for local development |
| Credential stuffing / brute-force API key | API key is a simple string comparison, no lockout or rate limiting on auth failures |
| CSRF | API is stateless and doesn't use cookies; CORSMiddleware blocks cross-origin requests |
| Extension tampering | Loaded unpacked in developer mode; no Chrome Web Store signing |

## Known Limitations

1. **API key is a single shared secret** — no multi-user support, no key rotation, no key revocation. Suitable for single-user or small-team deployments.

2. **Rate limiting uses in-memory storage** — resets on server restart. In a multi-process Gunicorn deployment, limits are per-worker, not global.

3. **PII patterns are US-centric** — phone number regex matches US format; SSN regex matches US format. International PII may not be redacted.

4. **Model integrity is on-disk only** — no runtime integrity verification. If an attacker swaps model files after startup, the running process continues with the compromised model.

5. **No audit logging** — authentication failures, rate limit triggers, and retrain operations are not logged separately from application logs.

6. **Feedback data is not encrypted at rest** — JSONL files and MySQL tables store feedback in plaintext (with PII already redacted).

## Security Configuration Recommendations

### Production Deployment

```bash
# Generate a strong random API key
python -c "import secrets; print(secrets.token_urlsafe(48))"

# Set in .env
SPAM_API_KEY=<generated-key>
SPAM_FEEDBACK_BACKEND=mysql
SPAM_BOOTSTRAP_MODEL_IF_MISSING=false
SPAM_TRAIN_ON_START=false
```

### Docker

```bash
# Mount .env as a secret (don't bake into image)
docker compose --profile mysql up --build
```

### HTTPS Termination

Place nginx in front of the backend:

```nginx
server {
    listen 443 ssl;
    server_name api.example.com;

    ssl_certificate /etc/letsencrypt/live/api.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.example.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

### Model Integrity Monitoring

Add a cron job to periodically verify model integrity:

```bash
# */5 * * * * cd /app && python backend/verify_model.py || alert
```

## Reporting Security Issues

If you discover a security vulnerability, please do not open a public issue. Instead, report it privately via email or GitHub's security advisory feature.
