# Architecture

## System Architecture

```mermaid
graph TB
    subgraph Client
        GM[Gmail UI]
        CE[Chrome Extension]
        GM --> CE
    end

    subgraph Backend["FastAPI Backend :8000"]
        MW[Middleware Layer]
        subgraph MW
            CORS[CORS Filter]
            RL[Rate Limiter]
            AUTH[API Key Auth]
        end

        subgraph API["API Layer /v1"]
            H[GET /health]
            P[POST /predict]
            PB[POST /predict/batch]
            F[POST /feedback]
            FS[GET /feedback/summary]
            R[POST /retrain]
        end

        subgraph Engine["Detection Engine"]
            W[1. Whitelist]
            TC[2. Trusted Catalog]
            RB[3. Rule-Based]
            BC[4. Benign Context]
            ML[5. ML Model]
        end

        subgraph MLSub["ML Subsystem"]
            VEC[TfidfVectorizer]
            LR[LogisticRegression]
            EXP[Explanation Engine]
        end

        subgraph Security["Security"]
            PII[PII Redaction]
            SHA[SHA-256 Integrity]
        end

        FSYS[Feedback Store]
    end

    subgraph Storage["Storage Layer"]
        JSONL[feedback.jsonl]
        MYSQL[MySQL 8.0]
        MODEL[model artifacts]
    end

    CE -->|HTTP/HTTPS| CORS
    CORS --> RL
    RL --> AUTH
    AUTH --> API

    P -->|secured only| F
    R -->|secured only| AUTH

    P --> Engine
    PB --> Engine
    Engine --> MLSub
    MLSub --> EXP
    Engine --> PII
    SHA --> MODEL

    F --> FSYS
    FSYS --> JSONL
    FSYS --> MYSQL
    R --> FSYS
    R --> MODEL
    H --> FSYS
```

## Request Flow

```mermaid
sequenceDiagram
    participant Gmail as Gmail
    participant Ext as Chrome Extension
    participant API as FastAPI
    participant Detector as Detection Engine
    participant Store as Feedback Store

    Gmail->>Ext: User opens email
    Ext->>API: POST /v1/predict
    API->>API: CORS check
    API->>API: Rate limit check
    API->>Detector: predict_email(sender, subject, body)
    Detector->>Detector: PII redaction
    Detector->>Detector: 1. Whitelist?
    Detector->>Detector: 2. Trusted catalog?
    Detector->>Detector: 3. Rule-based spam?
    Detector->>Detector: 4. Benign context?
    Detector->>Detector: 5. ML classification
    Detector->>Detector: Build explanations
    Detector-->>API: PredictionResult
    API-->>Ext: label, confidence, explanations
    Ext-->>Gmail: Display overlay banner

    opt User submits feedback
        Ext->>API: POST /v1/feedback (auth required)
        API->>Store: append_feedback_entry()
        Store-->>API: stored
        API-->>Ext: feedback_id, verdict
    end
```

## Prediction Flow (5-Layer Pipeline)

```mermaid
flowchart TD
    START([Email Received]) --> REDACT[PII Redaction]
    REDACT --> DOMAIN[Extract Sender Domain]
    DOMAIN --> WL{Whitelist?}
    WL -->|Yes| WL_RESULT[Label: whitelisted<br/>Confidence: 1.0]
    WL -->|No| TC{Trusted Catalog?}
    TC -->|Yes| TC_RESULT[Label: Not Spam<br/>Confidence: 0.97<br/>Layer: trusted_service]
    TC -->|No| RULES{Rule-Based Spam?}
    RULES -->|Yes| RULES_RESULT[Label: Spam<br/>Confidence: 0.86-0.99<br/>Layer: rules]
    RULES -->|No| BENIGN{Benign Context?}
    BENIGN -->|Yes| BENIGN_RESULT[Label: Not Spam<br/>Confidence: 0.76-0.82<br/>Layer: benign_context/promo]
    BENIGN -->|No| ML_LAYER[ML Classification]
    ML_LAYER --> BUILD[Build Feature Matrix]
    BUILD --> PREDICT[LogisticRegression.predict_proba]
    PREDICT --> THRESHOLD{spam_prob >= threshold?}
    THRESHOLD -->|Yes| SPAM[Label: Spam<br/>Layer: ml]
    THRESHOLD -->|No| HAM[Label: Not Spam<br/>Layer: ml]
    SPAM --> EXPLAIN
    HAM --> EXPLAIN
    EXPLAIN[Generate Explanations] --> RESPONSE([Return PredictionResult])
```

### Layer Details

| Layer | Decision Logic | Confidence | Explanation |
|---|---|---|---|
| **Whitelist** | Sender domain in user whitelist CSV | 1.0 | "Trusted sender matched your local whitelist" |
| **Trusted Catalog** | Sender domain in built-in service catalog | 0.97 | "Trusted service domain matched the curated built-in catalog" |
| **Rule-Based** | >=2 spam phrases or >=1 phrase + >=2 signals | 0.86–0.99 | Matched phrases and indicator signals |
| **Benign Context** | Conversational wording, no links/urgency | 0.82 | "Benign-context detection found conversational wording without phishing indicators" |
| **Benign Promo** | Promotional wording, no links, low caps | 0.76 | "Benign promotional detection found retail language without phishing indicators" |
| **ML Model** | LogisticRegression spam probability >= threshold | 0.00–0.99 | Top contributing features from model coefficients |

## Feature Matrix

The ML model operates on a combined feature matrix built from:

```mermaid
flowchart LR
    RAW[Raw Email Text] --> PREPROC[NLP Preprocessing]
    PREPROC --> WORD[Word TF-IDF]
    PREPROC --> CHAR[Char TF-IDF]
    RAW --> META[Meta Features]

    WORD --> CONCAT[sp.hstack]
    CHAR --> CONCAT
    META --> CONCAT
    CONCAT --> MATRIX[Combined CSR Matrix]
    MATRIX --> MODEL[LogisticRegression]
    MODEL --> PROB[spam_prob, ham_prob]
```

### Meta Features (16-dim)

| # | Feature | Description |
|---|---|---|
| 1 | `url_count` | Number of URLs detected |
| 2 | `caps_ratio` | Ratio of uppercase letters |
| 3 | `exclamation_count` | Count of `!` characters |
| 4 | `question_count` | Count of `?` characters |
| 5 | `money_count` | Money amounts ($, €, £) detected |
| 6 | `phone_count` | Phone numbers detected |
| 7 | `word_count` | Total word count |
| 8 | `avg_word_length` | Average word length |
| 9 | `digit_ratio` | Ratio of digit characters |
| 10 | `spam_phrase_hits` | Known phishing phrase matches |
| 11 | `urgency_hits` | Urgency keyword matches |
| 12 | `account_hits` | Account/security keyword matches |
| 13 | `call_to_action_hits` | CTA keyword matches |
| 14 | `symbol_ratio` | Ratio of symbol characters |
| 15 | `percent_hits` | Count of `%` characters |
| 16 | `mixed_token_hits` | Mixed letter-number tokens |

## Retraining Flow

```mermaid
sequenceDiagram
    participant Ext as Extension
    participant API as /v1/retrain
    participant Lock as RETRAIN_LOCK
    participant SP as subprocess.run
    participant Train as train_model.py
    participant Load as load_resources()

    Ext->>API: POST /v1/retrain (auth required)
    API->>Lock: acquire(blocking=False)

    alt Lock acquired
        API->>SP: subprocess.run(train_model.py)
        SP->>Train: Execute training pipeline

        alt Success (returncode=0)
            Train-->>SP: model artifacts saved
            SP-->>API: CompletedProcess
            API->>Load: Reload model from disk
            Load-->>API: New model loaded
            API-->>Ext: 200 - model_version, metrics
        else Timeout
            SP-->>API: TimeoutExpired
            API-->>Ext: 500 - "Retraining timed out"
        else Failure
            SP-->>API: returncode != 0
            API-->>Ext: 500 - stderr output
        end

        API->>Lock: release()
    else Lock not acquired
        API-->>Ext: 409 - "Retraining already in progress"
    end
```

## Storage Flow

```mermaid
flowchart TD
    FEEDBACK[POST /v1/feedback] --> RESOLVE[resolve_feedback_store]
    RESOLVE --> MODE{SPAM_FEEDBACK_BACKEND}
    MODE -->|file| FILE[append_feedback_file<br/>Write JSONL line]
    MODE -->|mysql| MYSQL[append_feedback_mysql<br/>CREATE TABLE IF NOT EXISTS<br/>INSERT with parameterized query]
    MODE -->|auto| CHECK{DB_HOST + DB_USER + DB_NAME?}
    CHECK -->|configured| MYSQL
    CHECK -->|not configured| FILE

    RETRAIN[POST /v1/retrain] --> TRAIN[train_model.py]
    TRAIN --> LOAD_FB[load_feedback_entries]
    LOAD_FB --> FB_MODE{feedback backend}
    FB_MODE -->|file| FB_FILE[Read JSONL lines]
    FB_MODE -->|mysql| FB_MYSQL[SELECT * FROM table]
    FB_FILE --> COLLAPSE[Collapse duplicates<br/>Map labels to 0/1]
    FB_MYSQL --> COLLAPSE
    COLLAPSE --> TRAIN_ML[Train with feedback samples]
```

## Security Flow

```mermaid
flowchart TD
    REQ[Incoming Request] --> CORS{CORS Origin?}
    CORS -->|Invalid| REJECT_CORS[No CORS headers]
    CORS -->|Valid| RATE{Rate Limit?}
    RATE -->|Exceeded| REJECT_429[429 Too Many Requests]
    RATE -->|OK| AUTH{Auth Required?}

    AUTH -->|No| PROCESS[Process Request]
    AUTH -->|Yes| KEY{API Key Valid?}
    KEY -->|No| REJECT_401[401 Unauthorized]
    KEY -->|Yes| PROCESS

    PROCESS --> PII[PII Redaction]
    PII --> DETECT[Detection Pipeline]
    DETECT --> RESPONSE[Response]
```

### Model Integrity on Startup

```mermaid
flowchart TD
    START[App Startup] --> LOAD[load_model]
    LOAD --> EXISTS{model & vectorizer exist?}
    EXISTS -->|No| RETURN_NONE[Return None<br/>Predict returns 500]
    EXISTS -->|Yes| HASH{Hash file exists?}
    HASH -->|No| LOAD_FILES[Load pickle files without check]
    HASH -->|Yes| VERIFY[hmac.compare_digest]
    VERIFY -->|Match| LOAD_FILES
    VERIFY -->|Mismatch| ERROR[ModelIntegrityError<br/>App crashes]
```

## Module Dependency Graph

```mermaid
graph TD
    MAIN[app/main.py] --> CONFIG[app/config.py]
    MAIN --> ROUTER[app/api/v1/router.py]
    MAIN --> DOMAIN[app/core/domain.py]
    MAIN --> REGISTRY[app/ml/registry.py]
    MAIN --> PREDICT[app/api/v1/predict.py]
    MAIN --> HEALTH[app/api/v1/health.py]
    MAIN --> FEEDBACK[app/api/v1/feedback.py]
    MAIN --> RETRAIN[app/api/v1/retrain.py]

    ROUTER --> PREDICT
    ROUTER --> HEALTH
    ROUTER --> FEEDBACK
    ROUTER --> RETRAIN

    PREDICT --> DETECTOR[app/core/detector.py]
    HEALTH --> FEEDBACK_STORE[app/storage/feedback.py]

    DETECTOR --> DOMAIN
    DETECTOR --> EXPLAIN[app/core/explain.py]
    DETECTOR --> FEATURES[app/core/features.py]
    DETECTOR --> RULES[app/core/rules.py]
    DETECTOR --> TEXT[app/core/text.py]
    DETECTOR --> PII_UTILS[app/utils/pii.py]

    FEATURES --> CONSTANTS[app/core/constants.py]
    RULES --> CONSTANTS
    RULES --> FEATURES
    TEXT --> CONSTANTS
    EXPLAIN --> CONSTANTS

    RETRAIN --> MAIN
    RETRAIN --> FEEDBACK_STORE
    RETRAIN --> AUTH[app/core/auth.py]
    FEEDBACK --> AUTH
    FEEDBACK --> FEEDBACK_STORE

    REGISTRY --> MODEL_FILES[(model artifacts)]
    DOMAIN --> CSV_FILES[(CSV data files)]
    FEEDBACK_STORE --> JSONL[(feedback.jsonl)]
    FEEDBACK_STORE --> MYSQL_DB[(MySQL)]
```

## Key Design Decisions

1. **Module-level model state**: The prediction, health, feedback, and retrain modules hold their state as module-level variables rather than dependency injection. This simplifies the codebase and avoids passing state through every function, at the cost of test isolation complexity (tests patch module state).

2. **CSV for configuration data**: Whitelist and trusted domain catalogs use CSV files rather than a database. This keeps the system self-contained and deployable with zero infrastructure dependencies. CSV files are read once at startup.

3. **JSONL for feedback**: Feedback entries are stored as newline-delimited JSON, making the storage human-readable, version-controllable, and trivially portable. MySQL is offered as an optional upgrade path for multi-instance deployments.

4. **SHA-256 sidecar files**: Model integrity uses hash sidecar files (`model.pkl.sha256`) rather than embedded checksums, allowing hash verification to be retrofitted to existing models and updated independently.

5. **Concurrency lock on retraining**: A `threading.Lock` prevents overlapping retraining jobs. The lock is non-blocking — concurrent requests receive 409 instead of queuing.

6. **PII at the boundary**: PII redaction occurs at the API entry point (`predict_email`) rather than in storage, ensuring redacted data never reaches the feedback store or model training pipeline.
