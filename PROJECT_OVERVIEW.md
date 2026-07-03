# Project Overview

## Vision

Spam Email Detection is a production-grade spam and phishing detection platform that brings enterprise ML practices to everyday email protection. It combines a Chrome extension for real-time Gmail scanning with a FastAPI backend running a dual-track ensemble of classical machine learning and transformer models — all backed by a user feedback loop, explainable predictions, and deployment-ready artifacts.

The project is designed to serve three audiences simultaneously:

- **End users** — anyone who wants to detect spam and phishing in their Gmail inbox with visual overlays and explainable results
- **ML engineers** — a reference architecture for dual-track ensemble systems, showing how to combine XGBoost with DeBERTa-v3 in a production pipeline
- **Researchers and students** — a complete, documented, tested ML project that demonstrates the full lifecycle from data preprocessing to deployment

---

## Design Philosophy

### Defense in Depth

The system applies five layers of progressively sophisticated analysis before reaching the ML model:

1. **User Whitelist** — immediate trust for sender domains the user has explicitly approved
2. **Trusted Service Catalog** — high-confidence clearance for known legitimate services (banks, payment processors, major platforms)
3. **Rule-Based Spam Detection** — pattern matching against curated phishing phrase libraries and behavioral indicator signals
4. **Benign Context Guard** — reverse-spam detection: identifies conversational and low-risk promotional emails that should not be flagged
5. **Dual-Track ML Ensemble** — XGBoost + DeBERTa-v3 late fusion for cases requiring deep analysis

This layered approach means the ML model only processes emails that genuinely need sophisticated analysis — reducing latency, improving user trust (rules and catalogs can be inspected and understood), and ensuring that edge cases degrade gracefully rather than crashing.

### Why a Dual-Track Architecture?

Spam and phishing detection presents two distinct challenges that a single model struggles to handle simultaneously:

| Challenge | Best Handled By | Why |
|---|---|---|
| **Keyword/pattern spam** ("FREE MONEY CLICK HERE") | Classical ML (XGBoost) | TF-IDF n-grams excel at surface-level token patterns. XGBoost handles sparse, high-dimensional feature spaces efficiently. |
| **Sophisticated phishing** (context-aware social engineering) | Transformer (DeBERTa-v3) | Disentangled attention captures semantic relationships. Contextual embeddings understand nuance that bag-of-words misses. |
| **Boundary cases** (legitimate marketing vs phishing) | Ensemble fusion | Weighted combination reduces false positives while maintaining recall. Both models must agree on ambiguous cases. |

A single XGBoost model would miss sophisticated phishing that uses natural language. A single DeBERTa model would be overkill for obvious keyword spam and would be slower at inference. The ensemble gives us the best of both worlds.

### Why Late Fusion Over Stacking or Meta-Learners?

The ensemble uses **weighted late fusion** rather than a learned meta-classifier (stacking):

- **Simplicity**: `p_spam = w · p_classical + (1-w) · p_transformer` — a single scalar weight, no additional parameters
- **Robustness**: Grid-searching 21 values on training OOF predictions avoids overfitting a meta-learner on limited holdout data
- **Interpretability**: The fusion weight directly tells us how much the system trusts each track
- **Graceful degradation**: If the transformer artifact is unavailable, the system falls back to classical-only without any code changes

### Explainability as a First-Class Requirement

Every prediction includes explanations showing which tokens and signals influenced the decision. For the classical model, this means extracting top positive/negative feature coefficients. For spam detections, users see actionable reasons: "Suspicious token: 'verify'", "Suspicious signal: contains urgency language." This transparency builds user trust and helps identify model weaknesses.

---

## Architecture Decisions

### Why XGBoost (not LogisticRegression) for Classical ML

The v1.0 baseline used LogisticRegression for its simplicity and interpretability. v3.0 upgraded to XGBoost because:

- **Non-linear feature interactions**: Spam signals often interact — "money" + "click" is far more suspicious than either alone. Tree-based models capture these naturally.
- **Sparse feature handling**: TF-IDF matrices with 25,000+ features are inherently sparse. XGBoost's sparsity-aware split finding is more efficient than LogisticRegression's dense gradient computation.
- **Robustness to irrelevant features**: With 32 meta-features, some may be noisy. Tree-based models automatically ignore irrelevant splits.

LogisticRegression remains as a fallback when the system is trained on small datasets (<5,000 rows) where XGBoost may overfit.

### Why DeBERTa-v3 (not BERT or RoBERTa)

DeBERTa-v3-base was selected as the primary transformer after evaluating:

- **Disentangled attention**: Separates content and position embeddings, improving phishing text understanding where token position (e.g., "click _here_" vs "here, click") matters
- **ELECTRA-style pre-training**: Replaced token detection (RTD) pre-training objective produces better representations for classification tasks than masked language modeling alone
- **Perplexity-F1 correlation**: DeBERTa-v3 consistently achieves higher F1 scores on short-text classification benchmarks compared to BERT-base and RoBERTa-base at comparable model sizes (~184MB)

Alternative transformers (RoBERTa, ELECTRA, ModernBERT, DistilBERT, BERT-base) are supported as configurable options in the training pipeline for experimentation.

### Why CSV for Configuration Data

Whitelist and trusted domain catalogs use CSV files rather than a database because:

- **Zero infrastructure dependency**: The system deploys with a single `pip install` — no database setup required
- **Human-editable**: Users can add trusted domains by editing a spreadsheet-format file
- **Version-controllable**: CSV diffs are readable in pull requests
- **Sufficient for scale**: Domain catalogs rarely exceed a few thousand entries

### Why JSONL for Feedback

User feedback is stored as newline-delimited JSON:

- **Append-only**: New entries append to the end without rewriting — safe for concurrent processes
- **Human-readable**: Each line is a valid JSON object, inspectable with standard tools
- **Portable**: JSONL files can be copied, backed up, and processed by any language
- **Optional MySQL upgrade**: When multi-instance deployment is needed, a single env var (`SPAM_FEEDBACK_BACKEND=mysql`) switches to MySQL

---

## Tradeoffs

### Accuracy vs Inference Latency

| Component | Latency (single email) | Memory |
|---|---|---|
| Classical (XGBoost) | ~2-5 ms | ~2 MB |
| Transformer (DeBERTa-v3) | ~40-80 ms | ~184 MB |
| Ensemble (combined) | ~45-85 ms | ~186 MB |

The ensemble adds negligible latency overhead (a single scalar multiplication) over the transformer path, but requires both models to be loaded in memory (~186 MB total).

For CPU-only deployments, the transformer inference time increases to ~200-500 ms, which is still acceptable for the extension's async scanning model but may feel slow for batch processing.

### Training Cost vs Model Quality

| Training Mode | Time | GPU Required | Best F1 |
|---|---|---|---|
| Fast-dev (500 rows) | ~5 min | No | Lower |
| Classical only | ~35 min | No | ~0.97 |
| Transformer only | ~90 min | Yes (T4+) | ~0.99 |
| Full ensemble | ~2.5 hours | Yes (T4+) | ~0.99+ |

The classical-only track provides a strong baseline without GPU dependency. Full ensemble training requires a T4 or better GPU but delivers marginal gains over transformer-only in exchange for better robustness on boundary cases.

### Model Size and Deployment

The ensemble requires ~186 MB of model artifacts (XGBoost ~2 MB + DeBERTa-v3 ~184 MB). This is manageable for server deployments but too large for browser-based inference. The Chrome extension is intentionally a thin client — all ML inference happens on the backend.

---

## Scalability

### Current Design

The system is designed for single-instance deployment with optional horizontal scaling:

- **Single server**: 4 Gunicorn workers handle ~200 concurrent requests with sub-100ms latency
- **Feedback storage**: JSONL for single-instance, MySQL for multi-instance
- **Retraining**: Serialized by a `threading.Lock` — intentional constraint to prevent resource contention

### Future Scaling Paths

- **Model serving separation**: Deploy the transformer model on a dedicated GPU instance while the classical model serves CPU-bound requests
- **Async inference queue**: For batch processing (e.g., scanning an entire inbox), queue predictions and process in parallel
- **Model distillation**: Train a smaller student model (DistilBERT-size) from the DeBERTa-v3 teacher for faster inference
- **Caching**: Cache predictions for identical email content (newsletters, automated alerts)

---

## Production Readiness

The project includes several features that move it beyond a prototype:

- **SHA-256 integrity verification**: All model artifacts are hashed at save time and verified at load time using constant-time comparison. Tampered or corrupted models are detected immediately.
- **PII redaction at the API boundary**: Emails, phone numbers, IPs, SSNs, and credit card numbers are redacted before they reach the model or feedback store. No PII is ever persisted.
- **Rate limiting and authentication**: 60 requests/minute per IP, API key authentication on mutation endpoints (feedback, retrain).
- **Graceful degradation**: Missing transformer artifact? Ensemble falls back to classical. Missing model entirely? Returns HTTP 500 rather than crashing. No config? Sensible defaults for everything.
- **Docker with non-root user**: Multi-stage builds reduce image size. Health checks monitor runtime. Compose profiles separate required and optional services.
- **225 deterministic tests**: Every production module is covered by isolated, fast tests that run in under 5 seconds total.
