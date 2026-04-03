# Development Notes

## Where The Project Ended Up

The project is no longer a basic single-model spam classifier. It is now a layered Gmail spam and phishing detector with:

- extension-side Gmail extraction and UI overlays
- backend-side whitelist, trusted-service, rules, benign-context, and ML layers
- explanation output for predictions
- user feedback capture and retraining
- optional MySQL-backed feedback persistence
- env-driven startup and deployment support

## Final Technical Direction

### Detection Stack

The backend classifies messages in this order:

1. user whitelist
2. trusted-service catalog
3. phishing and spam rules
4. benign-context guardrails
5. ML model

That made the system more stable than relying on ML alone.

### Model Design

The saved model is currently `LogisticRegression`, trained with:

- word TF-IDF features
- char TF-IDF features
- phishing-oriented metadata features

This keeps the classifier explainable while still performing well on the project dataset.

### Training Design

Training now:

- splits before fitting vectorizers to avoid leakage
- evaluates on a clean holdout
- retrains the selected estimator on the full dataset
- can add reviewed feedback samples into training
- records model and feedback metadata in `model_metadata.json`

### Feedback Loop

Reviewed predictions now feed the retraining pipeline through:

- local JSONL storage by default
- optional MySQL storage
- `POST /retrain` from the backend
- retrain controls in the extension options page

## Engineering Changes That Mattered

- shared preprocessing logic moved into `spam_detector_core.py`
- API logic became testable and metadata-backed
- feedback persistence was abstracted into `feedback_store.py`
- runtime config moved to env-driven configuration in `runtime_config.py`
- deployment startup moved to `run_server.py`
- Docker and Compose files were added for reproducible backend startup

## Current Verified Quality

- backend tests: `15/15` passing
- verifier scenarios: `6/6` passing
- holdout accuracy: `0.9750`
- spam F1: `0.9222`

## Remaining Weak Points

- the base dataset is still small and not very modern for real phishing
- the extension has not been manually exercised in every Gmail layout edge case
- retraining is user-triggered, not scheduled
- the backend still has no authentication or multi-user model management
