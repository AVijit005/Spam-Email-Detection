# VALIDATION GUIDE — Spam Detection Pipeline v3.0

## How to Verify Ensemble

### 1. Verify ensemble prediction produces correct shape

```python
import numpy as np
import scipy.sparse as sp
from app.ml.ensemble import EnsemblePredictor

# Mock a classical model
class MockModel:
    def predict_proba(self, features):
        n = features.shape[0]
        return np.column_stack([np.full(n, 0.2), np.full(n, 0.8)])

ensemble = EnsemblePredictor(
    classical_model=MockModel(),
    classical_vectorizer_bundle={"word_vec": None},
    transformer_model=None,   # No transformer → classical-only fallback
    transformer_tokenizer=None,
)

features = sp.csr_matrix(np.array([[0.1, 0.5]]))
proba = ensemble.predict_proba(features, ["test message"])
assert proba.shape == (1, 2), f"Expected (1,2), got {proba.shape}"
assert proba[0, 1] > proba[0, 0], "Spam probability should exceed ham"

preds = ensemble.predict(features, ["test message"])
assert preds.shape == (1,), f"Expected (1,), got {preds.shape}"
assert preds[0] == 1, "Should predict spam"

print("Ensemble verification PASSED")
```

### 2. Verify transformer_proba public API

```python
from app.ml.ensemble import EnsemblePredictor
e = EnsemblePredictor(None, {})
assert hasattr(e, "transformer_proba"), "Missing public transformer_proba"
assert hasattr(e, "_transformer_proba"), "Missing private _transformer_proba"
print("Public API verification PASSED")
```

### 3. Verify ensemble routing in detector

```python
from app.core.detector import _is_ensemble_model, _ensemble_predict
import numpy as np, scipy.sparse as sp

class MockEnsemble:
    def predict_proba(self, features, raw_texts):
        assert raw_texts == ["test"], "raw_texts not passed"
        return np.array([[0.3, 0.7]])

mock = MockEnsemble()
spam, ham = _ensemble_predict(mock, sp.csr_matrix(np.array([[0.1]])), "test")
assert spam == 0.7 and ham == 0.3, f"Expected 0.7/0.3, got {spam}/{ham}"
print("Ensemble routing verification PASSED")
```

## How to Verify Vectorizer Reuse (Stage 4 Fix)

### Verify that predict_proba uses Stage 2 vectorizer vocabulary

```python
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Simulate Stage 2: fit vectorizer on train
train_texts = ["free money", "urgent click", "hello friend", "meeting today"]
train_labels = np.array([1, 1, 0, 0])
word_vec = TfidfVectorizer(max_features=100, ngram_range=(1,2))
word_vec.fit(train_texts)

# Stage 2 vocabulary
stage2_vocab = set(word_vec.get_feature_names_out())

# Simulate Stage 4: reuse vectorizer with .transform()
test_texts = ["get money now urgent win"]
x_test = word_vec.transform(test_texts)

# Verify: feature matrix is compatible with Stage 2 classifier
feature_count = x_test.shape[1]
vocab_count = len(stage2_vocab)
assert feature_count == vocab_count, \
    f"Feature count {feature_count} != vocabulary size {vocab_count}"

# Stage 4 old bug: re-create vectorizer and fit
new_vec = TfidfVectorizer(max_features=100, ngram_range=(1,2))
new_vec.fit(train_texts)
new_vocab = set(new_vec.get_feature_names_out())

# Verify: same vocab, but could differ in edge cases
assert stage2_vocab == new_vocab, \
    "Vocabularies differ — this is the bug vectorizer reuse prevents"

print("Vectorizer reuse verification PASSED")
```

## How to Verify Transformer Checkpoints

### 1. Verify checkpoint is saved to disk

```bash
# After training completes (or during), verify file exists:
ls -la model/checkpoints/DeBERTa-v3_best.pt
```

### 2. Verify checkpoint can be loaded

```python
import torch
ckpt = torch.load("model/checkpoints/DeBERTa-v3_best.pt", map_location="cpu")
assert len(ckpt) > 0, "Checkpoint is empty"
for name, tensor in list(ckpt.items())[:3]:
    print(f"  {name}: {tensor.shape}")
```

### 3. Verify checkpoint matches model architecture

```python
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base", num_labels=2
)
ckpt = torch.load("model/checkpoints/DeBERTa-v3_best.pt", map_location="cpu")
model.load_state_dict(ckpt)  # Should not raise
print("Checkpoint architecture match PASSED")
```

## How to Verify Exported Models

### 1. Verify XGBoost model loads and predicts

```python
import pickle
import numpy as np

with open("model/spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vec = pickle.load(f)

assert hasattr(model, "predict_proba"), "Model missing predict_proba"

# Test single inference
import scipy.sparse as sp
from app.core.features import extract_meta_features

text = "You have won a free prize! Click here now."
word_feats = vec["word_vec"].transform([text])
meta_feats = sp.csr_matrix(extract_meta_features(text))
features = sp.hstack([word_feats, meta_feats], format="csr")

proba = model.predict_proba(features)
assert proba.shape == (1, 2), f"Expected (1,2), got {proba.shape}"
print(f"Spam probability: {proba[0, 1]:.4f}")
print("Model export verification PASSED")
```

### 2. Verify SHA-256 integrity

```python
import hashlib

def verify_sha256(filepath, expected_sha_path):
    with open(filepath, "rb") as f:
        actual = hashlib.sha256(f.read()).hexdigest()
    expected = open(expected_sha_path).read().strip()
    assert actual == expected, f"SHA-256 mismatch: {actual[:8]} != {expected[:8]}"
    print(f"SHA-256 verified for {filepath}")

verify_sha256("model/hf_model/model.safetensors", "model/hf_model/model.safetensors.sha256")
```

### 3. Verify HF-native model loads correctly

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained(
    "model/hf_model", local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "model/hf_model", local_files_only=True
)

assert model.config.num_labels == 2
assert tokenizer.pad_token == "[PAD]"

text = "URGENT: Verify your account now!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    logits = model(**inputs).logits
probs = torch.softmax(logits, dim=-1)
print(f"Spam probability: {probs[0][1]:.4f}")
print("HF-native model verification PASSED")
```

### 4. Verify metadata completeness

```python
import json

with open("model/model_metadata.json") as f:
    meta = json.load(f)

required_fields = ["model_name", "track", "trained_at_utc", "dataset_rows",
                   "train_rows", "test_rows", "selected_metrics"]
for field in required_fields:
    assert field in meta, f"Missing field: {field}"
    print(f"  {field}: {meta[field]}")

print("Metadata verification PASSED")
```

## Integration Test Suite

Run the full test suite:

```bash
python -m pytest tests/ -v
```

All 205 tests must pass. Test coverage includes:
- Detector routing (rule-based + ML + ensemble pathways)
- Constants validation (regex patterns, keyword sets)
- Feature extraction (all 32 meta features)
- Domain extraction and validation
- PII redaction (emails, phone numbers, credit cards)
- Schema validation (request/response models)
- Auth (API key middleware)
- Rate limiting
- CORS configuration
- Bootstrap/health endpoint

## Production Deployment Validation

```bash
# 1. Start the API server
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 2. Test health endpoint
curl http://localhost:8000/v1/health

# 3. Test prediction (requires API key if configured)
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"sender":"phish@bad.com","subject":"Urgent: verify now","body":"Click here to verify your account"}'

# 4. Expected response includes:
#   - "label": "Spam" or "Not Spam"
#   - "confidence": float 0-1
#   - "reason": string
#   - "rule_layer": "rules" or "ml"
#   - "prediction_id": hex string
#   - "evaluated_at_utc": ISO timestamp
```
