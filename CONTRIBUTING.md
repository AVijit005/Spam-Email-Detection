# Contributing

Thank you for your interest in contributing to Spam Email Detection. This guide covers everything you need to get started.

---

## Development Setup

### Prerequisites

- Python 3.11+
- Git
- (Optional) Docker for containerized development

### First-Time Setup

```bash
# Clone the repository
git clone https://github.com/AVijit005/Spam-Email-Detection.git
cd Spam-Email-Detection

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .\.venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Quick smoke test: train on 500 rows (~5 minutes)
python model/train_model.py --fast-dev

# Run the full test suite
python -m unittest discover -s tests -v
python -m unittest discover -s backend/tests -v
```

All 225 tests should pass before you start making changes.

---

## Project Structure

```
spam-email-detection/
├── app/                    # Production FastAPI application
│   ├── api/v1/             # REST endpoints (predict, feedback, health, retrain)
│   ├── core/               # Detection engine (pipeline, features, rules, text, explain)
│   ├── ml/                 # ML subsystem (ensemble, model registry)
│   ├── schemas/            # Pydantic request/response models
│   ├── storage/            # Feedback persistence (JSONL, MySQL)
│   └── utils/              # PII redaction utilities
├── model/                  # Training scripts, checkpoints, serialized artifacts
│   ├── train_model.py      # 6-stage training orchestrator
│   ├── train_classical.py  # Track A: classical ML pipeline
│   ├── train_transformer.py # Track B: transformer fine-tuning
│   └── shared.py           # Shared evaluation/metrics utilities
├── extension/              # Chrome extension (Manifest V3)
│   ├── content.js          # Gmail DOM integration
│   ├── background.js       # Service worker
│   ├── popup.js            # Extension popup UI
│   ├── options.js          # Settings page
│   └── utils/domParser.js  # Gmail DOM parsing
├── tests/                  # Test suite (205 tests)
│   ├── unit/               # 14 unit test files
│   └── integration/        # 6 integration test files
├── backend/                # Legacy utilities (kept for reference)
├── data/                   # Datasets, whitelists, feedback store
├── docs/                   # Architecture, deployment, security, testing docs
├── Dockerfile              # Multi-stage production image
├── docker-compose.yml      # Backend + optional MySQL
├── .env.example            # Environment template
└── requirements.txt        # Python dependencies
```

---

## Branch Strategy

| Branch | Purpose |
|---|---|
| `main` | Stable, tagged releases. All PRs merge here after review. |
| `pre-kaggle-final` | Active development branch for v3.0 Kaggle training. |
| `feature/*` | New features. Branch from `main`, merge back via PR. |
| `fix/*` | Bug fixes. Branch from `main`, merge back via PR. |
| `docs/*` | Documentation-only changes. |

### Branch Naming

```
feature/add-gmail-api-integration
fix/ensemble-routing-crash
docs/update-readme-badges
```

---

## Pull Request Workflow

1. **Fork** the repository (if external contributor) or create a feature branch
2. **Branch** from `main`: `git checkout -b feature/my-feature`
3. **Code** your changes following the coding standards below
4. **Test** — all existing tests must pass, and new features need tests:
   ```bash
   python -m unittest discover -s tests -v
   ```
5. **Commit** using conventional commit messages:
   ```bash
   git commit -m "feat: add batch prediction with configurable batch size"
   git commit -m "fix: resolve ensemble routing crash in predict endpoint"
   git commit -m "docs: add Kaggle training recovery procedures"
   ```
6. **Push** and open a pull request against `main`
7. **Review** — address any feedback from maintainers

### PR Checklist

- [ ] All 225 tests pass locally
- [ ] New code includes tests
- [ ] No new linting warnings introduced
- [ ] Documentation updated if behavior changes
- [ ] Commit messages follow conventional commit format
- [ ] Branch is up to date with `main`

---

## Coding Standards

### Python

- **Style**: [PEP 8](https://peps.python.org/pep-0008/) with 100-character line limit
- **Type hints**: Use for all function signatures. The codebase currently uses minimal type hints — new code should lead by example.
- **Docstrings**: Use triple-quoted docstrings for any new public functions. Describe parameters, return values, and raised exceptions.
- **Imports**: Standard library first, third-party second, local third. Sort alphabetically within each group.
- **Naming**: `snake_case` for functions and variables, `PascalCase` for classes, `UPPER_CASE` for constants.

```python
def predict_email(
    sender: str,
    subject: str,
    body: str,
    model: Any | None = None,
) -> PredictionResult:
    """Run the 5-layer detection pipeline on an email.

    Args:
        sender: Email sender address or domain.
        subject: Email subject line.
        body: Email body text.
        model: Optional model override for testing.

    Returns:
        PredictionResult with label, confidence, and explanations.

    Raises:
        ValueError: If body is empty after preprocessing.
    """
    ...
```

### JavaScript (Chrome Extension)

- **Style**: ES6+ with `const` and `let` (no `var`)
- **Naming**: `camelCase` for functions and variables
- **Async**: Use `async/await` for API calls
- **DOM**: Use `DomParser` utilities for Gmail DOM interaction — avoid raw `querySelector` in content scripts
- **No external libraries**: The extension uses vanilla JavaScript. Do not introduce jQuery, React, or other frameworks.

### Tests

- **Framework**: Python `unittest`
- **Isolation**: Every test must be independent — use `TemporaryDirectory` for file I/O, `unittest.mock.patch` for dependencies
- **Determinism**: No random seeds, no time-dependent assertions, no shared mutable state
- **Speed**: The full suite of 205 tests runs in ~4 seconds. New tests should not significantly increase this.
- **Coverage**: Every new production code path needs a test. Bug fixes need regression tests.

---

## Testing Requirements

### Before Opening a PR

```bash
# Run all tests
python -m unittest discover -s tests -v
python -m unittest discover -s backend/tests -v

# Expected output:
# Ran 205 tests ... OK
# Ran 20 tests ... OK
```

### Writing Tests

- **Unit tests** go in `tests/unit/` — test individual functions in isolation
- **Integration tests** go in `tests/integration/` — test API endpoints with `TestClient`
- **Test structure**: One test file per production module (`test_registry.py` for `app/ml/registry.py`)
- **Mocking**: Patch at the module level (`@mock.patch("app.core.detector.DOMAIN_CATALOG")`), not the call site

### Test Design Principles

1. **Deterministic**: Fixed inputs → fixed outputs. No randomness.
2. **Isolated**: Doesn't depend on test order or shared state.
3. **Fast**: Individual tests should complete in milliseconds.
4. **Self-checking**: Every assertion is explicit (`assertEqual`, `assertRaises`, not just "prints something and hope it looks right").

---

## Conventional Commits

This project uses [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | Use for |
|---|---|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `test:` | Adding or updating tests |
| `refactor:` | Code change that neither fixes a bug nor adds a feature |
| `perf:` | Performance improvement |
| `chore:` | Maintenance tasks (deps, config, CI) |

### Examples

```
feat: add scheduled retraining with cron expression
fix: handle empty body edge case in feature extraction
docs: add ensemble training guide with GPU requirements
test: add regression test for empty subject line
refactor: extract domain normalization to shared utility
```

---

## Questions?

If you're unsure about anything, open an issue or start a discussion. The maintainer reviews all contributions and is happy to provide guidance.
