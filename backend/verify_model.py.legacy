from __future__ import annotations

import json
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from spam_detector_core import DEFAULT_SPAM_THRESHOLD, load_domain_catalog, load_user_whitelist, predict_email
from app.ml.registry import load_model, ModelIntegrityError


MODEL_PATH = CURRENT_DIR / "model" / "spam_model.pkl"
VECTORIZER_PATH = CURRENT_DIR / "model" / "vectorizer.pkl"
METADATA_PATH = CURRENT_DIR / "model" / "model_metadata.json"
TRUSTED_DOMAINS_PATH = CURRENT_DIR / "data" / "trusted_domains.csv"
USER_WHITELIST_PATH = CURRENT_DIR / "data" / "whitelist.csv"


TEST_CASES = [
    {
        "sender": "promo@random-mailer.biz",
        "subject": "WINNER!! Claim your £900 prize reward now",
        "body": "Congratulations!!! Click here to verify and claim now.",
        "expected": "Spam",
    },
    {
        "sender": "security@unknown-alerts.net",
        "subject": "URGENT: account suspended",
        "body": "Confirm your identity immediately or your account will be deleted.",
        "expected": "Spam",
    },
    {
        "sender": "shipping@amazon.in",
        "subject": "Your order has shipped",
        "body": "Your Amazon package will arrive tomorrow.",
        "expected": "Not Spam",
    },
    {
        "sender": "boss@company.com",
        "subject": "Weekly review",
        "body": "Please send the revised slides before 4pm.",
        "expected": "whitelisted",
    },
    {
        "sender": "marketing@store.example",
        "subject": "Limited time offer",
        "body": "50% off all shoes this weekend only.",
        "expected": "Not Spam",
    },
    {
        "sender": "friend@example.org",
        "subject": "Lunch today?",
        "body": "Are we still meeting at 1pm near the office?",
        "expected": "Not Spam",
    },
]


def verify() -> int:
    print("=" * 60)
    print("  Spam Detector - Verification")
    print("=" * 60)

    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        print("Model artefacts are missing.")
        print("Run: python backend/model/train_model.py")
        return 1

    try:
        artifact = load_model(MODEL_PATH, VECTORIZER_PATH, METADATA_PATH)
    except ModelIntegrityError as exc:
        print(f"Integrity check failed: {exc}")
        return 1
    if artifact is None:
        print("Failed to load model artefacts.")
        return 1

    metadata = artifact.metadata
    model_version = str(metadata.get("model_name", "unknown"))
    spam_threshold = float(metadata.get("spam_threshold", DEFAULT_SPAM_THRESHOLD))
    whitelist_domains = load_user_whitelist(USER_WHITELIST_PATH)
    trusted_domain_catalog = load_domain_catalog(TRUSTED_DOMAINS_PATH)

    print(f"Loaded model      : {model_version}")
    print(f"Spam threshold    : {spam_threshold}")
    print(f"User whitelist    : {len(whitelist_domains)}")
    print(f"Domain catalog    : {len(trusted_domain_catalog)}")
    print(f"Running test cases: {len(TEST_CASES)}\n")

    passed = 0
    for case in TEST_CASES:
        result = predict_email(
            model=artifact.model,
            vectorizer=artifact.vectorizer,
            sender=case["sender"],
            subject=case["subject"],
            body=case["body"],
            whitelist_domains=whitelist_domains,
            trusted_service_domains=trusted_domain_catalog,
            model_version=model_version,
            spam_threshold=spam_threshold,
        )

        ok = result.label == case["expected"]
        marker = "PASS" if ok else "FAIL"
        print(
            f"[{marker}] expected={case['expected']:<11} got={result.label:<11} "
            f"layer={result.rule_layer:<14} confidence={result.confidence:.2f}"
        )
        print(f"       {case['subject']}")
        if result.signals:
            print(f"       signals: {', '.join(result.signals)}")
        passed += int(ok)

    print(f"\nResult: {passed}/{len(TEST_CASES)} passed ({(passed / len(TEST_CASES)) * 100:.1f}%)")
    return 0 if passed == len(TEST_CASES) else 1


if __name__ == "__main__":
    raise SystemExit(verify())
