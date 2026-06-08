from __future__ import annotations

import re

NLTK_RESOURCES = (
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
)

DEFAULT_SPAM_THRESHOLD = 0.55
DEFAULT_SUBJECT_WEIGHT = 1

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}")
PHONE_PATTERN = re.compile(r"\b\d[\d\-\(\)\s]{6,}\d\b")
MONEY_PATTERN = re.compile(r"[\$£€]\s*\d+[\d,\.]*|\d+[\d,\.]*\s*[\$£€]")
DOMAIN_PATTERN = re.compile(r"^(?:[a-z0-9-]+\.)+[a-z]{2,}$")
MIXED_TOKEN_PATTERN = re.compile(r"\b(?=\w*[a-z])(?=\w*\d)\w+\b", re.IGNORECASE)

URGENCY_KEYWORDS = {
    "urgent", "immediately", "asap", "suspended", "expire", "expired", "deadline", "warning",
}

ACCOUNT_KEYWORDS = {
    "account", "password", "login", "signin", "security", "verify",
    "verification", "identity", "otp", "bank",
}

CALL_TO_ACTION_KEYWORDS = {
    "click", "claim", "confirm", "reset", "verify", "open", "download", "visit", "login",
}

PROMOTIONAL_KEYWORDS = {
    "offer", "discount", "sale", "coupon", "deal", "shop", "weekend", "save", "percent", "shipping",
}

CONVERSATIONAL_KEYWORDS = {
    "lunch", "coffee", "dinner", "meeting", "office", "today", "tomorrow", "plans", "near", "still",
}

BUSINESS_KEYWORDS = {
    "review", "report", "slides", "project", "team", "agenda", "meeting", "office", "update",
    "client", "schedule",
}

PHISHING_PHRASES = [
    "you have won", "you've been selected", "claim your prize", "claim now", "winner",
    "won a lottery", "lottery prize", "free money", "million dollars", "million pound",
    "bitcoin", "cryptocurrency", "wire transfer", "western union", "moneygram",
    "click here to verify", "verify your account immediately", "account suspended",
    "account has been suspended", "confirm your identity", "password will expire",
    "urgent action required", "dear lucky winner",
]

META_FEATURE_NAMES = [
    "url_count", "caps_ratio", "exclamation_count", "question_count", "money_count",
    "phone_count", "word_count", "avg_word_length", "digit_ratio", "spam_phrase_hits",
    "urgency_hits", "account_hits", "call_to_action_hits", "symbol_ratio", "percent_hits",
    "mixed_token_hits",
]

META_FEATURE_LABELS = {
    "url_count": "contains links",
    "caps_ratio": "uses uppercase emphasis",
    "exclamation_count": "uses repeated exclamation marks",
    "question_count": "uses multiple question marks",
    "money_count": "mentions money values",
    "phone_count": "contains a phone number",
    "word_count": "message length",
    "avg_word_length": "long token pattern",
    "digit_ratio": "contains many digits",
    "spam_phrase_hits": "matches phishing phrases",
    "urgency_hits": "contains urgency language",
    "account_hits": "contains account-security language",
    "call_to_action_hits": "contains calls to action",
    "symbol_ratio": "contains many symbols",
    "percent_hits": "contains discount-style percentages",
    "mixed_token_hits": "contains mixed letter-number tokens",
}
