from __future__ import annotations

import re

_PII_PATTERNS: list[tuple[str, str]] = [
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '[EMAIL]'),
    (r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]'),
    (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP]'),
    (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),
    (r'\b(?:\d[ -]*?){13,19}\b', '[CCARD]'),
]


def redact_email_body(body: str) -> str:
    result = body
    for pattern, replacement in _PII_PATTERNS:
        result = re.sub(pattern, replacement, result)
    return result


def redact_subject(subject: str) -> str:
    result = subject
    for pattern, replacement in _PII_PATTERNS:
        result = re.sub(pattern, replacement, result)
    return result
