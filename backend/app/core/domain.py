from __future__ import annotations

import csv
from pathlib import Path

from app.core.constants import DOMAIN_PATTERN, EMAIL_PATTERN


def normalize_domain(value: str | None) -> str:
    if not value:
        return ""
    candidate = value.strip().lower().strip(" <>\"'[]()")
    if not candidate:
        return ""
    email_match = EMAIL_PATTERN.search(candidate)
    if email_match:
        candidate = email_match.group(0).split("@", 1)[1]
    candidate = candidate.split("://", 1)[-1]
    candidate = candidate.split("/", 1)[0]
    candidate = candidate.split("?", 1)[0]
    candidate = candidate.split("#", 1)[0]
    candidate = candidate.split(":", 1)[0]
    candidate = candidate.removeprefix("www.").strip(".")
    return candidate if DOMAIN_PATTERN.fullmatch(candidate) else ""


def extract_sender_domain(sender: str | None) -> str:
    return normalize_domain(sender)


def _read_rows(path: Path) -> tuple[list[list[str]], list[str], bool]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        return [], [], False
    first_row = [cell.strip().lower() for cell in rows[0]]
    has_header = "domain" in first_row or "email" in first_row
    data_rows = rows[1:] if has_header else rows
    return data_rows, first_row, has_header


def load_domain_catalog(*paths: str | Path) -> set[str]:
    domains: set[str] = set()
    for raw_path in paths:
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.exists():
            continue
        data_rows, _, _ = _read_rows(path)
        for row in data_rows:
            for cell in row:
                domain = normalize_domain(cell)
                if domain:
                    domains.add(domain)
    return domains


def load_user_whitelist(*paths: str | Path) -> set[str]:
    whitelist_domains: set[str] = set()
    for raw_path in paths:
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.exists():
            continue
        data_rows, first_row, has_header = _read_rows(path)
        domain_index = first_row.index("domain") if has_header and "domain" in first_row else None
        email_index = first_row.index("email") if has_header and "email" in first_row else None
        for row in data_rows:
            candidates: list[str]
            if domain_index is not None and domain_index < len(row):
                candidates = [row[domain_index]]
            elif email_index is not None and email_index < len(row):
                candidates = [row[email_index]]
            else:
                candidates = row
            for cell in candidates:
                domain = normalize_domain(cell)
                if domain:
                    whitelist_domains.add(domain)
    return whitelist_domains


def load_trusted_domains(*paths: str | Path) -> set[str]:
    return load_domain_catalog(*paths)
