from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.core.domain import (
    normalize_domain,
    extract_sender_domain,
    load_domain_catalog,
    load_user_whitelist,
    load_trusted_domains,
)


class TestNormalizeDomainEdgeCases(unittest.TestCase):
    def test_email_with_name_and_angle_brackets(self):
        self.assertEqual(normalize_domain("Alice <admin@test.com>"), "test.com")

    def test_url_with_path_query_fragment(self):
        self.assertEqual(normalize_domain("https://google.com/path?q=1#frag"), "google.com")

    def test_ip_address_rejected(self):
        self.assertEqual(normalize_domain("192.168.1.1"), "")

    def test_empty_string(self):
        self.assertEqual(normalize_domain(""), "")

    def test_none_input(self):
        self.assertEqual(normalize_domain(None), "")

    def test_url_with_port(self):
        self.assertEqual(normalize_domain("https://example.com:8080/path"), "example.com")

    def test_bare_domain(self):
        self.assertEqual(normalize_domain("example.com"), "example.com")

    def test_strips_brackets_and_quotes(self):
        self.assertEqual(normalize_domain("<'\"[test.com]\"'>"), "test.com")

    def test_strips_www_prefix(self):
        self.assertEqual(normalize_domain("www.example.com"), "example.com")

    def test_invalid_domain_rejected(self):
        self.assertEqual(normalize_domain("not a domain"), "")


class TestExtractSenderDomain(unittest.TestCase):
    def test_extracts_from_email_address(self):
        self.assertEqual(extract_sender_domain("user@company.org"), "company.org")

    def test_extracts_from_name_email_format(self):
        self.assertEqual(extract_sender_domain("John Doe <john@company.org>"), "company.org")

    def test_empty_sender(self):
        self.assertEqual(extract_sender_domain(""), "")

    def test_none_sender(self):
        self.assertEqual(extract_sender_domain(None), "")


class TestLoadDomainCatalog(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_loads_single_file(self):
        csv_path = self.tmp_path / "domains.csv"
        csv_path.write_text("example.com\ngoogle.com\n", encoding="utf-8")
        catalog = load_domain_catalog(csv_path)
        self.assertEqual(catalog, {"example.com", "google.com"})

    def test_loads_multiple_files(self):
        path1 = self.tmp_path / "a.csv"
        path2 = self.tmp_path / "b.csv"
        path1.write_text("site1.com\n", encoding="utf-8")
        path2.write_text("site2.com\n", encoding="utf-8")
        catalog = load_domain_catalog(path1, path2)
        self.assertEqual(catalog, {"site1.com", "site2.com"})

    def test_skips_missing_file(self):
        missing = self.tmp_path / "no_such_file.csv"
        result = load_domain_catalog(missing)
        self.assertEqual(result, set())

    def test_skips_empty_path(self):
        result = load_domain_catalog("", None)
        self.assertEqual(result, set())

    def test_deduplicates_across_files(self):
        path1 = self.tmp_path / "a.csv"
        path2 = self.tmp_path / "b.csv"
        path1.write_text("dup.com\n", encoding="utf-8")
        path2.write_text("dup.com\nother.com\n", encoding="utf-8")
        catalog = load_domain_catalog(path1, path2)
        self.assertEqual(catalog, {"dup.com", "other.com"})

    def test_handles_header_row(self):
        csv_path = self.tmp_path / "with_header.csv"
        csv_path.write_text("domain\nexample.com\n", encoding="utf-8")
        catalog = load_domain_catalog(csv_path)
        self.assertEqual(catalog, {"example.com"})

    def test_transforms_email_to_domain(self):
        csv_path = self.tmp_path / "emails.csv"
        csv_path.write_text("user@example.com\n", encoding="utf-8")
        catalog = load_domain_catalog(csv_path)
        self.assertEqual(catalog, {"example.com"})


class TestLoadUserWhitelist(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_loads_with_email_header(self):
        csv_path = self.tmp_path / "whitelist.csv"
        csv_path.write_text("email,domain\nboss@company.com,company.com\n", encoding="utf-8")
        whitelist = load_user_whitelist(csv_path)
        self.assertIn("company.com", whitelist)

    def test_loads_without_header(self):
        csv_path = self.tmp_path / "raw.csv"
        csv_path.write_text("company.com\npartner.org\n", encoding="utf-8")
        whitelist = load_user_whitelist(csv_path)
        self.assertEqual(whitelist, {"company.com", "partner.org"})

    def test_domain_column_preferred(self):
        csv_path = self.tmp_path / "both.csv"
        csv_path.write_text("email,domain\nboss@company.com,mycompany.com\n", encoding="utf-8")
        whitelist = load_user_whitelist(csv_path)
        self.assertIn("mycompany.com", whitelist)
        self.assertNotIn("company.com", whitelist)

    def test_falls_back_to_email_column_if_no_domain(self):
        csv_path = self.tmp_path / "email_only.csv"
        csv_path.write_text("email\nboss@company.com\n", encoding="utf-8")
        whitelist = load_user_whitelist(csv_path)
        self.assertIn("company.com", whitelist)


class TestLoadTrustedDomains(unittest.TestCase):
    def test_alias_calls_load_domain_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trusted.csv"
            path.write_text("amazon.in\n", encoding="utf-8")
            result = load_trusted_domains(path)
        self.assertEqual(result, {"amazon.in"})
