from __future__ import annotations

import unittest

from app.utils.pii import redact_email_body, redact_subject


class TestPiiRedaction(unittest.TestCase):
    def test_redact_email_addresses_in_body(self):
        result = redact_email_body("Contact user@example.com or admin@company.org today.")
        self.assertNotIn("user@example.com", result)
        self.assertNotIn("admin@company.org", result)
        self.assertIn("[EMAIL]", result)

    def test_redact_phone_numbers_in_body(self):
        result = redact_email_body("Call 555-123-4567 or (123) 456-7890 for help.")
        self.assertNotIn("555-123-4567", result)
        self.assertNotIn("(123) 456-7890", result)
        self.assertIn("[PHONE]", result)

    def test_redact_ip_addresses_in_body(self):
        result = redact_email_body("Connect to 192.168.1.1 or 10.0.0.42.")
        self.assertNotIn("192.168.1.1", result)
        self.assertNotIn("10.0.0.42", result)
        self.assertIn("[IP]", result)

    def test_redact_ssn_in_body(self):
        result = redact_email_body("SSN: 123-45-6789 was referenced.")
        self.assertNotIn("123-45-6789", result)
        self.assertIn("[SSN]", result)

    def test_redact_credit_card_in_body(self):
        result = redact_email_body("Card: 4111-1111-1111-1111 is on file.")
        self.assertNotIn("4111-1111-1111-1111", result)
        self.assertIn("[CCARD]", result)

    def test_preserves_non_pii_body_text(self):
        body = "Hello, please review the quarterly report at your earliest convenience."
        result = redact_email_body(body)
        self.assertEqual(result, body)

    def test_redact_email_addresses_in_subject(self):
        result = redact_subject("Meeting with alice@example.org")
        self.assertNotIn("alice@example.org", result)
        self.assertIn("[EMAIL]", result)

    def test_redact_phone_in_subject(self):
        result = redact_subject("Urgent: call 800-555-0199")
        self.assertNotIn("800-555-0199", result)
        self.assertIn("[PHONE]", result)

    def test_preserves_non_pii_subject(self):
        subject = "Weekly team sync agenda"
        result = redact_subject(subject)
        self.assertEqual(result, subject)

    def test_redaction_is_idempotent(self):
        body = "Email user@example.com and also bob@site.org"
        first = redact_email_body(body)
        second = redact_email_body(first)
        self.assertEqual(first, second)

    def test_empty_body_returns_empty(self):
        self.assertEqual(redact_email_body(""), "")

    def test_empty_subject_returns_empty(self):
        self.assertEqual(redact_subject(""), "")
