from __future__ import annotations

import unittest


class TestUtilsInit(unittest.TestCase):
    def test_pii_redact_functions_exported(self):
        from app.utils import pii
        self.assertTrue(hasattr(pii, "redact_email_body"))
        self.assertTrue(hasattr(pii, "redact_subject"))

    def test_redact_email_body_callable(self):
        from app.utils.pii import redact_email_body
        result = redact_email_body("test@example.com")
        self.assertNotIn("test@example.com", result)

    def test_redact_subject_callable(self):
        from app.utils.pii import redact_subject
        result = redact_subject("test@example.com")
        self.assertNotIn("test@example.com", result)
