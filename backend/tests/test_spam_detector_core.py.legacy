from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from spam_detector_core import assess_benign_email, assess_rule_based_spam, load_domain_catalog, load_user_whitelist, normalize_domain


class SpamDetectorCoreTests(unittest.TestCase):
    def test_normalize_domain_handles_email_and_url(self):
        self.assertEqual(normalize_domain("Alice <alerts@example.com>"), "example.com")
        self.assertEqual(normalize_domain("https://www.google.com/mail/u/0/"), "google.com")

    def test_loaders_keep_catalog_and_user_whitelist_separate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "trusted_domains.csv").write_text("example.com\nGoogle.com\n", encoding="utf-8")
            (temp_path / "whitelist.csv").write_text(
                "email,domain\nboss@company.com,company.com\n",
                encoding="utf-8",
            )

            catalog = load_domain_catalog(temp_path / "trusted_domains.csv")
            whitelist = load_user_whitelist(temp_path / "whitelist.csv")

        self.assertIn("example.com", catalog)
        self.assertIn("google.com", catalog)
        self.assertNotIn("company.com", catalog)
        self.assertIn("company.com", whitelist)

    def test_rule_based_spam_requires_multiple_signals(self):
        spam_assessment = assess_rule_based_spam(
            "URGENT: account suspended",
            "Click here to verify your identity and wire transfer payment now!!!",
        )
        marketing_assessment = assess_rule_based_spam(
            "Limited time offer",
            "50% off all shoes this weekend only.",
        )

        self.assertTrue(spam_assessment.is_spam)
        self.assertFalse(marketing_assessment.is_spam)

    def test_benign_assessment_handles_conversation_and_low_risk_promo(self):
        conversation = assess_benign_email(
            "Lunch today?",
            "Are we still meeting at 1pm near the office?",
        )
        promo = assess_benign_email(
            "Limited time offer",
            "50% off all shoes this weekend only.",
        )

        self.assertTrue(conversation.is_benign)
        self.assertEqual(conversation.rule_layer, "benign_context")
        self.assertTrue(promo.is_benign)
        self.assertEqual(promo.rule_layer, "benign_promo")


if __name__ == "__main__":
    unittest.main()
