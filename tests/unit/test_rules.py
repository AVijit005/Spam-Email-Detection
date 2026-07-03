from __future__ import annotations

import unittest

from app.core.rules import (
    RuleAssessment,
    BenignAssessment,
    assess_benign_email,
    assess_rule_based_spam,
    is_trusted_service_domain,
)


class TestAssessRuleBasedSpam(unittest.TestCase):
    def test_multiple_spam_phrases_triggers_spam(self):
        result = assess_rule_based_spam(
            "Urgent: account suspended",
            "Click here to verify your identity and wire transfer payment now!!!"
        )
        self.assertTrue(result.is_spam)
        self.assertGreater(result.confidence, 0.85)
        self.assertIn("multiple high-risk", result.reason.lower())

    def test_single_phrase_with_indicator_signals_triggers_spam(self):
        result = assess_rule_based_spam(
            "URGENT: Click here",
            "verify your account immediately!!!"
        )
        self.assertTrue(result.is_spam)

    def test_single_phrase_without_signals_is_not_spam(self):
        result = assess_rule_based_spam("", "you have won something small")
        self.assertFalse(result.is_spam)

    def test_no_phrases_no_signals_is_not_spam(self):
        result = assess_rule_based_spam("Hello", "How are you today?")
        self.assertFalse(result.is_spam)
        self.assertEqual(result.confidence, 0.0)

    def test_returns_signals_in_result(self):
        result = assess_rule_based_spam(
            "URGENT WARNING",
            "https://bad.com click here to verify your account immediately"
        )
        self.assertGreater(len(result.signals), 0)

    def test_confidence_capped_at_0_99(self):
        result = assess_rule_based_spam(
            "URGENT WARNING!!! CLAIM NOW",
            "you have won a lottery prize million dollars bitcoin "
            "wire transfer click here to verify verify your account immediately "
            "account suspended password will expire urgent action required"
        )
        if result.is_spam:
            self.assertLessEqual(result.confidence, 0.99)


class TestAssessBenignEmail(unittest.TestCase):
    def test_conversation_detected_as_benign(self):
        result = assess_benign_email(
            "Lunch today?",
            "Are we still meeting at 1pm near the office?"
        )
        self.assertTrue(result.is_benign)
        self.assertEqual(result.rule_layer, "benign_context")

    def test_low_risk_promo_detected_as_benign(self):
        result = assess_benign_email(
            "Limited time offer",
            "50% off all shoes this weekend only."
        )
        self.assertTrue(result.is_benign)
        self.assertEqual(result.rule_layer, "benign_promo")

    def test_promo_with_link_falls_through_to_ml(self):
        result = assess_benign_email(
            "Special offer",
            "Click https://shop.example.com for 50% off this weekend"
        )
        self.assertEqual(result.rule_layer, "ml")

    def test_promo_with_excessive_caps_falls_through(self):
        result = assess_benign_email(
            "LIMITED TIME OFFER",
            "SHOP NOW FOR 50% OFF EVERYTHING!!!"
        )
        self.assertFalse(result.is_benign)
        self.assertEqual(result.rule_layer, "ml")

    def test_work_conversation_with_context(self):
        result = assess_benign_email(
            "Meeting update",
            "Can we meet at the office today to review the slides for the client project?"
        )
        self.assertTrue(result.is_benign)
        self.assertEqual(result.rule_layer, "benign_context")
        self.assertIn("routine work context", result.signals)


class TestIsTrustedServiceDomain(unittest.TestCase):
    def test_exact_match(self):
        self.assertTrue(is_trusted_service_domain("google.com", {"google.com", "amazon.in"}))

    def test_subdomain_match(self):
        self.assertTrue(is_trusted_service_domain("mail.google.com", {"google.com"}))

    def test_no_match(self):
        self.assertFalse(is_trusted_service_domain("phishing.com", {"google.com"}))

    def test_empty_catalog(self):
        self.assertFalse(is_trusted_service_domain("google.com", set()))
        self.assertFalse(is_trusted_service_domain("google.com", None))

    def test_partial_domain_not_matched(self):
        self.assertFalse(is_trusted_service_domain("fakgoogle.com", {"google.com"}))
