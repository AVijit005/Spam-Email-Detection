from __future__ import annotations

import unittest

import numpy as np

from app.core.features import (
    _count_keyword_hits,
    _indicator_signals,
    _meta_feature_map,
    compose_email_text,
    extract_meta_features,
    matched_spam_phrases,
)
from app.core.constants import PHISHING_PHRASES


class TestComposeEmailText(unittest.TestCase):
    def test_default_subject_weight(self):
        result = compose_email_text("Verify account", "Click here")
        self.assertEqual(result, "Verify account Click here")

    def test_subject_weight_two(self):
        result = compose_email_text("Urgent", "Body", subject_weight=2)
        self.assertEqual(result, "Urgent Urgent Body")

    def test_empty_subject(self):
        result = compose_email_text("", "Body text")
        self.assertEqual(result, "Body text")

    def test_empty_body(self):
        result = compose_email_text("Subject", "")
        self.assertEqual(result, "Subject")

    def test_both_empty(self):
        result = compose_email_text("", "")
        self.assertEqual(result, "")

    def test_subject_weight_zero_falls_back_to_one(self):
        result = compose_email_text("Hi", "Body", subject_weight=0)
        self.assertEqual(result, "Hi Body")


class TestMatchedSpamPhrases(unittest.TestCase):
    def test_detects_known_phrases(self):
        result = matched_spam_phrases("", "you have won a big prize")
        self.assertIn("you have won", result)

    def test_subject_and_body_combined(self):
        result = matched_spam_phrases("claim your prize", "")
        self.assertIn("claim your prize", result)

    def test_no_phrases_returns_empty(self):
        result = matched_spam_phrases("Hello", "How are you today?")
        self.assertEqual(result, [])

    def test_case_insensitive(self):
        result = matched_spam_phrases("", "YOU HAVE WON")
        self.assertIn("you have won", result)


class TestExtractMetaFeatures(unittest.TestCase):
    def test_returns_correct_shape_single_text(self):
        result = extract_meta_features("Hello world")
        self.assertEqual(result.shape, (1, 32))

    def test_returns_correct_shape_batch(self):
        result = extract_meta_features(["Hello world", "Test"])
        self.assertEqual(result.shape, (2, 32))

    def test_url_count(self):
        result = extract_meta_features("Visit https://example.com and http://test.org")
        self.assertEqual(result[0][0], 2)

    def test_exclamation_count(self):
        result = extract_meta_features("Wow!!! Amazing!!!")
        self.assertGreaterEqual(result[0][2], 6)

    def test_word_count(self):
        result = extract_meta_features("one two three four five")
        self.assertEqual(result[0][6], 5)

    def test_urgency_hits(self):
        result = extract_meta_features("urgent warning immediately asap")
        self.assertEqual(result[0][10], 4)

    def test_account_hits(self):
        result = extract_meta_features("verify your account password login security")
        self.assertEqual(result[0][11], 5)

    def test_call_to_action_hits(self):
        result = extract_meta_features("click here to confirm and verify now")
        self.assertEqual(result[0][12], 3)

    def test_accepts_string_list_or_single_string(self):
        batch = extract_meta_features(["Hello", "World test"])
        single = extract_meta_features("Hello")
        self.assertEqual(batch.shape[0], 2)
        self.assertEqual(single.shape[0], 1)

    def test_empty_text_produces_valid_row(self):
        result = extract_meta_features("")
        self.assertEqual(result.shape, (1, 32))
        self.assertFalse(np.isnan(result).any())


class TestKeywordHits(unittest.TestCase):
    def test_counts_matches_only(self):
        hits = _count_keyword_hits("urgent warning test", {"urgent", "warning", "immediately"})
        self.assertEqual(hits, 2)

    def test_case_insensitive(self):
        hits = _count_keyword_hits("URGENT Warning", {"urgent", "warning"})
        self.assertEqual(hits, 2)


class TestIndicatorSignals(unittest.TestCase):
    def test_url_signal(self):
        signals = _indicator_signals("Visit https://example.com now")
        self.assertIn("contains a link", signals)

    def test_money_signal(self):
        signals = _indicator_signals("Price is $19.99 today")
        self.assertIn("mentions money amounts", signals)

    def test_phone_signal(self):
        signals = _indicator_signals("Call 555-123-4567")
        self.assertIn("contains a phone number", signals)

    def test_aggressive_punctuation(self):
        signals = _indicator_signals("Wow!!! Amazing!!! Incredible!!!")
        self.assertIn("uses aggressive punctuation", signals)

    def test_urgency_signal(self):
        signals = _indicator_signals("urgent warning immediately asap")
        self.assertIn("contains urgency language", signals)

    def test_benign_text_no_signals(self):
        signals = _indicator_signals("Hello, how are you today?")
        self.assertEqual(signals, [])


class TestMetaFeatureMap(unittest.TestCase):
    def test_returns_dict_with_all_keys(self):
        fm = _meta_feature_map("Hello world test")
        self.assertIsInstance(fm, dict)
        self.assertIn("url_count", fm)
        self.assertIn("caps_ratio", fm)
        self.assertIn("word_count", fm)
        self.assertEqual(fm["word_count"], 3)
