from __future__ import annotations

import unittest

from app.core.text import preprocess_text


class TestPreprocessText(unittest.TestCase):
    def test_tokenizes_basic_text(self):
        result = preprocess_text("Running quickly stores")
        self.assertIn("running", result)
        self.assertIn("quickly", result)
        self.assertIn("stores", result)

    def test_replaces_url_with_urltoken(self):
        result = preprocess_text("Visit https://example.com/path now")
        self.assertIn("urltoken", result)
        self.assertNotIn("https", result)
        self.assertNotIn("example.com", result)

    def test_replaces_email_with_emailtoken(self):
        result = preprocess_text("Email user@example.com for details")
        self.assertIn("emailtoken", result)
        self.assertNotIn("user@example.com", result)

    def test_replaces_phone_with_phonetoken(self):
        result = preprocess_text("Call 555-123-4567 today")
        self.assertIn("phonetoken", result)
        self.assertNotIn("555-123-4567", result)

    def test_replaces_money_with_moneytoken(self):
        result = preprocess_text("Price is $19.99 or €50")
        self.assertIn("moneytoken", result)
        self.assertNotIn("$19", result)

    def test_strips_special_characters(self):
        result = preprocess_text("hello!!! world??? (test) [now]")
        self.assertNotIn("!", result)
        self.assertNotIn("?", result)
        self.assertNotIn("(", result)
        self.assertNotIn("[", result)

    def test_returns_empty_for_empty_input(self):
        self.assertEqual(preprocess_text(""), "")

    def test_returns_empty_for_non_string(self):
        self.assertEqual(preprocess_text(None), "")
        self.assertEqual(preprocess_text(123), "")

    def test_keeps_high_value_stopwords(self):
        result = preprocess_text("Free cash prize click now urgent limited offer")
        self.assertIn("free", result)
        self.assertIn("cash", result)
        self.assertIn("prize", result)
        self.assertIn("click", result)
        self.assertIn("now", result)
        self.assertIn("urgent", result)
        self.assertIn("limited", result)
        self.assertIn("offer", result)

    def test_removes_low_value_stopwords(self):
        result = preprocess_text("The meeting is about the project")
        self.assertNotIn("the", result)
        self.assertNotIn("is", result)
        self.assertNotIn("about", result)

    def test_filters_single_character_tokens(self):
        result = preprocess_text("A simple test with a few words")
        self.assertNotIn("a", result.split())

    def test_lowercases_all_text(self):
        result = preprocess_text("URGENT MEETING NOW")
        for token in result.split():
            self.assertEqual(token, token.lower())
