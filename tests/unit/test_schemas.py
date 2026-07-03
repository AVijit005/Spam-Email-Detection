from __future__ import annotations

import unittest

from pydantic import ValidationError

from app.schemas.email import EmailRequest, BatchPredictionRequest
from app.schemas.feedback import FeedbackRequest


class TestEmailRequest(unittest.TestCase):
    def test_valid_request_with_all_fields(self):
        req = EmailRequest(sender="test@example.com", subject="Hello", body="World")
        self.assertEqual(req.sender, "test@example.com")
        self.assertEqual(req.subject, "Hello")

    def test_defaults_empty_strings(self):
        req = EmailRequest()
        self.assertEqual(req.sender, "")
        self.assertEqual(req.subject, "")
        self.assertEqual(req.body, "")

    def test_subject_max_length(self):
        EmailRequest(subject="A" * 998)

    def test_subject_exceeds_max_length(self):
        with self.assertRaises(ValidationError):
            EmailRequest(subject="A" * 999)

    def test_body_max_length(self):
        EmailRequest(body="B" * 100_000)

    def test_body_exceeds_max_length(self):
        with self.assertRaises(ValidationError):
            EmailRequest(body="B" * 100_001)

    def test_sender_max_length(self):
        EmailRequest(sender="A" * 320)

    def test_sender_exceeds_max_length(self):
        with self.assertRaises(ValidationError):
            EmailRequest(sender="A" * 321)


class TestBatchPredictionRequest(unittest.TestCase):
    def test_valid_batch(self):
        req = BatchPredictionRequest(emails=[
            {"sender": "a@b.com", "subject": "S1", "body": "B1"},
            {"sender": "c@d.com", "subject": "S2", "body": "B2"},
        ])
        self.assertEqual(len(req.emails), 2)

    def test_empty_emails_default(self):
        req = BatchPredictionRequest()
        self.assertEqual(req.emails, [])

    def test_max_fifty_emails_allowed(self):
        emails = [{"sender": "a@b.com", "subject": "s", "body": "b"} for _ in range(50)]
        req = BatchPredictionRequest(emails=emails)
        self.assertEqual(len(req.emails), 50)

    def test_exceeds_fifty_emails_rejected(self):
        emails = [{"sender": "a@b.com", "subject": "s", "body": "b"} for _ in range(51)]
        with self.assertRaises(ValidationError):
            BatchPredictionRequest(emails=emails)


class TestFeedbackRequest(unittest.TestCase):
    def test_valid_feedback(self):
        req = FeedbackRequest(
            prediction_id="pred-001",
            user_label="Spam",
            predicted_label="Spam",
        )
        self.assertEqual(req.user_label, "Spam")

    def test_notes_max_length(self):
        FeedbackRequest(
            prediction_id="p", user_label="Spam", predicted_label="Spam",
            notes="A" * 1000,
        )

    def test_notes_exceeds_max_length(self):
        with self.assertRaises(ValidationError):
            FeedbackRequest(
                prediction_id="p", user_label="Spam", predicted_label="Spam",
                notes="A" * 1001,
            )

    def test_default_source_is_extension_popup(self):
        req = FeedbackRequest(
            prediction_id="p", user_label="Spam", predicted_label="Spam",
        )
        self.assertEqual(req.source, "extension_popup")

    def test_missing_prediction_id_rejected(self):
        with self.assertRaises(ValidationError):
            FeedbackRequest(user_label="Spam", predicted_label="Spam")

    def test_missing_user_label_rejected(self):
        with self.assertRaises(ValidationError):
            FeedbackRequest(prediction_id="p", predicted_label="Spam")
