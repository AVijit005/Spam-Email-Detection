from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from model.train_model import load_feedback_dataset


class TrainModelFeedbackTests(unittest.TestCase):
    def test_load_feedback_dataset_collapses_duplicates_and_maps_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_path = Path(temp_dir) / "feedback.jsonl"
            entries = [
                {
                    "stored_at_utc": "2026-04-03T10:00:00+00:00",
                    "subject": "Verify account",
                    "body": "Click here to verify account.",
                    "user_label": "Spam",
                    "verdict": "false_negative",
                },
                {
                    "stored_at_utc": "2026-04-03T10:05:00+00:00",
                    "subject": "Lunch today?",
                    "body": "Are we still meeting at 1pm near the office?",
                    "user_label": "Not Spam",
                    "verdict": "correct",
                },
                {
                    "stored_at_utc": "2026-04-03T10:10:00+00:00",
                    "subject": "Lunch today?",
                    "body": "Are we still meeting at 1pm near the office?",
                    "user_label": "Not Spam",
                    "verdict": "correct",
                },
            ]
            feedback_path.write_text(
                "\n".join(json.dumps(entry) for entry in entries) + "\nnot-json\n",
                encoding="utf-8",
            )

            dataframe, stats = load_feedback_dataset(feedback_path)

        self.assertEqual(len(dataframe), 2)
        self.assertEqual(stats["feedback_rows_used"], 2)
        self.assertEqual(stats["duplicates_collapsed"], 1)
        self.assertEqual(stats["invalid_json_lines"], 1)
        self.assertEqual(stats["label_counts"]["0"], 1)
        self.assertEqual(stats["label_counts"]["1"], 1)
        self.assertIn(3.0, dataframe["sample_weight"].tolist())
        self.assertIn(1.5, dataframe["sample_weight"].tolist())


if __name__ == "__main__":
    unittest.main()
