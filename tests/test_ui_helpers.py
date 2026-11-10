"""Unit tests for TaskTune UI helper functions."""

import unittest

from src.ui_helpers import (
    SAMPLE_QUERIES,
    get_confidence_rating,
    get_label_emoji,
    truncate_text,
)


class TestUIHelpers(unittest.TestCase):
    def test_sample_queries_exist_and_non_empty(self) -> None:
        self.assertGreaterEqual(len(SAMPLE_QUERIES), 3)
        for label, query in SAMPLE_QUERIES.items():
            self.assertTrue(len(query.strip()) > 10)

    def test_get_label_emoji(self) -> None:
        self.assertEqual(get_label_emoji("billing"), "💳")
        self.assertEqual(get_label_emoji("technical_support"), "🔧")
        self.assertEqual(get_label_emoji("account_access"), "🔑")
        self.assertEqual(get_label_emoji("unknown_category"), "📌")

    def test_truncate_text(self) -> None:
        short_text = "Short text"
        self.assertEqual(truncate_text(short_text, max_chars=50), "Short text")

        long_text = "This is a very long text that exceeds the limit of twenty characters"
        truncated = truncate_text(long_text, max_chars=20)
        self.assertTrue(truncated.endswith("…"))
        self.assertLessEqual(len(truncated), 22)

    def test_get_confidence_rating(self) -> None:
        high_lbl, high_icon = get_confidence_rating(0.95)
        self.assertEqual(high_lbl, "High Confidence")
        self.assertEqual(high_icon, "🟢")

        med_lbl, med_icon = get_confidence_rating(0.65)
        self.assertEqual(med_lbl, "Moderate Confidence")
        self.assertEqual(med_icon, "🟡")

        low_lbl, low_icon = get_confidence_rating(0.35)
        self.assertEqual(low_lbl, "Low Confidence")
        self.assertEqual(low_icon, "🔴")


if __name__ == "__main__":
    unittest.main()

