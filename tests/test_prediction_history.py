"""Unit tests for in-memory prediction history management."""

import unittest

from src.prediction_history import (
    MAX_HISTORY_ITEMS,
    add_history_record,
    clear_history,
    create_history_record,
)


class TestPredictionHistory(unittest.TestCase):
    def test_create_history_record(self) -> None:
        rec = create_history_record(
            message="Test message",
            predicted_label="billing",
            confidence=0.92,
            sequence=1,
            timestamp="2026-11-10 12:00:00 UTC",
        )
        self.assertEqual(rec["sequence"], 1)
        self.assertEqual(rec["message"], "Test message")
        self.assertEqual(rec["predicted_label"], "billing")
        self.assertEqual(rec["confidence"], 0.92)
        self.assertEqual(rec["timestamp"], "2026-11-10 12:00:00 UTC")

    def test_add_history_record_increments_sequence(self) -> None:
        history: list[dict] = []
        add_history_record(history, "msg 1", "billing", 0.8)
        add_history_record(history, "msg 2", "account_access", 0.95)

        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["sequence"], 1)
        self.assertEqual(history[1]["sequence"], 2)
        self.assertEqual(history[1]["predicted_label"], "account_access")

    def test_add_history_record_caps_at_max_items(self) -> None:
        history: list[dict] = []
        for i in range(15):
            add_history_record(history, f"msg {i}", "billing", 0.9, max_items=10)

        self.assertEqual(len(history), 10)
        # Should keep the most recent ones
        self.assertEqual(history[-1]["message"], "msg 14")

    def test_clear_history(self) -> None:
        history: list[dict] = [{"sample": 1}]
        cleared = clear_history()
        self.assertEqual(cleared, [])


if __name__ == "__main__":
    unittest.main()
