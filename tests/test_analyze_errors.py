"""Tests for classifier error analysis."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.analyze_errors import analyze_errors, main, save_results


class TestAnalyzeErrors(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_path = Path(self.temp_dir.name) / "dataset.jsonl"
        examples = {
            "billing": [
                "invoice refund payment",
                "card charge subscription",
                "payment receipt invoice",
                "billing renewal refund",
                "unexpected monthly fee",
            ],
            "technical_support": [
                "software application crash",
                "server timeout error",
                "dashboard loading failure",
                "technical export bug",
                "app crashing with 500 error",
            ],
            "account_access": [
                "account password login",
                "reset password access",
                "locked account authentication",
                "login verification code",
                "two factor auth lockout",
            ],
        }
        with self.dataset_path.open("w", encoding="utf-8") as dataset_file:
            for label, texts in examples.items():
                for text in texts:
                    dataset_file.write(
                        json.dumps({"text": text, "label": label}) + "\n"
                    )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_returns_expected_structure_and_error_counts(self) -> None:
        results = analyze_errors(
            self.dataset_path, test_size=0.33, random_state=42
        )

        self.assertIn("metadata", results)
        self.assertIn("summary", results)
        self.assertIn("per_class_error_counts", results)
        self.assertIn("misclassified_examples", results)
        self.assertIn("test_predictions", results)

        summary = results["summary"]
        self.assertEqual(summary["total_test_examples"], len(results["test_predictions"]))
        self.assertEqual(summary["total_errors"], len(results["misclassified_examples"]))
        self.assertAlmostEqual(
            summary["error_rate"],
            summary["total_errors"] / summary["total_test_examples"],
        )

        # Check per-class error counts match misclassified counts by true label
        per_class = results["per_class_error_counts"]
        for label in results["metadata"]["label_order"]:
            expected_count = sum(
                1 for item in results["misclassified_examples"] if item["true_label"] == label
            )
            self.assertEqual(per_class[label], expected_count)

    def test_prediction_fields(self) -> None:
        results = analyze_errors(
            self.dataset_path, test_size=0.33, random_state=42
        )
        for pred in results["test_predictions"]:
            self.assertIn("index", pred)
            self.assertIn("text", pred)
            self.assertIn("true_label", pred)
            self.assertIn("predicted_label", pred)
            self.assertIn("is_error", pred)
            self.assertEqual(pred["is_error"], pred["true_label"] != pred["predicted_label"])

    def test_reproducibility(self) -> None:
        run1 = analyze_errors(self.dataset_path, test_size=0.33, random_state=42)
        run2 = analyze_errors(self.dataset_path, test_size=0.33, random_state=42)

        self.assertEqual(run1["test_predictions"], run2["test_predictions"])
        self.assertEqual(run1["summary"], run2["summary"])
        self.assertEqual(run1["per_class_error_counts"], run2["per_class_error_counts"])

    def test_save_results(self) -> None:
        results = analyze_errors(self.dataset_path, test_size=0.33, random_state=42)
        output_path = Path(self.temp_dir.name) / "error_analysis.json"
        save_results(results, output_path)

        self.assertTrue(output_path.exists())
        loaded = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["summary"], results["summary"])

    def test_invalid_test_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            analyze_errors(self.dataset_path, test_size=0.0)
        with self.assertRaises(ValueError):
            analyze_errors(self.dataset_path, test_size=1.0)

    def test_insufficient_samples_raises(self) -> None:
        tiny_path = Path(self.temp_dir.name) / "tiny.jsonl"
        tiny_path.write_text(
            '{"text": "bill", "label": "billing"}\n'
            '{"text": "tech", "label": "technical_support"}\n',
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            analyze_errors(tiny_path, test_size=0.5)

    def test_cli_main_success(self) -> None:
        output_file = Path(self.temp_dir.name) / "cli_errors.json"
        test_args = [
            "analyze_errors.py",
            "--file",
            str(self.dataset_path),
            "--output",
            str(output_file),
            "--test-size",
            "0.33",
            "--random-state",
            "42",
        ]
        with patch.object(sys, "argv", test_args):
            exit_code = main()
            self.assertEqual(exit_code, 0)
            self.assertTrue(output_file.exists())


if __name__ == "__main__":
    unittest.main()
