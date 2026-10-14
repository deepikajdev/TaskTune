"""Tests for the fair comparison between classification models."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.compare_models import compare_models, main, save_results


class TestCompareModels(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_path = Path(self.temp_dir.name) / "dataset.jsonl"
        examples = {
            "billing": [
                "invoice refund payment",
                "card charge subscription",
                "payment receipt invoice",
                "billing renewal refund",
            ],
            "technical_support": [
                "software application crash",
                "server timeout error",
                "dashboard loading failure",
                "technical export bug",
            ],
            "account_access": [
                "account password login",
                "reset password access",
                "locked account authentication",
                "login verification code",
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

    def test_shared_split_and_consistent_evaluation(self) -> None:
        results = compare_models(
            self.dataset_path, test_size=0.25, random_state=42
        )
        meta = results["metadata"]
        lr_metrics = results["models"]["logistic_regression"]["metrics"]
        nb_metrics = results["models"]["multinomial_nb"]["metrics"]

        # Exactly 3 test examples and 9 train examples
        self.assertEqual(meta["train_examples"], 9)
        self.assertEqual(meta["test_examples"], 3)
        self.assertEqual(len(meta["train_indices"]), 9)
        self.assertEqual(len(meta["test_indices"]), 3)
        self.assertFalse(set(meta["train_indices"]) & set(meta["test_indices"]))

        # Shared label order
        expected_labels = ["account_access", "billing", "technical_support"]
        self.assertEqual(meta["label_order"], expected_labels)

        for metrics in (lr_metrics, nb_metrics):
            self.assertEqual(list(metrics["per_class"]), expected_labels)
            self.assertEqual(
                sum(item["support"] for item in metrics["per_class"].values()), 3
            )
            self.assertEqual(len(metrics["confusion_matrix"]), 3)
            self.assertTrue(all(len(row) == 3 for row in metrics["confusion_matrix"]))
            self.assertEqual(sum(map(sum, metrics["confusion_matrix"])), 3)
            self.assertTrue(0 <= metrics["accuracy"] <= 1)
            self.assertTrue(0 <= metrics["macro_f1"] <= 1)

    def test_comparison_summary(self) -> None:
        results = compare_models(
            self.dataset_path, test_size=0.25, random_state=42
        )
        summary = results["comparison_summary"]
        self.assertIn("accuracy_diff", summary)
        self.assertIn("macro_f1_diff", summary)
        self.assertIn(summary["higher_accuracy_model"], ["logistic_regression", "multinomial_nb", "tied"])
        self.assertIn(summary["higher_macro_f1_model"], ["logistic_regression", "multinomial_nb", "tied"])

    def test_reproducibility(self) -> None:
        run1 = compare_models(self.dataset_path, test_size=0.25, random_state=42)
        run2 = compare_models(self.dataset_path, test_size=0.25, random_state=42)

        self.assertEqual(run1["metadata"]["train_indices"], run2["metadata"]["train_indices"])
        self.assertEqual(run1["metadata"]["test_indices"], run2["metadata"]["test_indices"])
        self.assertEqual(run1["models"], run2["models"])
        self.assertEqual(run1["comparison_summary"], run2["comparison_summary"])

    def test_save_results(self) -> None:
        results = compare_models(self.dataset_path, test_size=0.25, random_state=42)
        output_file = Path(self.temp_dir.name) / "comparison_out.json"
        save_results(results, output_file)

        self.assertTrue(output_file.exists())
        loaded = json.loads(output_file.read_text(encoding="utf-8"))
        self.assertEqual(loaded["comparison_summary"], results["comparison_summary"])

    def test_invalid_test_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            compare_models(self.dataset_path, test_size=0.0)
        with self.assertRaises(ValueError):
            compare_models(self.dataset_path, test_size=1.0)

    def test_insufficient_samples_raises(self) -> None:
        tiny_path = Path(self.temp_dir.name) / "tiny.jsonl"
        tiny_path.write_text(
            '{"text": "bill", "label": "billing"}\n'
            '{"text": "tech", "label": "technical_support"}\n',
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            compare_models(tiny_path, test_size=0.5)

    def test_cli_main_success(self) -> None:
        output_file = Path(self.temp_dir.name) / "cli_models.json"
        test_args = [
            "compare_models.py",
            "--file",
            str(self.dataset_path),
            "--output",
            str(output_file),
            "--test-size",
            "0.25",
            "--random-state",
            "42",
        ]
        with patch.object(sys, "argv", test_args):
            exit_code = main()
            self.assertEqual(exit_code, 0)
            self.assertTrue(output_file.exists())


if __name__ == "__main__":
    unittest.main()
