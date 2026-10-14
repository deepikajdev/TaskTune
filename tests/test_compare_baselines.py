"""Tests for the majority-class and Logistic Regression comparison."""

import json
from pathlib import Path
import tempfile
import unittest

from src.compare_baselines import compare_baselines, save_results


class TestBaselineComparison(unittest.TestCase):
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

    def test_models_share_stratified_split_and_label_order(self) -> None:
        results = compare_baselines(
            self.dataset_path, test_size=0.25, random_state=17
        )
        metadata = results["metadata"]
        majority = results["models"]["majority_class"]["metrics"]
        classifier = results["models"]["tfidf_logistic_regression"]["metrics"]

        self.assertEqual(metadata["train_examples"], 9)
        self.assertEqual(metadata["test_examples"], 3)
        self.assertEqual(len(metadata["train_indices"]), 9)
        self.assertEqual(len(metadata["test_indices"]), 3)
        self.assertFalse(
            set(metadata["train_indices"]) & set(metadata["test_indices"])
        )
        self.assertEqual(
            metadata["label_order"],
            ["account_access", "billing", "technical_support"],
        )
        for metrics in (majority, classifier):
            self.assertEqual(
                list(metrics["per_class"]), metadata["label_order"]
            )
            self.assertEqual(
                sum(item["support"] for item in metrics["per_class"].values()), 3
            )
            self.assertEqual(len(metrics["confusion_matrix"]), 3)
            self.assertTrue(
                all(len(row) == 3 for row in metrics["confusion_matrix"])
            )
            self.assertEqual(
                sum(map(sum, metrics["confusion_matrix"])), 3
            )

    def test_split_and_metrics_are_reproducible(self) -> None:
        first = compare_baselines(
            self.dataset_path, test_size=0.25, random_state=29
        )
        second = compare_baselines(
            self.dataset_path, test_size=0.25, random_state=29
        )

        self.assertEqual(first["metadata"]["train_indices"], second["metadata"]["train_indices"])
        self.assertEqual(first["metadata"]["test_indices"], second["metadata"]["test_indices"])
        self.assertEqual(first["models"], second["models"])

    def test_results_are_saved_as_json(self) -> None:
        results = compare_baselines(self.dataset_path, random_state=17)
        output_path = Path(self.temp_dir.name) / "results" / "comparison.json"

        save_results(results, output_path)

        with output_path.open(encoding="utf-8") as results_file:
            saved_results = json.load(results_file)
        self.assertEqual(saved_results, results)
        self.assertEqual(
            len(saved_results["metadata"]["dataset_sha256"]), 64
        )

    def test_invalid_dataset_fails_validation(self) -> None:
        self.dataset_path.write_text(
            '{"text": "message", "label": "unknown"}\n',
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "Dataset validation failed"):
            compare_baselines(self.dataset_path)


if __name__ == "__main__":
    unittest.main()
