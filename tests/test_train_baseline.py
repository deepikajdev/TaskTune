"""Automated tests for TaskTune's classical text-classification baseline."""

import json
from pathlib import Path
import tempfile
import unittest

from src.train_baseline import load_dataset, run_experiment


class TestBaselineExperiment(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_path = Path(self.temp_dir.name) / "dataset.jsonl"
        examples = {
            "billing": [
                "invoice payment charge refund",
                "billing card subscription receipt",
                "payment refund invoice statement",
                "subscription charge billing card",
            ],
            "technical_support": [
                "application crash software error",
                "technical bug server timeout",
                "software update loading error",
                "application server bug crash",
            ],
            "account_access": [
                "account password login access",
                "login password reset account",
                "account locked authentication login",
                "password access verification account",
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

    def test_experiment_reports_held_out_metrics_and_confusion_matrix(self) -> None:
        results = run_experiment(self.dataset_path, test_size=0.25, random_state=17)

        self.assertEqual(results["train_examples"], 9)
        self.assertEqual(results["test_examples"], 3)
        self.assertTrue(
            all(
                metrics["support"] == 1
                for metrics in results["class_metrics"].values()
            )
        )
        self.assertEqual(set(results["labels"]), {
            "billing",
            "technical_support",
            "account_access",
        })
        self.assertGreaterEqual(results["accuracy"], 0.0)
        self.assertLessEqual(results["accuracy"], 1.0)
        self.assertEqual(set(results["class_metrics"]), set(results["labels"]))
        self.assertEqual(
            sum(item["support"] for item in results["class_metrics"].values()), 3
        )
        self.assertEqual(len(results["confusion_matrix"]), 3)
        self.assertTrue(
            all(len(row) == 3 for row in results["confusion_matrix"])
        )
        self.assertEqual(
            results,
            run_experiment(self.dataset_path, test_size=0.25, random_state=17),
        )

    def test_invalid_jsonl_label_fails_before_training(self) -> None:
        self.dataset_path.write_text(
            '{"text": "not a supported category", "label": "unknown"}\n',
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "Dataset validation failed"):
            run_experiment(self.dataset_path)

    def test_invalid_test_size_is_reported(self) -> None:
        with self.assertRaisesRegex(ValueError, "--test-size"):
            run_experiment(self.dataset_path, test_size=1.0)

    def test_each_label_needs_two_examples_for_stratification(self) -> None:
        self.dataset_path.write_text(
            '{"text": "only one example", "label": "billing"}\n',
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "at least two examples"):
            run_experiment(self.dataset_path)

    def test_load_dataset_reports_missing_file(self) -> None:
        with self.assertRaisesRegex(ValueError, "Dataset validation failed"):
            load_dataset(self.dataset_path.parent / "missing.jsonl")


if __name__ == "__main__":
    unittest.main()
