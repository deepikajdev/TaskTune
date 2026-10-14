"""Tests for stratified cross-validation evaluation."""

import json
from pathlib import Path
import tempfile
import unittest

from src.evaluate_cross_validation import (
    evaluate_cross_validation,
    save_results,
)


class TestCrossValidationEvaluation(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_path = Path(self.temp_dir.name) / "dataset.jsonl"
        examples = {
            "billing": [
                "invoice payment refund",
                "subscription card charge",
                "billing receipt renewal",
                "payment declined invoice",
                "unexpected monthly charge",
                "tax on latest statement",
            ],
            "technical_support": [
                "application crashes at startup",
                "server request timeout",
                "dashboard loading error",
                "report export is broken",
                "software update causes a bug",
                "notifications arrive late",
            ],
            "account_access": [
                "forgot my account password",
                "cannot sign in to account",
                "login verification code missing",
                "account locked after attempts",
                "reset password link expired",
                "lost access to authenticator",
            ],
        }
        with self.dataset_path.open("w", encoding="utf-8") as dataset_file:
            for label, messages in examples.items():
                for text in messages:
                    dataset_file.write(
                        json.dumps({"text": text, "label": label}) + "\n"
                    )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_reports_fold_metrics_and_class_counts(self) -> None:
        results = evaluate_cross_validation(
            self.dataset_path, folds=3, random_state=23
        )
        metadata = results["metadata"]

        self.assertEqual(metadata["dataset_examples"], 18)
        self.assertEqual(metadata["class_counts"], {
            "account_access": 6,
            "billing": 6,
            "technical_support": 6,
        })
        self.assertEqual(metadata["folds"], 3)
        self.assertEqual(len(results["fold_results"]), 3)
        for fold in results["fold_results"]:
            self.assertEqual(fold["train_examples"], 12)
            self.assertEqual(fold["test_examples"], 6)
            self.assertEqual(
                set(fold["metrics"]["per_class"]), set(metadata["label_order"])
            )
            self.assertTrue(
                0 <= fold["metrics"]["accuracy"] <= 1
            )
            self.assertTrue(
                0 <= fold["metrics"]["macro_f1"] <= 1
            )

        for metric in ("accuracy", "macro_f1"):
            self.assertGreaterEqual(results["summary"]["mean"][metric], 0)
            self.assertGreaterEqual(
                results["summary"]["population_std"][metric], 0
            )

    def test_metrics_and_fold_assignments_are_deterministic(self) -> None:
        first = evaluate_cross_validation(
            self.dataset_path, folds=3, random_state=31
        )
        second = evaluate_cross_validation(
            self.dataset_path, folds=3, random_state=31
        )

        self.assertEqual(first["fold_results"], second["fold_results"])
        self.assertEqual(first["summary"], second["summary"])

    def test_fails_if_any_class_has_too_few_examples(self) -> None:
        self.dataset_path.write_text(
            '{"text": "billing one", "label": "billing"}\n'
            '{"text": "billing two", "label": "billing"}\n'
            '{"text": "support one", "label": "technical_support"}\n'
            '{"text": "support two", "label": "technical_support"}\n'
            '{"text": "access only", "label": "account_access"}\n',
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "every class needs at least 3"):
            evaluate_cross_validation(self.dataset_path, folds=3)

    def test_writes_valid_json_to_requested_directory(self) -> None:
        results = evaluate_cross_validation(
            self.dataset_path, folds=3, random_state=23
        )
        output_path = Path(self.temp_dir.name) / "results" / "cross_validation.json"

        save_results(results, output_path)

        with output_path.open(encoding="utf-8") as results_file:
            saved_results = json.load(results_file)
        self.assertEqual(saved_results, results)
        self.assertEqual(
            len(saved_results["metadata"]["dataset_sha256"]), 64
        )

    def test_rejects_less_than_two_folds(self) -> None:
        with self.assertRaisesRegex(ValueError, "--folds must be at least 2"):
            evaluate_cross_validation(self.dataset_path, folds=1)


if __name__ == "__main__":
    unittest.main()
