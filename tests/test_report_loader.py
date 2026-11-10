"""Unit tests for saved evaluation report loader."""

from pathlib import Path
import tempfile
import unittest

from src.report_loader import (
    get_baseline_metrics,
    get_cross_validation_metrics,
    load_json_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_REPORT = PROJECT_ROOT / "results" / "baseline_comparison.json"
CV_REPORT = PROJECT_ROOT / "results" / "cross_validation.json"


class TestReportLoader(unittest.TestCase):
    def test_load_existing_baseline_report(self) -> None:
        metrics = get_baseline_metrics(BASELINE_REPORT)
        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertIn("accuracy", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertIn("per_class", metrics)
        self.assertGreater(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["train_examples"], 56)
        self.assertEqual(metrics["test_examples"], 19)

    def test_load_existing_cv_report(self) -> None:
        cv_metrics = get_cross_validation_metrics(CV_REPORT)
        self.assertIsNotNone(cv_metrics)
        assert cv_metrics is not None
        self.assertEqual(cv_metrics["num_folds"], 5)
        self.assertGreater(cv_metrics["mean_accuracy"], 0.0)
        self.assertIn("account_access", cv_metrics["per_class"])

    def test_missing_report_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "does_not_exist.json"
            self.assertIsNone(load_json_report(missing))
            self.assertIsNone(get_baseline_metrics(missing))
            self.assertIsNone(get_cross_validation_metrics(missing))

    def test_corrupt_report_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            corrupt = Path(temp_dir) / "corrupt.json"
            corrupt.write_text("{not valid json", encoding="utf-8")
            self.assertIsNone(load_json_report(corrupt))
            self.assertIsNone(get_baseline_metrics(corrupt))


if __name__ == "__main__":
    unittest.main()
