"""Tests for model persistence, serialization round-trip, and trust assumptions."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from sklearn.pipeline import Pipeline

from src.model_persistence import (
    load_model,
    predict_with_model,
    save_model,
    train_and_save_model,
)
from src.train_baseline import create_classifier, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "sample_dataset.jsonl"
SUPPORTED_LABELS = {"billing", "technical_support", "account_access"}


class TestModelPersistence(unittest.TestCase):
    def test_round_trip_save_and_load(self) -> None:
        """Verify model can be saved and loaded with identical predictions and scores."""
        texts, labels = load_dataset(DATASET_PATH)
        original_model = create_classifier(random_state=42)
        original_model.fit(texts, labels)

        test_message = "I need a refund for my subscription invoice."
        orig_res = predict_with_model(original_model, test_message, include_probabilities=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_file = Path(temp_dir) / "test_model.joblib"
            saved_path = save_model(original_model, model_file)
            self.assertEqual(saved_path.resolve(), model_file.resolve())
            self.assertTrue(model_file.exists())

            loaded_model = load_model(model_file)
            self.assertIsInstance(loaded_model, Pipeline)

            loaded_res = predict_with_model(loaded_model, test_message, include_probabilities=True)
            self.assertEqual(loaded_res["predicted_label"], orig_res["predicted_label"])
            self.assertAlmostEqual(loaded_res["confidence"], orig_res["confidence"], places=6)
            for label in orig_res["probabilities"]:
                self.assertAlmostEqual(
                    loaded_res["probabilities"][label],
                    orig_res["probabilities"][label],
                    places=6,
                )

    def test_train_and_save_model_helper(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_file = Path(temp_dir) / "subfolder" / "pipeline.joblib"
            saved_path = train_and_save_model(DATASET_PATH, model_file)
            self.assertTrue(saved_path.is_file())

            loaded_model = load_model(saved_path)
            label = predict_with_model(loaded_model, "Login authentication failed")
            self.assertIn(label, SUPPORTED_LABELS)

    def test_load_nonexistent_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent = Path(temp_dir) / "missing_model.joblib"
            with self.assertRaises(FileNotFoundError):
                load_model(nonexistent)

    def test_load_corrupted_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            corrupt_file = Path(temp_dir) / "corrupt.joblib"
            corrupt_file.write_text("not a valid serialized joblib object", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_model(corrupt_file)

    def test_save_invalid_object_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_file = Path(temp_dir) / "invalid.joblib"
            with self.assertRaises(ValueError):
                save_model("not a model", invalid_file)

    def test_predict_message_cli_with_saved_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_file = Path(temp_dir) / "model.joblib"
            train_and_save_model(DATASET_PATH, model_file)

            env = os.environ.copy()
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "src" / "predict_message.py"),
                    "I cannot reset my password",
                    "--model",
                    str(model_file),
                    "--scores",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                check=False,
                text=True,
                env=env,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Predicted category: account_access", result.stdout)
            self.assertIn("Confidence score:", result.stdout)
            self.assertIn("Class probabilities:", result.stdout)


if __name__ == "__main__":
    unittest.main()

