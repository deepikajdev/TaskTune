"""Tests for the customer-message prediction command-line interface."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICT_SCRIPT = PROJECT_ROOT / "src" / "predict_message.py"
DATASET_PATH = PROJECT_ROOT / "data" / "sample_dataset.jsonl"
SUPPORTED_LABELS = {"billing", "technical_support", "account_access"}


class TestPredictMessageCLI(unittest.TestCase):
    def _run_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        return subprocess.run(
            [sys.executable, str(PREDICT_SCRIPT), *arguments],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=False,
            text=True,
            env=environment,
        )

    def test_cli_prints_a_supported_label(self) -> None:
        result = self._run_cli(
            "I was charged twice for my subscription.",
            "--file",
            str(DATASET_PATH),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        prefix = "Predicted category: "
        self.assertTrue(result.stdout.startswith(prefix))
        self.assertIn(result.stdout[len(prefix):].strip(), SUPPORTED_LABELS)

    def test_empty_message_is_reported(self) -> None:
        result = self._run_cli("", "--file", str(DATASET_PATH))

        self.assertEqual(result.returncode, 1)
        self.assertIn("Message cannot be empty", result.stderr)

    def test_whitespace_message_is_reported(self) -> None:
        result = self._run_cli("   ", "--file", str(DATASET_PATH))

        self.assertEqual(result.returncode, 1)
        self.assertIn("Message cannot be empty", result.stderr)

    def test_missing_dataset_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing_path = Path(directory) / "missing.jsonl"
            result = self._run_cli(
                "Please help me sign in.", "--file", str(missing_path)
            )

        self.assertEqual(result.returncode, 1)
        self.assertIn("File not found", result.stderr)

    def test_invalid_dataset_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            invalid_dataset = Path(directory) / "invalid.jsonl"
            invalid_dataset.write_text(
                '{"text": "An example message", "label": "unknown"}\n',
                encoding="utf-8",
            )
            result = self._run_cli(
                "Please help me sign in.", "--file", str(invalid_dataset)
            )

        self.assertEqual(result.returncode, 1)
        self.assertIn("Dataset validation failed", result.stderr)
        self.assertIn("Invalid label", result.stderr)


if __name__ == "__main__":
    unittest.main()
