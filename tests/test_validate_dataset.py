"""Automated unit tests for TaskTune dataset validator.

These tests use Python's standard `unittest` library and temporary files
to verify validator behavior in isolation without depending on real dataset files.
"""

from pathlib import Path
import tempfile
import unittest

from src.validate_dataset import ALLOWED_LABELS, validate_dataset


class TestDatasetValidator(unittest.TestCase):
    """Test suite covering all validation rules for JSONL datasets."""

    def setUp(self) -> None:
        """Create a temporary directory for test dataset files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        """Clean up the temporary directory after each test."""
        self.temp_dir.cleanup()

    def _write_jsonl(self, filename: str, content: str) -> Path:
        """Helper to write raw text content to a temporary JSONL file."""
        file_path = self.test_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return file_path

    def test_valid_dataset(self) -> None:
        """Verify that a properly formatted JSONL passes with accurate counts."""
        content = (
            '{"text": "Please refund my payment.", "label": "billing"}\n'
            '{"text": "The app is crashing on startup.", "label": "technical_support"}\n'
            '{"text": "I forgot my password.", "label": "account_access"}\n'
        )
        file_path = self._write_jsonl("valid.jsonl", content)
        is_valid, report = validate_dataset(file_path)

        self.assertTrue(is_valid)
        self.assertEqual(len(report["errors"]), 0)
        self.assertEqual(report["total_examples"], 3)
        self.assertEqual(report["label_counts"]["billing"], 1)
        self.assertEqual(report["label_counts"]["technical_support"], 1)
        self.assertEqual(report["label_counts"]["account_access"], 1)

    def test_invalid_label(self) -> None:
        """Verify that unknown labels are rejected."""
        content = '{"text": "Where is my parcel?", "label": "shipping_inquiry"}\n'
        file_path = self._write_jsonl("invalid_label.jsonl", content)
        is_valid, report = validate_dataset(file_path)

        self.assertFalse(is_valid)
        self.assertTrue(any("Invalid label" in err for err in report["errors"]))

    def test_empty_or_whitespace_label(self) -> None:
        """Verify empty and whitespace-only labels are rejected clearly."""
        content = (
            '{"text": "I need help with my invoice.", "label": ""}\n'
            '{"text": "I cannot access my profile.", "label": "   "}\n'
        )
        file_path = self._write_jsonl("empty_label.jsonl", content)
        is_valid, report = validate_dataset(file_path)

        self.assertFalse(is_valid)
        self.assertEqual(
            sum("label' field is empty or contains only whitespace" in error
                for error in report["errors"]),
            2,
        )

    def test_non_string_text_and_label(self) -> None:
        """Verify fields with non-string values are rejected."""
        content = '{"text": 42, "label": null}\n'
        file_path = self._write_jsonl("wrong_field_types.jsonl", content)
        is_valid, report = validate_dataset(file_path)

        self.assertFalse(is_valid)
        self.assertTrue(any("'text' must be a string" in err for err in report["errors"]))
        self.assertTrue(any("'label' must be a string" in err for err in report["errors"]))

    def test_missing_fields(self) -> None:
        """Verify that missing 'text' or 'label' fields trigger errors."""
        content = (
            '{"label": "billing"}\n'
            '{"text": "Only text without label."}\n'
        )
        file_path = self._write_jsonl("missing_fields.jsonl", content)
        is_valid, report = validate_dataset(file_path)

        self.assertFalse(is_valid)
        self.assertTrue(any("Missing required field 'text'" in err for err in report["errors"]))
        self.assertTrue(any("Missing required field 'label'" in err for err in report["errors"]))

    def test_empty_or_whitespace_text(self) -> None:
        """Verify that empty text or whitespace-only text fails."""
        content = (
            '{"text": "", "label": "billing"}\n'
            '{"text": "   ", "label": "technical_support"}\n'
        )
        file_path = self._write_jsonl("empty_text.jsonl", content)
        is_valid, report = validate_dataset(file_path)

        self.assertFalse(is_valid)
        self.assertTrue(any("empty or contains only whitespace" in err for err in report["errors"]))

    def test_malformed_json(self) -> None:
        """Verify that broken JSON syntax is detected and reported with line number."""
        content = (
            '{"text": "Valid line", "label": "billing"}\n'
            '{"text": "Broken line", label: billing}\n'  # Missing quotes around key and value
        )
        file_path = self._write_jsonl("malformed.jsonl", content)
        is_valid, report = validate_dataset(file_path)

        self.assertFalse(is_valid)
        self.assertTrue(any("Malformed JSON syntax" in err for err in report["errors"]))

    def test_duplicate_text_detection(self) -> None:
        """Verify that identical text examples (even with case/whitespace variations) fail."""
        content = (
            '{"text": "Cannot login to my account.", "label": "account_access"}\n'
            '{"text": "CANNOT LOGIN TO MY ACCOUNT.", "label": "account_access"}\n'
        )
        file_path = self._write_jsonl("duplicates.jsonl", content)
        is_valid, report = validate_dataset(file_path)

        self.assertFalse(is_valid)
        self.assertTrue(any("Duplicate text detected" in err for err in report["errors"]))

    def test_missing_file(self) -> None:
        """Verify graceful error when file does not exist."""
        non_existent = self.test_dir / "does_not_exist.jsonl"
        is_valid, report = validate_dataset(non_existent)

        self.assertFalse(is_valid)
        self.assertTrue(any("File not found" in err for err in report["errors"]))

    def test_empty_file(self) -> None:
        """Verify that an empty file is flagged."""
        file_path = self._write_jsonl("empty.jsonl", "")
        is_valid, report = validate_dataset(file_path)

        self.assertFalse(is_valid)
        self.assertTrue(any("is empty" in err for err in report["errors"]))

    def test_json_non_dict_object(self) -> None:
        """Verify that JSON primitives/arrays instead of objects are rejected."""
        content = (
            '["text", "billing"]\n'
            '"just a json string"\n'
            '12345\n'
        )
        file_path = self._write_jsonl("non_dict.jsonl", content)
        is_valid, report = validate_dataset(file_path)

        self.assertFalse(is_valid)
        self.assertEqual(len(report["errors"]), 3)
        self.assertTrue(all("Expected a JSON object (dictionary)" in err for err in report["errors"]))

    def test_blank_lines_generate_warnings(self) -> None:
        """Verify that blank lines produce warnings but allow valid records through."""
        content = (
            '{"text": "Invoice query", "label": "billing"}\n'
            '\n'
            '   \n'
            '{"text": "Login failure", "label": "account_access"}\n'
        )
        file_path = self._write_jsonl("blank_lines.jsonl", content)
        is_valid, report = validate_dataset(file_path)

        self.assertTrue(is_valid)
        self.assertEqual(report["total_examples"], 2)
        self.assertEqual(len(report["warnings"]), 2)
        self.assertTrue(all("Empty blank line" in w for w in report["warnings"]))

    def test_custom_allowed_labels(self) -> None:
        """Verify that validator accepts a custom set of allowed labels."""
        content = '{"text": "Shipment delayed", "label": "logistics"}\n'
        file_path = self._write_jsonl("custom_labels.jsonl", content)

        # Standard allowed labels reject it
        is_valid_default, _ = validate_dataset(file_path)
        self.assertFalse(is_valid_default)

        # Custom allowed labels accept it
        is_valid_custom, report = validate_dataset(file_path, allowed_labels={"logistics", "sales"})
        self.assertTrue(is_valid_custom)
        self.assertEqual(report["label_counts"]["logistics"], 1)


if __name__ == "__main__":
    unittest.main()
