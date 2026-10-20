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

    def test_record_missing_both_text_and_label(self) -> None:
        """Verify that a record missing both required fields reports two errors."""
        content = '{"id": 1, "category": "billing"}\n'
        file_path = self._write_jsonl("missing_both.jsonl", content)
        is_valid, report = validate_dataset(file_path)

        self.assertFalse(is_valid)
        self.assertTrue(any("Missing required field 'text'" in err for err in report["errors"]))
        self.assertTrue(any("Missing required field 'label'" in err for err in report["errors"]))

    def test_extra_fields_still_pass(self) -> None:
        """Verify that records with extra unexpected fields are accepted."""
        content = '{"text": "Refund please", "label": "billing", "priority": "high", "id": 99}\n'
        file_path = self._write_jsonl("extra_fields.jsonl", content)
        is_valid, report = validate_dataset(file_path)

        self.assertTrue(is_valid)
        self.assertEqual(report["total_examples"], 1)
        self.assertEqual(len(report["errors"]), 0)

    def test_whitespace_variant_duplicates(self) -> None:
        """Verify duplicates are caught when text differs only by internal whitespace."""
        content = (
            '{"text": "I need help with billing", "label": "billing"}\n'
            '{"text": "I  need  help  with  billing", "label": "billing"}\n'
        )
        file_path = self._write_jsonl("ws_duplicates.jsonl", content)
        is_valid, report = validate_dataset(file_path)

        self.assertFalse(is_valid)
        self.assertTrue(any("Duplicate text detected" in err for err in report["errors"]))

    def test_multiple_malformed_json_formats(self) -> None:
        """Verify various kinds of malformed JSON are all detected."""
        content = (
            '{"text": "truncated\n'                            # truncated JSON
            '{text: "no quotes on key", label: "billing"}\n'   # unquoted keys
            '{"text": "trailing comma", "label": "billing",}\n'  # trailing comma
        )
        file_path = self._write_jsonl("multi_malformed.jsonl", content)
        is_valid, report = validate_dataset(file_path)

        self.assertFalse(is_valid)
        malformed_errors = [e for e in report["errors"] if "Malformed JSON syntax" in e]
        self.assertEqual(len(malformed_errors), 3)

    def test_unicode_text_passes(self) -> None:
        """Verify that records with unicode and special characters pass validation."""
        content = '{"text": "Mon paiement a échoué — j\\u0027ai besoin d\\u0027aide 🙏", "label": "billing"}\n'
        file_path = self._write_jsonl("unicode.jsonl", content)
        is_valid, report = validate_dataset(file_path)

        self.assertTrue(is_valid)
        self.assertEqual(report["total_examples"], 1)
        self.assertEqual(len(report["errors"]), 0)

    def test_mixed_valid_and_invalid_counts(self) -> None:
        """Verify correct counting when valid and invalid records are interleaved."""
        content = (
            '{"text": "Valid billing query", "label": "billing"}\n'
            '{"text": "", "label": "billing"}\n'                    # empty text
            '{"text": "Valid tech query", "label": "technical_support"}\n'
            '{"label": "account_access"}\n'                         # missing text
            '{"text": "Valid access query", "label": "account_access"}\n'
        )
        file_path = self._write_jsonl("mixed.jsonl", content)
        is_valid, report = validate_dataset(file_path)

        self.assertFalse(is_valid)
        self.assertEqual(report["total_examples"], 4)
        self.assertEqual(len(report["errors"]), 2)
        self.assertEqual(report["label_counts"]["billing"], 2)
        self.assertEqual(report["label_counts"]["technical_support"], 1)
        self.assertEqual(report["label_counts"]["account_access"], 1)

    def test_non_string_text_types(self) -> None:
        """Verify that boolean and list text values are rejected."""
        content = (
            '{"text": true, "label": "billing"}\n'
            '{"text": ["help", "me"], "label": "technical_support"}\n'
        )
        file_path = self._write_jsonl("non_string_text.jsonl", content)
        is_valid, report = validate_dataset(file_path)

        self.assertFalse(is_valid)
        text_type_errors = [e for e in report["errors"] if "'text' must be a string" in e]
        self.assertEqual(len(text_type_errors), 2)

    def test_only_blank_lines(self) -> None:
        """Verify that a file containing only blank lines is considered empty."""
        content = "\n\n   \n\n"
        file_path = self._write_jsonl("only_blanks.jsonl", content)
        is_valid, report = validate_dataset(file_path)

        self.assertTrue(is_valid)
        self.assertEqual(report["total_examples"], 0)
        self.assertEqual(len(report["warnings"]), 4)


if __name__ == "__main__":
    unittest.main()
