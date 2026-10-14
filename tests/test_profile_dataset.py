"""Tests for dataset profiling."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.profile_dataset import main, profile_dataset, save_profile


class TestProfileDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_path = Path(self.temp_dir.name) / "sample.jsonl"
        lines = [
            '{"text": "First payment issue", "label": "billing"}',
            '{"text": "Second payment issue with more words", "label": "billing"}',
            '{"text": "App crash bug on login", "label": "technical_support"}',
            '{"text": "Forgot password and cannot sign in", "label": "account_access"}',
        ]
        self.dataset_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_profile_clean_dataset(self) -> None:
        profile = profile_dataset(self.dataset_path)

        self.assertIn("metadata", profile)
        self.assertIn("overview", profile)
        self.assertIn("class_distribution", profile)
        self.assertIn("overall_text_statistics", profile)
        self.assertIn("per_class_text_statistics", profile)

        overview = profile["overview"]
        self.assertEqual(overview["total_lines"], 4)
        self.assertEqual(overview["valid_records"], 4)
        self.assertEqual(overview["invalid_records"], 0)
        self.assertEqual(overview["blank_lines"], 0)
        self.assertEqual(overview["duplicate_records"], 0)
        self.assertGreater(overview["unique_vocabulary_tokens"], 0)

        dist = profile["class_distribution"]
        self.assertEqual(dist["billing"]["count"], 2)
        self.assertEqual(dist["billing"]["percentage"], 50.0)
        self.assertEqual(dist["technical_support"]["count"], 1)
        self.assertEqual(dist["technical_support"]["percentage"], 25.0)
        self.assertEqual(dist["account_access"]["count"], 1)
        self.assertEqual(dist["account_access"]["percentage"], 25.0)

        char_stats = profile["overall_text_statistics"]["character_length"]
        self.assertGreater(char_stats["min"], 0)
        self.assertGreaterEqual(char_stats["max"], char_stats["min"])
        self.assertGreaterEqual(char_stats["mean"], char_stats["min"])

    def test_detects_duplicates(self) -> None:
        dup_file = Path(self.temp_dir.name) / "dup.jsonl"
        lines = [
            '{"text": "Hello world", "label": "billing"}',
            '{"text": "Hello world", "label": "billing"}',  # exact dup
            '{"text": "  hello   world  ", "label": "billing"}',  # normalized dup
            '{"text": "Different message", "label": "billing"}',
        ]
        dup_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        profile = profile_dataset(dup_file)
        self.assertEqual(profile["overview"]["duplicate_records"], 2)
        self.assertEqual(len(profile["duplicates"]), 2)
        self.assertEqual(profile["duplicates"][0]["line_number"], 2)
        self.assertEqual(profile["duplicates"][0]["duplicate_of_line"], 1)
        self.assertEqual(profile["duplicates"][1]["line_number"], 3)
        self.assertEqual(profile["duplicates"][1]["duplicate_of_line"], 1)

    def test_detects_invalid_and_blank_records(self) -> None:
        invalid_file = Path(self.temp_dir.name) / "invalid.jsonl"
        lines = [
            "",  # blank line
            '{"text": "valid item", "label": "billing"}',
            "not a json string",  # malformed JSON
            '{"only_text": "missing label"}',  # missing field
            '{"text": "", "label": "billing"}',  # empty text
            '{"text": "ok text", "label": "unknown_label"}',  # invalid label
        ]
        invalid_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        profile = profile_dataset(invalid_file)
        self.assertEqual(profile["overview"]["blank_lines"], 1)
        self.assertEqual(profile["overview"]["invalid_records"], 4)
        self.assertEqual(profile["overview"]["valid_records"], 1)
        self.assertEqual(len(profile["invalid_records"]), 4)

    def test_file_not_found(self) -> None:
        missing = Path(self.temp_dir.name) / "non_existent.jsonl"
        with self.assertRaises(FileNotFoundError):
            profile_dataset(missing)

    def test_save_profile(self) -> None:
        profile = profile_dataset(self.dataset_path)
        out_file = Path(self.temp_dir.name) / "profile.json"
        save_profile(profile, out_file)

        self.assertTrue(out_file.exists())
        loaded = json.loads(out_file.read_text(encoding="utf-8"))
        self.assertEqual(loaded["overview"], profile["overview"])

    def test_cli_main_success(self) -> None:
        out_file = Path(self.temp_dir.name) / "cli_profile.json"
        test_args = [
            "profile_dataset.py",
            "--file",
            str(self.dataset_path),
            "--output",
            str(out_file),
        ]
        with patch.object(sys, "argv", test_args):
            exit_code = main()
            self.assertEqual(exit_code, 0)
            self.assertTrue(out_file.exists())


if __name__ == "__main__":
    unittest.main()
