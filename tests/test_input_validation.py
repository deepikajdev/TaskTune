"""Tests for the input validation module."""

import unittest

from src.input_validation import (
    MAX_MESSAGE_LENGTH,
    friendly_validation_error,
    validate_message,
)


class TestValidateMessage(unittest.TestCase):
    def test_valid_message_returned_stripped(self) -> None:
        self.assertEqual(validate_message("  hello world  "), "hello world")

    def test_empty_string_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_message("")
        self.assertIn("blank", str(ctx.exception).lower())

    def test_whitespace_only_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            validate_message("   \t\n  ")

    def test_too_long_message_raises_value_error(self) -> None:
        long_msg = "a" * (MAX_MESSAGE_LENGTH + 1)
        with self.assertRaises(ValueError) as ctx:
            validate_message(long_msg)
        self.assertIn(str(MAX_MESSAGE_LENGTH), str(ctx.exception))

    def test_exact_max_length_is_accepted(self) -> None:
        exactly_max = "a" * MAX_MESSAGE_LENGTH
        result = validate_message(exactly_max)
        self.assertEqual(len(result), MAX_MESSAGE_LENGTH)

    def test_integer_input_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            validate_message(42)  # type: ignore[arg-type]

    def test_none_input_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            validate_message(None)  # type: ignore[arg-type]

    def test_list_input_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            validate_message(["hello"])  # type: ignore[arg-type]

    def test_normal_customer_message_accepted(self) -> None:
        msg = "I was charged twice for my subscription."
        self.assertEqual(validate_message(msg), msg)

    def test_unicode_message_accepted(self) -> None:
        msg = "El sistema ha fallado con código 500 🔧"
        self.assertEqual(validate_message(msg), msg)


class TestFriendlyValidationError(unittest.TestCase):
    def test_formats_with_emoji_prefix(self) -> None:
        exc = ValueError("Something went wrong")
        result = friendly_validation_error(exc)
        self.assertTrue(result.startswith("⚠️"))
        self.assertIn("Something went wrong", result)

    def test_works_with_type_error(self) -> None:
        exc = TypeError("Not a string")
        result = friendly_validation_error(exc)
        self.assertIn("Not a string", result)


if __name__ == "__main__":
    unittest.main()

