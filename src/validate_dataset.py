"""TaskTune Dataset Validator.

This script checks that a dataset file in JSONL format is healthy,
correctly formatted, and safe to use for machine learning.

Checks performed:
1. File existence and non-emptiness.
2. Valid JSON on every line.
3. Presence of required keys: 'text' and 'label'.
4. Non-empty text field (must not be whitespace only).
5. Label matches one of the expected classes:
   - billing
   - technical_support
   - account_access
6. Detection of duplicate text examples.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# The three customer support classes we support in TaskTune
ALLOWED_LABELS: Set[str] = {"billing", "technical_support", "account_access"}


def validate_dataset(
    file_path: Path | str,
    allowed_labels: Optional[Set[str]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Validate a JSONL dataset file.

    Args:
        file_path: Path to the .jsonl file to check.
        allowed_labels: Set of valid label names. Defaults to ALLOWED_LABELS.

    Returns:
        A tuple of (is_valid, report_dict) where is_valid is a boolean and
        report_dict contains total_count, label_counts, errors, and warnings.
    """
    if allowed_labels is None:
        allowed_labels = ALLOWED_LABELS

    path = Path(file_path)
    errors: List[str] = []
    warnings: List[str] = []
    label_counts: Counter = Counter()
    seen_texts: Dict[str, int] = {}  # Normalized text -> line number
    total_examples = 0

    # 1. Check if the file exists
    if not path.exists():
        errors.append(f"File not found: '{path}'")
        return False, {
            "total_examples": 0,
            "label_counts": {},
            "errors": errors,
            "warnings": warnings,
        }

    # 2. Check if the file is readable and non-empty
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except (OSError, UnicodeError) as exc:
        errors.append(f"Could not read file '{path}': {exc}")
        return False, {
            "total_examples": 0,
            "label_counts": {},
            "errors": errors,
            "warnings": warnings,
        }

    if not lines:
        errors.append(f"The file '{path}' is empty.")
        return False, {
            "total_examples": 0,
            "label_counts": {},
            "errors": errors,
            "warnings": warnings,
        }

    # 3. Process line by line
    for line_num, line in enumerate(lines, start=1):
        stripped = line.strip()
        # Skip completely blank lines if any, but record a warning
        if not stripped:
            warnings.append(f"Line {line_num}: Empty blank line encountered.")
            continue

        # Check for valid JSON
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            errors.append(f"Line {line_num}: Malformed JSON syntax ({exc.msg}).")
            continue

        if not isinstance(record, dict):
            errors.append(f"Line {line_num}: Expected a JSON object (dictionary), got {type(record).__name__}.")
            continue

        # Check required fields
        if "text" not in record:
            errors.append(f"Line {line_num}: Missing required field 'text'.")
        if "label" not in record:
            errors.append(f"Line {line_num}: Missing required field 'label'.")

        if "text" not in record or "label" not in record:
            continue

        text = record["text"]
        label = record["label"]

        # Validate 'text' field
        if not isinstance(text, str):
            errors.append(f"Line {line_num}: 'text' must be a string, got {type(text).__name__}.")
        elif not text.strip():
            errors.append(f"Line {line_num}: 'text' field is empty or contains only whitespace.")

        # Validate 'label' field
        if not isinstance(label, str):
            errors.append(f"Line {line_num}: 'label' must be a string, got {type(label).__name__}.")
        elif not label.strip():
            errors.append(f"Line {line_num}: 'label' field is empty or contains only whitespace.")
        elif label not in allowed_labels:
            errors.append(
                f"Line {line_num}: Invalid label '{label}'. "
                f"Expected one of: {sorted(list(allowed_labels))}."
            )
        else:
            label_counts[label] += 1

        # Check for duplicates (case-insensitive, normalized whitespace)
        if isinstance(text, str) and text.strip():
            norm_text = " ".join(text.strip().lower().split())
            if norm_text in seen_texts:
                first_seen = seen_texts[norm_text]
                errors.append(
                    f"Line {line_num}: Duplicate text detected (identical to line {first_seen})."
                )
            else:
                seen_texts[norm_text] = line_num

        total_examples += 1

    is_valid = len(errors) == 0
    report = {
        "total_examples": total_examples,
        "label_counts": dict(label_counts),
        "errors": errors,
        "warnings": warnings,
    }
    return is_valid, report


def print_report(file_path: Path | str, is_valid: bool, report: Dict[str, Any]) -> None:
    """Print a clean, beginner-friendly summary to the terminal."""
    print("=" * 60)
    print(f"TaskTune Dataset Validation Report")
    print(f"File: {file_path}")
    print("=" * 60)

    if report["warnings"]:
        print(f"\n[WARNING] ({len(report['warnings'])} warnings):")
        for warning in report["warnings"]:
            print(f"   - {warning}")

    if not is_valid:
        print(f"\n[FAIL] Validation FAILED with {len(report['errors'])} error(s):")
        for error in report["errors"]:
            print(f"   - {error}")
        print("\nPlease fix the above errors before proceeding.")
        print("=" * 60)
        return

    print(f"\n[PASS] Validation PASSED!")
    print(f"   Total valid examples: {report['total_examples']}")
    print("\n[INFO] Label Distribution:")
    for label, count in sorted(report["label_counts"].items()):
        percentage = (count / report["total_examples"]) * 100 if report["total_examples"] else 0
        print(f"   - {label:<20}: {count:>3} ({percentage:>5.1f}%)")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a TaskTune customer-support dataset file in JSONL format."
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default="data/sample_dataset.jsonl",
        help="Path to the .jsonl dataset file (default: data/sample_dataset.jsonl)",
    )
    args = parser.parse_args()

    is_valid, report = validate_dataset(args.file)
    print_report(args.file, is_valid, report)

    return 0 if is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
