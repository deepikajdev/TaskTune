"""Profile dataset health, class distribution, text statistics, and anomalies.

Dataset profiling provides a comprehensive inspection of a dataset before training.
It reuses the validation rules from `validate_dataset.py` to examine:
1. Total lines, valid records, empty lines, and invalid records.
2. Class distribution (counts and percentage breakdown).
3. Duplicates (exact text duplicates and whitespace-normalized duplicates).
4. Basic text statistics (character lengths and word counts: min, max, mean, median).
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Optional, Set

if __package__:
    from .validate_dataset import ALLOWED_LABELS
else:
    from validate_dataset import ALLOWED_LABELS


def _compute_stats(numbers: list[int | float]) -> dict[str, float]:
    """Calculate min, max, mean, and median for a list of numbers."""
    if not numbers:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
    return {
        "min": float(min(numbers)),
        "max": float(max(numbers)),
        "mean": round(float(statistics.mean(numbers)), 2),
        "median": round(float(statistics.median(numbers)), 2),
    }


def profile_dataset(
    file_path: Path | str,
    allowed_labels: Optional[Set[str]] = None,
) -> dict[str, Any]:
    """Inspect and profile a JSONL dataset file.

    Args:
        file_path: Path to the JSONL file.
        allowed_labels: Optional set of allowed class labels (defaults to ALLOWED_LABELS).

    Returns:
        A dictionary containing dataset metadata, record breakdown, class distribution,
        duplicate details, invalid records, and text statistics.
    """
    if allowed_labels is None:
        allowed_labels = ALLOWED_LABELS

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: '{path}'")

    try:
        content = path.read_text(encoding="utf-8")
        lines = content.splitlines()
    except (OSError, UnicodeError) as exc:
        raise OSError(f"Could not read dataset file '{path}': {exc}") from exc

    total_lines = len(lines)
    blank_lines: list[int] = []
    invalid_records: list[dict[str, Any]] = []
    valid_records: list[dict[str, Any]] = []

    seen_exact_texts: dict[str, int] = {}
    seen_normalized_texts: dict[str, int] = {}
    duplicates: list[dict[str, Any]] = []

    label_counts: Counter = Counter()
    all_char_lengths: list[int] = []
    all_word_counts: list[int] = []
    class_char_lengths: dict[str, list[int]] = {label: [] for label in sorted(allowed_labels)}
    class_word_counts: dict[str, list[int]] = {label: [] for label in sorted(allowed_labels)}
    vocabulary: set[str] = set()

    for line_number, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            blank_lines.append(line_number)
            continue

        # JSON parsing check
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            invalid_records.append(
                {
                    "line_number": line_number,
                    "reason": f"Malformed JSON: {exc.msg}",
                    "raw_line": stripped,
                }
            )
            continue

        if not isinstance(record, dict):
            invalid_records.append(
                {
                    "line_number": line_number,
                    "reason": f"Expected JSON object, got {type(record).__name__}",
                    "raw_line": stripped,
                }
            )
            continue

        # Check required fields
        missing_fields = [f for f in ("text", "label") if f not in record]
        if missing_fields:
            invalid_records.append(
                {
                    "line_number": line_number,
                    "reason": f"Missing required field(s): {', '.join(missing_fields)}",
                    "raw_line": stripped,
                }
            )
            continue

        text = record["text"]
        label = record["label"]

        # Validate types and non-emptiness
        issues: list[str] = []
        if not isinstance(text, str):
            issues.append(f"'text' must be a string, got {type(text).__name__}")
        elif not text.strip():
            issues.append("'text' is empty or only whitespace")

        if not isinstance(label, str):
            issues.append(f"'label' must be a string, got {type(label).__name__}")
        elif not label.strip():
            issues.append("'label' is empty or only whitespace")
        elif label not in allowed_labels:
            issues.append(f"Invalid label '{label}'. Allowed: {sorted(allowed_labels)}")

        if issues:
            invalid_records.append(
                {
                    "line_number": line_number,
                    "reason": "; ".join(issues),
                    "raw_line": stripped,
                }
            )
            continue

        # Duplicate checking
        exact_text = text.strip()
        norm_text = " ".join(exact_text.lower().split())

        is_duplicate = False
        duplicate_reason = ""
        duplicate_of = -1

        if exact_text in seen_exact_texts:
            is_duplicate = True
            duplicate_reason = "exact duplicate text"
            duplicate_of = seen_exact_texts[exact_text]
        elif norm_text in seen_normalized_texts:
            is_duplicate = True
            duplicate_reason = "normalized duplicate text (case/whitespace variance)"
            duplicate_of = seen_normalized_texts[norm_text]

        if is_duplicate:
            duplicates.append(
                {
                    "line_number": line_number,
                    "duplicate_of_line": duplicate_of,
                    "type": duplicate_reason,
                    "text": exact_text,
                    "label": label,
                }
            )
        else:
            seen_exact_texts[exact_text] = line_number
            seen_normalized_texts[norm_text] = line_number

        # Text statistics
        char_len = len(exact_text)
        words = norm_text.split()
        word_count = len(words)

        all_char_lengths.append(char_len)
        all_word_counts.append(word_count)
        if label in class_char_lengths:
            class_char_lengths[label].append(char_len)
            class_word_counts[label].append(word_count)
        else:
            class_char_lengths[label] = [char_len]
            class_word_counts[label] = [word_count]

        vocabulary.update(words)
        label_counts[label] += 1
        valid_records.append(
            {
                "line_number": line_number,
                "text": exact_text,
                "label": label,
                "char_length": char_len,
                "word_count": word_count,
            }
        )

    # Calculate class distribution percentages
    total_valid = len(valid_records)
    class_distribution = {}
    for label, count in sorted(label_counts.items()):
        percentage = round((count / total_valid) * 100, 2) if total_valid > 0 else 0.0
        class_distribution[label] = {
            "count": count,
            "percentage": percentage,
        }

    # Per-class text stats
    per_class_stats = {}
    for label in sorted(label_counts.keys()):
        per_class_stats[label] = {
            "character_length": _compute_stats(class_char_lengths.get(label, [])),
            "word_count": _compute_stats(class_word_counts.get(label, [])),
        }

    return {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": str(path),
            "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "total_lines": total_lines,
        },
        "overview": {
            "total_lines": total_lines,
            "valid_records": total_valid,
            "invalid_records": len(invalid_records),
            "blank_lines": len(blank_lines),
            "duplicate_records": len(duplicates),
            "unique_vocabulary_tokens": len(vocabulary),
        },
        "class_distribution": class_distribution,
        "overall_text_statistics": {
            "character_length": _compute_stats(all_char_lengths),
            "word_count": _compute_stats(all_word_counts),
        },
        "per_class_text_statistics": per_class_stats,
        "duplicates": duplicates,
        "invalid_records": invalid_records,
        "blank_lines": blank_lines,
    }


def export_summary(profile: dict[str, Any]) -> dict[str, Any]:
    """Extract a compact, reproducible summary from a full dataset profile.

    Returns a flat dictionary containing:
    - record_count: total valid records
    - class_counts: per-label counts
    - duplicate_count: number of duplicate records detected
    - text_length_stats: overall character-length and word-count statistics
    - dataset_sha256: hash of the source dataset file
    """
    overview = profile["overview"]
    return {
        "record_count": overview["valid_records"],
        "class_counts": {
            label: info["count"]
            for label, info in profile["class_distribution"].items()
        },
        "duplicate_count": overview["duplicate_records"],
        "text_length_stats": profile["overall_text_statistics"],
        "dataset_sha256": profile["metadata"]["dataset_sha256"],
    }


def save_profile(profile: dict[str, Any], output_path: Path | str) -> None:
    """Save the dataset profiling dictionary as formatted JSON."""
    path = Path(output_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(profile, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise OSError(f"Could not write profile file '{path}': {exc}") from exc


def print_profile(profile: dict[str, Any], output_path: Path | str | None = None) -> None:
    """Print an easy-to-read summary of dataset profiling."""
    meta = profile["metadata"]
    overview = profile["overview"]
    dist = profile["class_distribution"]
    overall_stats = profile["overall_text_statistics"]
    duplicates = profile["duplicates"]
    invalids = profile["invalid_records"]

    print("=" * 70)
    print("TaskTune Dataset Profiling Report")
    print(f"Dataset : {meta['dataset']}")
    print(f"SHA-256 : {meta['dataset_sha256'][:16]}...")
    print("=" * 70)

    print("\nDataset Overview:")
    print(f"  - Total Lines       : {overview['total_lines']}")
    print(f"  - Valid Records     : {overview['valid_records']}")
    print(f"  - Invalid Records   : {overview['invalid_records']}")
    print(f"  - Blank Lines       : {overview['blank_lines']}")
    print(f"  - Duplicate Records : {overview['duplicate_records']}")
    print(f"  - Vocabulary Size   : {overview['unique_vocabulary_tokens']} unique tokens")

    print("\nClass Distribution:")
    for label, info in dist.items():
        print(f"  - {label:<20}: {info['count']:>4} ({info['percentage']:>5.2f}%)")

    print("\nText Length Statistics (Overall):")
    char_s = overall_stats["character_length"]
    word_s = overall_stats["word_count"]
    print(
        f"  - Character Length  : min={char_s['min']:.0f}, max={char_s['max']:.0f}, "
        f"mean={char_s['mean']:.1f}, median={char_s['median']:.1f}"
    )
    print(
        f"  - Word Count        : min={word_s['min']:.0f}, max={word_s['max']:.0f}, "
        f"mean={word_s['mean']:.1f}, median={word_s['median']:.1f}"
    )

    if duplicates:
        print(f"\n[!] Detected {len(duplicates)} Duplicate(s):")
        for dup in duplicates:
            print(
                f"  - Line {dup['line_number']} is {dup['type']} "
                f"of line {dup['duplicate_of_line']}: \"{dup['text'][:50]}...\""
            )

    if invalids:
        print(f"\n[!] Detected {len(invalids)} Invalid Record(s):")
        for inv in invalids:
            print(f"  - Line {inv['line_number']}: {inv['reason']}")

    if output_path:
        print(f"\nSaved detailed JSON dataset profile to: {output_path}")
    print("=" * 70)


def main() -> int:
    """CLI entry point for dataset profiling."""
    parser = argparse.ArgumentParser(
        description="Profile a TaskTune dataset for class distribution, duplicates, and stats."
    )
    parser.add_argument(
        "--file",
        type=str,
        default="data/sample_dataset.jsonl",
        help="Path to JSONL dataset (default: data/sample_dataset.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/dataset_profile.json",
        help="Path to save JSON profile (default: results/dataset_profile.json).",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Optional path to save a compact JSON summary (record count, class counts, duplicates, text stats).",
    )
    args = parser.parse_args()

    try:
        profile = profile_dataset(args.file)
    except (FileNotFoundError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    save_profile(profile, args.output)
    if args.summary:
        summary = export_summary(profile)
        save_profile(summary, args.summary)
        print(f"Saved compact JSON summary to: {args.summary}")
    print_profile(profile, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
