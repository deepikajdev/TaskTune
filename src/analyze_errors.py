"""Analyze and report classification errors for the TaskTune classifier.

Error analysis helps machine learning practitioners understand where and why
a model makes mistakes. Instead of only looking at high-level accuracy, this
script examines every prediction on the held-out test set, extracts misclassified
examples, and tallies error counts for each category.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

if __package__:
    from .train_baseline import create_classifier, load_dataset
else:
    from train_baseline import create_classifier, load_dataset


def analyze_errors(
    file_path: Path | str,
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train on the training split and analyze errors on the held-out test split.

    Args:
        file_path: Path to the validated .jsonl dataset file.
        test_size: Fraction of the dataset reserved for testing (0 < test_size < 1).
        random_state: Random seed for reproducible dataset splitting.

    Returns:
        A dictionary containing dataset metadata, overall error summary,
        per-class error counts, all test predictions, and misclassified examples.
    """
    if not 0 < test_size < 1:
        raise ValueError("--test-size must be greater than 0 and less than 1.")

    dataset_path = Path(file_path)
    texts, labels = load_dataset(dataset_path)
    label_order = sorted(set(labels))
    label_counts = Counter(labels)

    too_small = [label for label in label_order if label_counts[label] < 2]
    if too_small:
        raise ValueError(
            "A stratified split needs at least two examples for every label. "
            f"Insufficient labels: {', '.join(too_small)}."
        )

    indices = list(range(len(texts)))
    try:
        train_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )
    except ValueError as exc:
        raise ValueError(
            "Could not create a stratified train/test split. Adjust --test-size "
            "so both splits can contain every label. "
            f"Details: {exc}"
        ) from exc

    train_texts = [texts[idx] for idx in train_indices]
    train_labels = [labels[idx] for idx in train_indices]
    test_texts = [texts[idx] for idx in test_indices]
    test_labels = [labels[idx] for idx in test_indices]

    classifier = create_classifier(random_state)
    classifier.fit(train_texts, train_labels)
    predicted_labels = classifier.predict(test_texts).tolist()

    # Collect predictions for all test examples
    test_predictions: list[dict[str, Any]] = []
    misclassified_examples: list[dict[str, Any]] = []

    for test_pos, original_idx in enumerate(test_indices):
        true_label = test_labels[test_pos]
        pred_label = predicted_labels[test_pos]
        text_content = test_texts[test_pos]
        is_error = true_label != pred_label

        record = {
            "index": int(original_idx),
            "text": text_content,
            "true_label": true_label,
            "predicted_label": pred_label,
            "is_error": is_error,
        }
        test_predictions.append(record)

        if is_error:
            misclassified_examples.append(record)

    # Calculate per-class error counts (how many actual examples of each class were misclassified)
    per_class_error_counts = {label: 0 for label in label_order}
    for item in misclassified_examples:
        per_class_error_counts[item["true_label"]] += 1

    total_test = len(test_labels)
    total_errors = len(misclassified_examples)
    accuracy = float(accuracy_score(test_labels, predicted_labels))
    error_rate = float(total_errors / total_test) if total_test > 0 else 0.0

    return {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": str(dataset_path),
            "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "dataset_examples": len(texts),
            "python_version": sys.version.split()[0],
            "scikit_learn_version": sklearn.__version__,
            "test_size": test_size,
            "random_state": random_state,
            "train_examples": len(train_indices),
            "test_examples": total_test,
            "label_order": label_order,
        },
        "summary": {
            "total_test_examples": total_test,
            "total_errors": total_errors,
            "accuracy": accuracy,
            "error_rate": error_rate,
        },
        "per_class_error_counts": per_class_error_counts,
        "misclassified_examples": misclassified_examples,
        "test_predictions": test_predictions,
    }


def save_results(results: dict[str, Any], output_path: Path | str) -> None:
    """Save the error analysis dictionary as formatted JSON."""
    path = Path(output_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise OSError(f"Could not write results file '{path}': {exc}") from exc


def print_report(results: dict[str, Any], output_path: Path | str | None = None) -> None:
    """Print a readable summary of error analysis to the terminal."""
    meta = results["metadata"]
    summary = results["summary"]
    errors = results["per_class_error_counts"]
    misclassified = results["misclassified_examples"]

    print("=" * 65)
    print("TaskTune Classifier Error Analysis Report")
    print(f"Dataset: {meta['dataset']}")
    print(
        f"Split: {meta['train_examples']} training / "
        f"{meta['test_examples']} test examples (stratified)"
    )
    print("=" * 65)

    print(
        f"Test Accuracy : {summary['accuracy']:.4f} "
        f"({summary['total_test_examples'] - summary['total_errors']}/{summary['total_test_examples']} correct)"
    )
    print(
        f"Total Errors  : {summary['total_errors']} "
        f"(error rate: {summary['error_rate'] * 100:.2f}%)"
    )

    print("\nPer-Class Error Counts (actual class misclassified):")
    for label, count in errors.items():
        print(f"  - {label:<20}: {count} error(s)")

    if not misclassified:
        print("\nNo errors found on the test split! Perfect classification.")
    else:
        print(f"\nMisclassified Examples ({len(misclassified)} total):")
        for idx, item in enumerate(misclassified, start=1):
            print(f"\n  [{idx}] Record #{item['index']}")
            print(f"      True Label     : {item['true_label']}")
            print(f"      Predicted Label: {item['predicted_label']}")
            print(f"      Text           : \"{item['text']}\"")

    if output_path:
        print(f"\nSaved detailed JSON results to: {output_path}")
    print("=" * 65)


def main() -> int:
    """CLI entry point for running error analysis."""
    parser = argparse.ArgumentParser(
        description="Run error analysis on the TaskTune classifier."
    )
    parser.add_argument(
        "--file",
        type=str,
        default="data/sample_dataset.jsonl",
        help="Path to the JSONL dataset (default: data/sample_dataset.jsonl).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction of dataset held out for evaluation (default: 0.25).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for reproducible train/test split (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/error_analysis.json",
        help="Path to save JSON error report (default: results/error_analysis.json).",
    )
    args = parser.parse_args()

    try:
        results = analyze_errors(
            file_path=args.file,
            test_size=args.test_size,
            random_state=args.random_state,
        )
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    save_results(results, args.output)
    print_report(results, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
