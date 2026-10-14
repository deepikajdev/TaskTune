"""Train and evaluate the TaskTune TF-IDF and LogisticRegression baseline."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

if __package__:
    from .validate_dataset import validate_dataset
else:
    from validate_dataset import validate_dataset


def create_classifier(random_state: int = 42) -> Pipeline:
    """Create the project's standard TF-IDF and Logistic Regression pipeline."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )


def load_dataset(file_path: Path | str) -> tuple[list[str], list[str]]:
    """Validate a JSONL file and return its texts and labels."""
    path = Path(file_path)
    is_valid, report = validate_dataset(path)
    if not is_valid:
        details = "\n".join(f"- {error}" for error in report["errors"])
        raise ValueError(f"Dataset validation failed:\n{details}")

    texts: list[str] = []
    labels: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as dataset_file:
            for line_number, line in enumerate(dataset_file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Dataset changed or became invalid at line {line_number}: {exc.msg}."
                    ) from exc
                texts.append(record["text"])
                labels.append(record["label"])
    except OSError as exc:
        raise OSError(f"Could not read dataset file '{path}': {exc}") from exc

    if not texts:
        raise ValueError(f"Dataset file '{path}' contains no usable records.")
    return texts, labels


def run_experiment(
    file_path: Path | str,
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit on a stratified training split and evaluate on held-out examples."""
    if not 0 < test_size < 1:
        raise ValueError("--test-size must be greater than 0 and less than 1.")

    texts, labels = load_dataset(file_path)
    label_names = sorted(set(labels))
    label_counts = {label: labels.count(label) for label in label_names}
    too_small = [label for label, count in label_counts.items() if count < 2]
    if too_small:
        raise ValueError(
            "A stratified split needs at least two examples for every label. "
            f"Insufficient labels: {', '.join(too_small)}."
        )

    try:
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts,
            labels,
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

    model = create_classifier(random_state)
    model.fit(train_texts, train_labels)
    predictions = model.predict(test_texts)

    report = classification_report(
        test_labels,
        predictions,
        labels=label_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "labels": label_names,
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "class_metrics": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in label_names
        },
        "confusion_matrix": confusion_matrix(
            test_labels, predictions, labels=label_names
        ).tolist(),
        "train_examples": len(train_texts),
        "test_examples": len(test_texts),
    }


def print_report(results: dict[str, Any], file_path: Path | str) -> None:
    """Print accuracy, per-class metrics, and a labeled confusion matrix."""
    print("TaskTune TF-IDF + Logistic Regression Baseline")
    print(f"Dataset: {file_path}")
    print(
        f"Split: {results['train_examples']} training / "
        f"{results['test_examples']} test examples (stratified)"
    )
    print(f"Accuracy: {results['accuracy']:.4f}")

    print("\nPer-class metrics (precision / recall / F1 / support):")
    for label in results["labels"]:
        metrics = results["class_metrics"][label]
        print(
            f"  {label}: {metrics['precision']:.4f} / {metrics['recall']:.4f} / "
            f"{metrics['f1']:.4f} / {metrics['support']}"
        )

    labels = results["labels"]
    matrix = results["confusion_matrix"]
    label_width = max(14, max(len(label) for label in labels))
    cell_width = max(8, label_width)
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(f"{'actual \\ predicted':>{label_width}} " + " ".join(
        f"{label:>{cell_width}}" for label in labels
    ))
    for label, row in zip(labels, matrix):
        print(f"{label:>{label_width}} " + " ".join(
            f"{count:>{cell_width}d}" for count in row
        ))

    print(
        "\nThis small sample is useful for checking the baseline pipeline, "
        "not for claiming production-level performance."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the TaskTune TF-IDF + Logistic Regression baseline."
    )
    parser.add_argument(
        "--file",
        "-f",
        default="data/sample_dataset.jsonl",
        help="Path to the JSONL dataset (default: data/sample_dataset.jsonl)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction held out for testing (default: 0.25)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for reproducible splitting and model fitting (default: 42)",
    )
    args = parser.parse_args()

    try:
        results = run_experiment(args.file, args.test_size, args.random_state)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_report(results, args.file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
