"""Compare a majority-class predictor with TF-IDF + Logistic Regression."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

if __package__:
    from .train_baseline import create_classifier, load_dataset
else:
    from train_baseline import create_classifier, load_dataset


def _evaluate(
    actual: list[str], predicted: list[str], labels: list[str]
) -> dict[str, Any]:
    """Calculate metrics using one shared label order for every model."""
    report = classification_report(
        actual,
        predicted,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in labels
        },
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=labels
        ).tolist(),
    }


def compare_baselines(
    file_path: Path | str,
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict[str, Any]:
    """Evaluate both models on the same stratified, held-out test examples."""
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

    train_texts = [texts[index] for index in train_indices]
    test_texts = [texts[index] for index in test_indices]
    train_labels = [labels[index] for index in train_indices]
    test_labels = [labels[index] for index in test_indices]

    training_label_counts = Counter(train_labels)
    majority_label = min(
        label_order, key=lambda label: (-training_label_counts[label], label)
    )
    majority_predictions = [majority_label] * len(test_labels)

    classifier = create_classifier(random_state)
    classifier.fit(train_texts, train_labels)
    classifier_predictions = classifier.predict(test_texts).tolist()

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
            "split_strategy": "stratified train/test split",
            "train_examples": len(train_indices),
            "test_examples": len(test_indices),
            "train_indices": train_indices,
            "test_indices": test_indices,
            "label_order": label_order,
            "confusion_matrix_axes": "rows = actual; columns = predicted",
        },
        "models": {
            "majority_class": {
                "predicted_label": majority_label,
                "metrics": _evaluate(test_labels, majority_predictions, label_order),
            },
            "tfidf_logistic_regression": {
                "vectorizer": "TfidfVectorizer; fit on training text only",
                "classifier": "LogisticRegression(max_iter=1000)",
                "metrics": _evaluate(
                    test_labels, classifier_predictions, label_order
                ),
            },
        },
    }


def save_results(results: dict[str, Any], output_path: Path | str) -> None:
    """Write the experiment results as readable JSON."""
    path = Path(output_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise OSError(f"Could not write results file '{path}': {exc}") from exc


def print_report(results: dict[str, Any], output_path: Path | str) -> None:
    """Print a concise comparison with labeled confusion matrices."""
    metadata = results["metadata"]
    print("TaskTune Baseline Comparison")
    print(f"Dataset: {metadata['dataset']} ({metadata['dataset_examples']} examples)")
    print(
        f"Split: {metadata['train_examples']} training / "
        f"{metadata['test_examples']} test examples; "
        f"stratified, seed {metadata['random_state']}"
    )

    for model_name, model_result in results["models"].items():
        metrics = model_result["metrics"]
        print(f"\n{model_name}")
        if "predicted_label" in model_result:
            print(f"  Always predicts: {model_result['predicted_label']}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Macro-F1: {metrics['macro_f1']:.4f}")
        print("  Per-class precision / recall / F1 / support:")
        for label in metadata["label_order"]:
            class_metrics = metrics["per_class"][label]
            print(
                f"    {label}: {class_metrics['precision']:.4f} / "
                f"{class_metrics['recall']:.4f} / {class_metrics['f1']:.4f} / "
                f"{class_metrics['support']}"
            )

        labels = metadata["label_order"]
        matrix = metrics["confusion_matrix"]
        width = max(14, max(len(label) for label in labels))
        print("  Confusion matrix (rows = actual, columns = predicted):")
        print(
            f"    {'actual \\ predicted':>{width}} "
            + " ".join(f"{label:>{width}}" for label in labels)
        )
        for label, row in zip(labels, matrix):
            print(
                f"    {label:>{width}} "
                + " ".join(f"{count:>{width}d}" for count in row)
            )

    print(f"\nSaved reproducible results to: {output_path}")
    print(
        "These sample results are for exercising the pipeline, not evidence "
        "of real-world performance."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare majority-class and TF-IDF Logistic Regression baselines."
    )
    parser.add_argument(
        "--file",
        "-f",
        default="data/sample_dataset.jsonl",
        help="Path to the JSONL dataset (default: data/sample_dataset.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="results/baseline_comparison.json",
        help="Where to save the JSON report (default: results/baseline_comparison.json)",
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
        results = compare_baselines(args.file, args.test_size, args.random_state)
        save_results(results, args.output)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_report(results, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
