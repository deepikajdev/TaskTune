"""Compare two competitive text classification models on TaskTune data.

A fair machine learning comparison requires:
1. Identical data splits: Both models train on the exact same training examples
   and are tested on the exact same held-out test examples.
2. Identical feature pipelines: Both use TF-IDF fitted strictly on the training split
   to prevent data leakage.
3. Identical evaluation metrics: Both are evaluated on Accuracy, Macro-F1,
   per-class Precision/Recall/F1, and confusion matrices with identical label ordering.

This script compares two classical models suitable for customer support text routing:
- Model 1: Logistic Regression (optimization-based linear classifier)
- Model 2: Multinomial Naive Bayes (probabilistic frequency-based classifier)
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

if __package__:
    from .train_baseline import load_dataset
else:
    from train_baseline import load_dataset


def create_logistic_regression_pipeline(random_state: int = 42) -> Pipeline:
    """Create a TF-IDF + Logistic Regression pipeline."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )


def create_naive_bayes_pipeline() -> Pipeline:
    """Create a TF-IDF + Multinomial Naive Bayes pipeline."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("classifier", MultinomialNB()),
        ]
    )


def _evaluate(
    actual: list[str], predicted: list[str], labels: list[str]
) -> dict[str, Any]:
    """Calculate metrics using a unified label order."""
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


def compare_models(
    file_path: Path | str,
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train and evaluate both classifiers on the same stratified split.

    Args:
        file_path: Path to the JSONL dataset.
        test_size: Fraction of dataset reserved for testing.
        random_state: Seed for reproducible data splitting.

    Returns:
        Dictionary containing metadata, results for each model, and a side-by-side comparison.
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

    # Model 1: Logistic Regression
    lr_pipeline = create_logistic_regression_pipeline(random_state)
    lr_pipeline.fit(train_texts, train_labels)
    lr_preds = lr_pipeline.predict(test_texts).tolist()
    lr_metrics = _evaluate(test_labels, lr_preds, label_order)

    # Model 2: Multinomial Naive Bayes
    nb_pipeline = create_naive_bayes_pipeline()
    nb_pipeline.fit(train_texts, train_labels)
    nb_preds = nb_pipeline.predict(test_texts).tolist()
    nb_metrics = _evaluate(test_labels, nb_preds, label_order)

    # Side-by-side difference summary (Logistic Regression minus Naive Bayes)
    comparison_summary = {
        "accuracy_diff": round(lr_metrics["accuracy"] - nb_metrics["accuracy"], 4),
        "macro_f1_diff": round(lr_metrics["macro_f1"] - nb_metrics["macro_f1"], 4),
        "higher_accuracy_model": (
            "logistic_regression"
            if lr_metrics["accuracy"] > nb_metrics["accuracy"]
            else (
                "multinomial_nb"
                if nb_metrics["accuracy"] > lr_metrics["accuracy"]
                else "tied"
            )
        ),
        "higher_macro_f1_model": (
            "logistic_regression"
            if lr_metrics["macro_f1"] > nb_metrics["macro_f1"]
            else (
                "multinomial_nb"
                if nb_metrics["macro_f1"] > lr_metrics["macro_f1"]
                else "tied"
            )
        ),
    }

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
            "logistic_regression": {
                "name": "TF-IDF + Logistic Regression",
                "type": "Linear classifier (L-BFGS solver)",
                "metrics": lr_metrics,
            },
            "multinomial_nb": {
                "name": "TF-IDF + Multinomial Naive Bayes",
                "type": "Probabilistic classifier",
                "metrics": nb_metrics,
            },
        },
        "comparison_summary": comparison_summary,
    }


def save_results(results: dict[str, Any], output_path: Path | str) -> None:
    """Write model comparison results to JSON."""
    path = Path(output_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise OSError(f"Could not write results file '{path}': {exc}") from exc


def _format_matrix(matrix: list[list[int]], labels: list[str]) -> str:
    """Format a 2D confusion matrix with aligned headers."""
    col_width = max(18, max(len(l) for l in labels) + 2)
    header = f"{'actual \\ pred':>{col_width}} " + " ".join(
        f"{lbl:>{col_width}}" for lbl in labels
    )
    rows = [header]
    for lbl, row in zip(labels, matrix):
        formatted_row = f"{lbl:>{col_width}} " + " ".join(
            f"{val:>{col_width}}" for val in row
        )
        rows.append(formatted_row)
    return "\n".join(rows)


def print_report(results: dict[str, Any], output_path: Path | str | None = None) -> None:
    """Print a clean comparison report to the terminal."""
    meta = results["metadata"]
    lr_data = results["models"]["logistic_regression"]
    nb_data = results["models"]["multinomial_nb"]
    labels = meta["label_order"]

    print("=" * 75)
    print("TaskTune Classification Model Comparison")
    print(f"Dataset: {meta['dataset']}")
    print(
        f"Split  : {meta['train_examples']} training / "
        f"{meta['test_examples']} test examples (stratified)"
    )
    print("=" * 75)

    print("\nSummary Metrics Comparison:")
    print(f"{'Metric':<20} {'Logistic Regression':<24} {'Multinomial Naive Bayes':<24}")
    print("-" * 68)
    print(
        f"{'Accuracy':<20} "
        f"{lr_data['metrics']['accuracy']:<24.4f} "
        f"{nb_data['metrics']['accuracy']:<24.4f}"
    )
    print(
        f"{'Macro F1':<20} "
        f"{lr_data['metrics']['macro_f1']:<24.4f} "
        f"{nb_data['metrics']['macro_f1']:<24.4f}"
    )

    print("\nPer-Class F1 Score Comparison:")
    print(f"{'Class':<20} {'Logistic Regression':<24} {'Multinomial Naive Bayes':<24}")
    print("-" * 68)
    for lbl in labels:
        lr_f1 = lr_data["metrics"]["per_class"][lbl]["f1"]
        nb_f1 = nb_data["metrics"]["per_class"][lbl]["f1"]
        print(f"{lbl:<20} {lr_f1:<24.4f} {nb_f1:<24.4f}")

    print("\nConfusion Matrix: Logistic Regression")
    print(_format_matrix(lr_data["metrics"]["confusion_matrix"], labels))

    print("\nConfusion Matrix: Multinomial Naive Bayes")
    print(_format_matrix(nb_data["metrics"]["confusion_matrix"], labels))

    summary = results["comparison_summary"]
    print(f"\nHigher Accuracy Model: {summary['higher_accuracy_model']}")
    print(f"Higher Macro F1 Model: {summary['higher_macro_f1_model']}")

    if output_path:
        print(f"\nSaved detailed JSON comparison to: {output_path}")
    print("=" * 75)


def main() -> int:
    """CLI entry point for comparing classification models."""
    parser = argparse.ArgumentParser(
        description="Fairly compare two text classification models on TaskTune data."
    )
    parser.add_argument(
        "--file",
        type=str,
        default="data/sample_dataset.jsonl",
        help="Path to JSONL dataset (default: data/sample_dataset.jsonl).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction held out for testing (default: 0.25).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for reproducible split (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/model_comparison.json",
        help="Path to save JSON comparison results (default: results/model_comparison.json).",
    )
    args = parser.parse_args()

    try:
        results = compare_models(
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
