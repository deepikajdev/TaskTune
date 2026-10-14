"""Evaluate the TaskTune classifier with stratified cross-validation."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import sklearn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold

if __package__:
    from .train_baseline import create_classifier, load_dataset
else:
    from train_baseline import create_classifier, load_dataset


def _fold_metrics(
    actual: list[str], predicted: list[str], label_order: list[str]
) -> dict[str, Any]:
    """Calculate one fold's overall and per-class evaluation metrics."""
    report = classification_report(
        actual,
        predicted,
        labels=label_order,
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
            for label in label_order
        },
    }


def _summarize_metrics(folds: list[dict[str, Any]]) -> dict[str, Any]:
    """Return fold-wise means and population standard deviations."""
    metric_names = ("accuracy", "macro_f1")
    averages = {
        name: statistics.mean(fold["metrics"][name] for fold in folds)
        for name in metric_names
    }
    standard_deviations = {
        name: statistics.pstdev(fold["metrics"][name] for fold in folds)
        for name in metric_names
    }

    labels = list(folds[0]["metrics"]["per_class"])
    averages["per_class"] = {
        label: {
            metric: statistics.mean(
                fold["metrics"]["per_class"][label][metric] for fold in folds
            )
            for metric in ("precision", "recall", "f1")
        }
        for label in labels
    }
    standard_deviations["per_class"] = {
        label: {
            metric: statistics.pstdev(
                fold["metrics"]["per_class"][label][metric] for fold in folds
            )
            for metric in ("precision", "recall", "f1")
        }
        for label in labels
    }
    return {"mean": averages, "population_std": standard_deviations}


def evaluate_cross_validation(
    file_path: Path | str,
    folds: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run stratified folds with TF-IDF fitted independently inside each fold."""
    if folds < 2:
        raise ValueError("--folds must be at least 2.")

    dataset_path = Path(file_path)
    texts, labels = load_dataset(dataset_path)
    label_order = sorted(set(labels))
    label_counts = Counter(labels)
    too_small = [
        label for label in label_order if label_counts[label] < folds
    ]
    if too_small:
        details = ", ".join(
            f"{label} ({label_counts[label]})" for label in too_small
        )
        raise ValueError(
            f"Cannot use {folds} folds because every class needs at least "
            f"{folds} examples. Too-small classes: {details}."
        )

    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )
    fold_results: list[dict[str, Any]] = []
    for fold_number, (train_indices, test_indices) in enumerate(
        splitter.split(texts, labels), start=1
    ):
        train_texts = [texts[index] for index in train_indices]
        train_labels = [labels[index] for index in train_indices]
        test_texts = [texts[index] for index in test_indices]
        test_labels = [labels[index] for index in test_indices]

        classifier = create_classifier(random_state)
        classifier.fit(train_texts, train_labels)
        predictions = classifier.predict(test_texts).tolist()
        fold_results.append(
            {
                "fold": fold_number,
                "train_examples": len(train_indices),
                "test_examples": len(test_indices),
                "test_class_counts": dict(sorted(Counter(test_labels).items())),
                "metrics": _fold_metrics(test_labels, predictions, label_order),
            }
        )

    return {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": str(dataset_path),
            "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "dataset_examples": len(texts),
            "class_counts": dict(sorted(label_counts.items())),
            "python_version": sys.version.split()[0],
            "scikit_learn_version": sklearn.__version__,
            "evaluation": "StratifiedKFold with shuffle=True",
            "folds": folds,
            "random_state": random_state,
            "label_order": label_order,
            "standard_deviation": "population standard deviation across folds",
        },
        "fold_results": fold_results,
        "summary": _summarize_metrics(fold_results),
    }


def save_results(results: dict[str, Any], output_path: Path | str) -> None:
    """Save cross-validation metrics and metadata as readable JSON."""
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
    """Print fold scores, per-class scores, and fold-wise summary statistics."""
    metadata = results["metadata"]
    print("TaskTune Stratified Cross-Validation")
    print(
        f"Dataset: {metadata['dataset']} "
        f"({metadata['dataset_examples']} examples)"
    )
    print(f"Folds: {metadata['folds']} (stratified, seed {metadata['random_state']})")
    print(f"Class counts: {metadata['class_counts']}")

    for fold in results["fold_results"]:
        metrics = fold["metrics"]
        print(
            f"\nFold {fold['fold']}: {fold['train_examples']} train / "
            f"{fold['test_examples']} test examples; "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"macro-F1={metrics['macro_f1']:.4f}"
        )
        for label in metadata["label_order"]:
            class_metrics = metrics["per_class"][label]
            print(
                f"  {label}: precision={class_metrics['precision']:.4f}, "
                f"recall={class_metrics['recall']:.4f}, "
                f"F1={class_metrics['f1']:.4f}"
            )

    print("\nAcross-fold mean (population standard deviation):")
    for metric in ("accuracy", "macro_f1"):
        mean = results["summary"]["mean"][metric]
        std = results["summary"]["population_std"][metric]
        print(f"  {metric}: {mean:.4f} ({std:.4f})")
    for label in metadata["label_order"]:
        means = results["summary"]["mean"]["per_class"][label]
        stds = results["summary"]["population_std"]["per_class"][label]
        print(
            f"  {label}: precision={means['precision']:.4f} ({stds['precision']:.4f}), "
            f"recall={means['recall']:.4f} ({stds['recall']:.4f}), "
            f"F1={means['f1']:.4f} ({stds['f1']:.4f})"
        )
    print(f"\nSaved JSON results to: {output_path}")
    print(
        "This small illustrative dataset provides an initial cross-validation "
        "estimate, not proof of real-world performance."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate TaskTune's TF-IDF classifier with stratified cross-validation."
    )
    parser.add_argument(
        "--file",
        "-f",
        default="data/sample_dataset.jsonl",
        help="Path to the JSONL dataset (default: data/sample_dataset.jsonl)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of stratified folds (default: 5)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for reproducible folds and model fitting (default: 42)",
    )
    parser.add_argument(
        "--output",
        default="results/cross_validation.json",
        help="Where to save results (default: results/cross_validation.json)",
    )
    args = parser.parse_args()

    try:
        results = evaluate_cross_validation(
            args.file,
            folds=args.folds,
            random_state=args.random_state,
        )
        save_results(results, args.output)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_report(results, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
