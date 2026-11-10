"""Helpers for loading and formatting saved evaluation reports.

Reads saved evaluation artifacts from the results/ directory and extracts
key performance indicators (accuracy, macro-F1, per-class metrics).
Handles missing or unreadable files gracefully without fabricating numbers.
"""

import json
from pathlib import Path
from typing import Any


def load_json_report(file_path: Path | str) -> dict[str, Any] | None:
    """Load a JSON evaluation report safely.

    Returns the parsed dictionary if valid, or None if the file is missing
    or invalid.
    """
    path = Path(file_path)
    if not path.is_file():
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return None
    except (OSError, json.JSONDecodeError):
        return None


def get_baseline_metrics(
    report_path: Path | str = "results/baseline_comparison.json",
) -> dict[str, Any] | None:
    """Extract baseline Logistic Regression evaluation metrics.

    Returns a dict with:
        - accuracy (float)
        - macro_f1 (float)
        - per_class (dict of class -> {precision, recall, f1, support})
        - train_examples (int)
        - test_examples (int)
        - created_at (str)
    Or None if the file is missing or lacks expected keys.
    """
    data = load_json_report(report_path)
    if not data:
        return None

    try:
        model_info = data["models"]["tfidf_logistic_regression"]
        metrics = model_info["metrics"]
        metadata = data.get("metadata", {})

        return {
            "accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "per_class": metrics.get("per_class", {}),
            "train_examples": metadata.get("train_examples", 0),
            "test_examples": metadata.get("test_examples", 0),
            "created_at": metadata.get("created_at_utc", "Unknown"),
            "dataset": metadata.get("dataset", "sample_dataset.jsonl"),
        }
    except (KeyError, TypeError, ValueError):
        return None


def get_cross_validation_metrics(
    report_path: Path | str = "results/cross_validation.json",
) -> dict[str, Any] | None:
    """Extract 5-fold cross-validation metrics summary.

    Returns a dict with:
        - mean_accuracy (float)
        - std_accuracy (float)
        - mean_macro_f1 (float)
        - std_macro_f1 (float)
        - num_folds (int)
        - per_class (dict of class -> {f1, precision, recall})
    Or None if missing.
    """
    data = load_json_report(report_path)
    if not data:
        return None

    try:
        summary = data.get("summary", {})
        mean_metrics = summary.get("mean", {})
        std_metrics = summary.get("population_std", {})
        folds = data.get("fold_results", [])

        return {
            "num_folds": len(folds),
            "mean_accuracy": float(mean_metrics.get("accuracy", 0.0)),
            "std_accuracy": float(std_metrics.get("accuracy", 0.0)),
            "mean_macro_f1": float(mean_metrics.get("macro_f1", 0.0)),
            "std_macro_f1": float(std_metrics.get("macro_f1", 0.0)),
            "per_class": mean_metrics.get("per_class", {}),
        }
    except (KeyError, TypeError, ValueError):
        return None
