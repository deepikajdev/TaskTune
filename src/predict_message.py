"""Predict a customer inquiry category with TaskTune's text classifier."""

import argparse
from pathlib import Path
import sys
from typing import Any

if __package__:
    from .train_baseline import create_classifier, load_dataset
    from .model_persistence import load_model, predict_with_model
else:
    from train_baseline import create_classifier, load_dataset
    from model_persistence import load_model, predict_with_model


def predict_message(
    message: str,
    file_path: Path | str | None = "data/sample_dataset.jsonl",
    model_path: Path | str | None = None,
    include_probabilities: bool = False,
) -> str | dict[str, Any]:
    """Predict category for one new message using a dataset or pre-saved model.

    If model_path is provided, loads the pre-trained model directly.
    Otherwise, trains on the validated dataset at file_path.
    If include_probabilities is True, returns a dictionary containing the
    predicted label, confidence, per-class probabilities, and an explanatory note.
    Otherwise, returns only the predicted label string.
    """
    clean_message = message.strip()
    if not clean_message:
        raise ValueError("Message cannot be empty or contain only whitespace.")

    if model_path is not None:
        model = load_model(model_path)
        return predict_with_model(
            model, clean_message, include_probabilities=include_probabilities
        )

    if file_path is None:
        raise ValueError("Either file_path or model_path must be provided.")

    texts, labels = load_dataset(file_path)
    classifier = create_classifier()
    classifier.fit(texts, labels)

    return predict_with_model(
        classifier, clean_message, include_probabilities=include_probabilities
    )


def predict_with_confidence(
    message: str,
    file_path: Path | str | None = "data/sample_dataset.jsonl",
    model_path: Path | str | None = None,
) -> dict[str, Any]:
    """Convenience helper to predict one message with class probabilities."""
    result = predict_message(
        message,
        file_path=file_path,
        model_path=model_path,
        include_probabilities=True,
    )
    if isinstance(result, dict):
        return result
    return {"predicted_label": result, "probabilities": {}, "confidence": 1.0}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classify one customer message using TaskTune's TF-IDF baseline."
    )
    parser.add_argument("message", help="Customer message to classify")
    parser.add_argument(
        "--file",
        "-f",
        default="data/sample_dataset.jsonl",
        help="Path to the JSONL training data (default: data/sample_dataset.jsonl)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="Optional path to a pre-trained serialized model file (.joblib)",
    )
    parser.add_argument(
        "--scores",
        "--probabilities",
        action="store_true",
        dest="show_probabilities",
        help="Display predicted class probability scores and disclaimer.",
    )
    args = parser.parse_args()

    try:
        if args.show_probabilities:
            result = predict_with_confidence(
                args.message, file_path=args.file if not args.model else None, model_path=args.model
            )
            print(f"Predicted category: {result['predicted_label']}")
            print(f"Confidence score: {result['confidence']:.4f}")
            print("Class probabilities:")
            for label, score in sorted(result["probabilities"].items()):
                print(f"  {label}: {score:.4f}")
            print("Note: Confidence scores are model estimates, not guaranteed correctness.")
        else:
            label = predict_message(
                args.message, file_path=args.file if not args.model else None, model_path=args.model
            )
            print(f"Predicted category: {label}")
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
