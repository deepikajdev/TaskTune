"""Predict a customer inquiry category with TaskTune's text classifier."""

import argparse
from pathlib import Path
import sys
from typing import Any

if __package__:
    from .train_baseline import create_classifier, load_dataset
else:
    from train_baseline import create_classifier, load_dataset


def predict_message(
    message: str,
    file_path: Path | str,
    include_probabilities: bool = False,
) -> str | dict[str, Any]:
    """Train on the validated dataset and predict one new message.

    If include_probabilities is True, returns a dictionary containing the
    predicted label, confidence, per-class probabilities, and an explanatory note.
    Otherwise, returns only the predicted label string.
    """
    if not message.strip():
        raise ValueError("Message cannot be empty or contain only whitespace.")

    texts, labels = load_dataset(file_path)
    classifier = create_classifier()
    classifier.fit(texts, labels)

    clean_message = message.strip()
    predicted_label = str(classifier.predict([clean_message])[0])

    if not include_probabilities:
        return predicted_label

    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba([clean_message])[0]
        classes = classifier.classes_
        probabilities = {
            str(cls_name): float(prob) for cls_name, prob in zip(classes, probs)
        }
    else:
        probabilities = {predicted_label: 1.0}

    confidence = probabilities.get(predicted_label, float(max(probabilities.values())))
    return {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "probabilities": probabilities,
        "note": "Confidence scores are model estimates, not guaranteed correctness.",
    }


def predict_with_confidence(
    message: str, file_path: Path | str
) -> dict[str, Any]:
    """Convenience helper to predict one message with class probabilities."""
    result = predict_message(message, file_path, include_probabilities=True)
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
        "--scores",
        "--probabilities",
        action="store_true",
        dest="show_probabilities",
        help="Display predicted class probability scores and disclaimer.",
    )
    args = parser.parse_args()

    try:
        if args.show_probabilities:
            result = predict_with_confidence(args.message, args.file)
            print(f"Predicted category: {result['predicted_label']}")
            print(f"Confidence score: {result['confidence']:.4f}")
            print("Class probabilities:")
            for label, score in sorted(result["probabilities"].items()):
                print(f"  {label}: {score:.4f}")
            print("Note: Confidence scores are model estimates, not guaranteed correctness.")
        else:
            label = predict_message(args.message, args.file)
            print(f"Predicted category: {label}")
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
