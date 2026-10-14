"""Predict a customer inquiry category with TaskTune's text classifier."""

import argparse
import sys
from pathlib import Path

if __package__:
    from .train_baseline import create_classifier, load_dataset
else:
    from train_baseline import create_classifier, load_dataset


def predict_message(message: str, file_path: Path | str) -> str:
    """Train on the validated dataset and predict one new message."""
    if not message.strip():
        raise ValueError("Message cannot be empty or contain only whitespace.")

    texts, labels = load_dataset(file_path)
    classifier = create_classifier()
    classifier.fit(texts, labels)
    return str(classifier.predict([message.strip()])[0])


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
    args = parser.parse_args()

    try:
        label = predict_message(args.message, args.file)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Predicted category: {label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
