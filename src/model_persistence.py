"""Model serialization and persistence utilities for TaskTune.

Trust and Security Assumptions:
--------------------------------
This module uses `joblib` for persisting and restoring trained scikit-learn
pipelines. Like Python's standard `pickle`, `joblib` deserialization can execute
arbitrary code embedded within serialized files.

Security best practices:
1. Only deserialize model files from trusted sources and verified local paths.
2. Never load model files received over insecure networks or uploaded by
   unauthenticated users without cryptographic verification (e.g. SHA-256).
3. Treat saved model artifacts as executable binaries with full user permissions.
"""

import argparse
from pathlib import Path
import sys
from typing import Any

import joblib

if __package__:
    from .train_baseline import create_classifier, load_dataset
else:
    from train_baseline import create_classifier, load_dataset


def save_model(model: Any, path: Path | str) -> Path:
    """Serialize a trained model pipeline to disk using joblib.

    Parameters
    ----------
    model : Any
        A fitted scikit-learn estimator or Pipeline.
    path : Path | str
        Destination file path.

    Returns
    -------
    Path
        The destination path where the model was saved.
    """
    if not hasattr(model, "predict"):
        raise ValueError("Model must implement a `predict` method.")

    target_path = Path(path).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target_path)
    return target_path


def load_model(path: Path | str) -> Any:
    """Load a serialized model pipeline from disk.

    Trust assumption: Caller must ensure `path` points to a trusted local file.

    Parameters
    ----------
    path : Path | str
        Path to the serialized model file.

    Returns
    -------
    Any
        The deserialized scikit-learn estimator or Pipeline.
    """
    model_path = Path(path).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    try:
        model = joblib.load(model_path)
    except Exception as exc:
        raise ValueError(f"Failed to deserialize model file '{model_path}': {exc}") from exc

    if not hasattr(model, "predict"):
        raise ValueError("Loaded object does not appear to be a valid model (missing `predict`).")

    return model


def train_and_save_model(
    dataset_path: Path | str,
    output_path: Path | str,
    random_state: int = 42,
) -> Path:
    """Train the project baseline on the validated dataset and save it to disk.

    Parameters
    ----------
    dataset_path : Path | str
        Path to validated JSONL dataset.
    output_path : Path | str
        Target destination for the serialized model artifact.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    Path
        Path to the saved model file.
    """
    texts, labels = load_dataset(dataset_path)
    classifier = create_classifier(random_state=random_state)
    classifier.fit(texts, labels)
    return save_model(classifier, output_path)


def predict_with_model(
    model: Any,
    message: str,
    include_probabilities: bool = False,
) -> str | dict[str, Any]:
    """Generate a prediction and optional class probabilities using a loaded model."""
    clean_message = message.strip()
    if not clean_message:
        raise ValueError("Message cannot be empty or contain only whitespace.")

    predicted_label = str(model.predict([clean_message])[0])
    if not include_probabilities:
        return predicted_label

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([clean_message])[0]
        classes = model.classes_
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train, save, or inspect TaskTune classification models."
    )
    parser.add_argument(
        "--dataset",
        "-d",
        default="data/sample_dataset.jsonl",
        help="Path to training dataset (default: data/sample_dataset.jsonl)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="models/baseline_model.joblib",
        help="Output destination path for serialized model (default: models/baseline_model.joblib)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for model training (default: 42)",
    )
    args = parser.parse_args()

    try:
        saved_path = train_and_save_model(
            args.dataset, args.output, random_state=args.random_state
        )
        print(f"Successfully trained and saved model to: {saved_path}")
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

