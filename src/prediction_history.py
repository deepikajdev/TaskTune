"""Prediction history helpers for TaskTune Streamlit session state.

Maintains an in-memory session log of recent user queries, predicted labels,
and confidence scores without external databases or network services.
"""

from datetime import datetime, timezone
from typing import Any

MAX_HISTORY_ITEMS = 50


def create_history_record(
    message: str,
    predicted_label: str,
    confidence: float,
    sequence: int,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Create a structured prediction record for session state storage.

    Parameters
    ----------
    message : str
        The input message.
    predicted_label : str
        The predicted intent category.
    confidence : float
        Estimated model confidence (0.0 to 1.0).
    sequence : int
        1-based sequential query index in the active session.
    timestamp : str | None
        Optional ISO-formatted UTC timestamp string.

    Returns
    -------
    dict[str, Any]
        A clean dictionary representing the history entry.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    return {
        "sequence": sequence,
        "timestamp": timestamp,
        "message": message,
        "predicted_label": predicted_label,
        "confidence": float(confidence),
    }


def add_history_record(
    history: list[dict[str, Any]],
    message: str,
    predicted_label: str,
    confidence: float,
    max_items: int = MAX_HISTORY_ITEMS,
) -> list[dict[str, Any]]:
    """Append a new prediction to the history list and cap at max_items.

    Returns a new or updated list with the most recent items.
    """
    next_sequence = len(history) + 1
    record = create_history_record(
        message=message,
        predicted_label=predicted_label,
        confidence=confidence,
        sequence=next_sequence,
    )
    history.append(record)
    if len(history) > max_items:
        del history[:-max_items]
    return history


def clear_history() -> list[dict[str, Any]]:
    """Return an empty history list."""
    return []
