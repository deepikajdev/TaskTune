"""Input validation helpers for TaskTune message classification.

Provides shared, beginner-readable validation logic used by both the CLI
and the Streamlit web application.
"""

MAX_MESSAGE_LENGTH = 500  # characters

# Sentinel object used for type-checking validation of unsupported input types
_UNSUPPORTED_TYPES = (int, float, bytes, list, dict, bool, type(None))


def validate_message(message: object) -> str:
    """Validate a user-supplied message and return the cleaned string.

    Parameters
    ----------
    message : object
        The raw value from the user (expected to be a str).

    Returns
    -------
    str
        The stripped message, guaranteed non-empty and within length limits.

    Raises
    ------
    TypeError
        If `message` is not a string.
    ValueError
        If the message is blank, whitespace-only, or exceeds MAX_MESSAGE_LENGTH.
    """
    if isinstance(message, _UNSUPPORTED_TYPES):
        raise TypeError(
            f"Message must be a string, got {type(message).__name__}. "
            "Please enter a text message."
        )

    if not isinstance(message, str):
        raise TypeError(
            f"Message must be a string, got {type(message).__name__}."
        )

    cleaned = message.strip()

    if not cleaned:
        raise ValueError(
            "Message cannot be blank or contain only whitespace. "
            "Please enter a customer message to classify."
        )

    if len(cleaned) > MAX_MESSAGE_LENGTH:
        raise ValueError(
            f"Message is too long ({len(cleaned)} characters). "
            f"Please shorten it to {MAX_MESSAGE_LENGTH} characters or fewer."
        )

    return cleaned


def friendly_validation_error(exc: Exception) -> str:
    """Format a validation exception as a friendly one-line user message."""
    return f"⚠️ {exc}"

