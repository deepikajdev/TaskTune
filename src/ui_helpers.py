"""UI helpers and sample queries for the TaskTune Streamlit dashboard."""

from typing import Final

SAMPLE_QUERIES: Final[dict[str, str]] = {
    "💳 Billing": "I was charged twice on my credit card for the monthly subscription.",
    "🔧 Tech Support": "The web app crashes with a 500 internal server error when uploading files.",
    "🔑 Account Access": "I forgot my password and cannot receive the 2FA authentication code.",
}

LABEL_EMOJIS: Final[dict[str, str]] = {
    "billing": "💳",
    "technical_support": "🔧",
    "account_access": "🔑",
}


def get_label_emoji(label: str) -> str:
    """Return the associated emoji for a customer inquiry category."""
    return LABEL_EMOJIS.get(label.strip().lower(), "📌")


def truncate_text(text: str, max_chars: int = 70) -> str:
    """Truncate text cleanly with an ellipsis if it exceeds max_chars."""
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    return f"{stripped[:max_chars].rstrip()}…"


def get_confidence_rating(confidence: float) -> tuple[str, str]:
    """Return a descriptive confidence label and emoji indicator.

    Parameters
    ----------
    confidence : float
        Confidence score between 0.0 and 1.0.

    Returns
    -------
    tuple[str, str]
        (rating_text, emoji_icon)
    """
    if confidence >= 0.80:
        return "High Confidence", "🟢"
    if confidence >= 0.50:
        return "Moderate Confidence", "🟡"
    return "Low Confidence", "🔴"

