"""TaskTune FastAPI prediction and health check endpoints.

Exposes RESTful endpoints for customer inquiry classification,
reusing TaskTune's existing trained pipeline and input validation.
"""

from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.input_validation import MAX_MESSAGE_LENGTH, validate_message
from src.predict_message import predict_with_confidence
from src.report_loader import get_baseline_metrics

DATASET_PATH = PROJECT_ROOT / "data" / "sample_dataset.jsonl"
BASELINE_REPORT_PATH = PROJECT_ROOT / "results" / "baseline_comparison.json"

app = FastAPI(
    title="TaskTune Inquiry Classification API",
    description=(
        "Production-ready REST API for classifying customer support inquiries "
        "into billing, technical_support, or account_access categories."
    ),
    version="0.2.0",
)


class PredictionRequest(BaseModel):
    message: str = Field(
        ...,
        description="Raw customer support message text",
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
        examples=["I was charged twice on my credit card for the monthly renewal."],
    )
    include_probabilities: bool = Field(
        default=True,
        description="Whether to include full per-class estimated probability distribution",
    )


class PredictionResponse(BaseModel):
    message: str
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]
    disclaimer: str
    timestamp_utc: str


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model_ready: bool
    dataset_present: bool
    timestamp_utc: str


@app.get("/", tags=["General"])
def root() -> dict[str, Any]:
    """Root endpoint providing service metadata and documentation links."""
    return {
        "service": "TaskTune Inquiry Classification API",
        "version": "0.2.0",
        "docs_url": "/docs",
        "health_url": "/health",
        "predict_url": "/predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check() -> HealthResponse:
    """Health check verifying dataset readiness and service responsiveness."""
    dataset_exists = DATASET_PATH.is_file()
    return HealthResponse(
        status="healthy",
        service="TaskTune API",
        version="0.2.0",
        model_ready=dataset_exists,
        dataset_present=dataset_exists,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
def predict_endpoint(payload: PredictionRequest) -> PredictionResponse:
    """Classify a customer inquiry into an operational intent category."""
    try:
        cleaned = validate_message(payload.message)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    if not DATASET_PATH.is_file():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Training dataset unavailable. Service cannot train or load model.",
        )

    try:
        result = predict_with_confidence(cleaned, DATASET_PATH)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
        ) from exc

    return PredictionResponse(
        message=cleaned,
        predicted_label=result["predicted_label"],
        confidence=result["confidence"],
        probabilities=result["probabilities"] if payload.include_probabilities else {},
        disclaimer=result.get("note", "Scores are model estimates, not guaranteed correctness."),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/metrics", tags=["Evaluation"])
def get_metrics() -> dict[str, Any]:
    """Retrieve verified offline benchmark evaluation metrics from saved reports."""
    metrics = get_baseline_metrics(BASELINE_REPORT_PATH)
    if not metrics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation benchmark report not found at results/baseline_comparison.json",
        )
    return {
        "dataset": metrics.get("dataset", "sample_dataset.jsonl"),
        "train_examples": metrics.get("train_examples", 0),
        "test_examples": metrics.get("test_examples", 0),
        "accuracy": metrics.get("accuracy", 0.0),
        "macro_f1": metrics.get("macro_f1", 0.0),
        "per_class": metrics.get("per_class", {}),
        "notice": "These metrics reflect offline evaluation on the benchmark dataset.",
    }
