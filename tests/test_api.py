"""Unit and integration tests for TaskTune FastAPI endpoints."""

import unittest
from fastapi.testclient import TestClient

from src.api import app
from src.validate_dataset import ALLOWED_LABELS


class TestFastAPIEndpoints(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_root_endpoint(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["service"], "TaskTune Inquiry Classification API")
        self.assertIn("docs_url", data)

    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["dataset_present"])

    def test_predict_valid_billing_message(self) -> None:
        payload = {
            "message": "I was charged twice on my credit card for the monthly subscription.",
            "include_probabilities": True,
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["predicted_label"], "billing")
        self.assertIn(data["predicted_label"], ALLOWED_LABELS)
        self.assertGreater(data["confidence"], 0.5)
        self.assertEqual(set(data["probabilities"].keys()), ALLOWED_LABELS)

    def test_predict_technical_support_message(self) -> None:
        payload = {
            "message": "The application crashes with a 500 internal server error when uploading files.",
            "include_probabilities": True,
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["predicted_label"], "technical_support")

    def test_predict_empty_message_returns_422_or_400(self) -> None:
        payload = {"message": "", "include_probabilities": True}
        response = self.client.post("/predict", json=payload)
        self.assertIn(response.status_code, (400, 422))

    def test_predict_whitespace_message_returns_400(self) -> None:
        payload = {"message": "   \n\t  ", "include_probabilities": True}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("detail", response.json())

    def test_predict_too_long_message_returns_400_or_422(self) -> None:
        long_message = "a" * 501
        payload = {"message": long_message, "include_probabilities": True}
        response = self.client.post("/predict", json=payload)
        self.assertIn(response.status_code, (400, 422))

    def test_metrics_endpoint(self) -> None:
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("accuracy", data)
        self.assertIn("macro_f1", data)
        self.assertGreater(data["accuracy"], 0.8)


if __name__ == "__main__":
    unittest.main()
