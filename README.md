# TaskTune: Customer Support Inquiry Classification & Evaluation Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

TaskTune is an end-to-end machine learning system and developer evaluation framework for automated customer support inquiry routing. It features verified classical ML baselines, deterministic offline benchmarking pipelines, a real-time Streamlit dashboard, a high-throughput FastAPI inference microservice, and serialization workflows.

---

## 📌 Executive Summary & Architecture

```
                              ┌───────────────────────────────────────────────┐
                              │           Customer Support Inquiry            │
                              └──────────────────────┬────────────────────────┘
                                                     │
                                                     ▼
                                      ┌──────────────────────────────┐
                                      │   Input Validation & Cleansing│
                                      │ (Length, whitespace, Unicode) │
                                      └──────────────┬───────────────┘
                                                     │
                                                     ▼
                                      ┌──────────────────────────────┐
                                      │    TF-IDF Vectorization      │
                                      │ (Unigrams + Bigrams, Sublinear)
                                      └──────────────┬───────────────┘
                                                     │
                                                     ▼
                                      ┌──────────────────────────────┐
                                      │ Logistic Regression Baseline │
                                      │   (Softmax Probabilities)    │
                                      └──────┬────────────────┬──────┘
                                             │                │
                      ┌──────────────────────┘                └──────────────────────┐
                      ▼                                                              ▼
        ┌───────────────────────────┐                                  ┌───────────────────────────┐
        │   FastAPI Microservice    │                                  │   Streamlit Web Dashboard │
        │  `GET /health`            │                                  │  - Live classification    │
        │  `POST /predict`          │                                  │  - Class probabilities    │
        │  `GET /metrics`           │                                  │  - Session query history  │
        └───────────────────────────┘                                  └───────────────────────────┘
```

### Core Capabilities
1. **Classical ML Classification Pipeline:** TF-IDF unigram + bigram representation with sublinear scaling paired with L2-regularized Logistic Regression and Multinomial Naive Bayes.
2. **Leakage-Free Validation & Cross-Validation:** Stratified train/test splitting (75/25) and 5-fold cross-validation where all text preprocessing and vocabulary extraction occur strictly within training folds.
3. **Interactive Streamlit Web Dashboard (`app.py`):** Real-time inquiry classification, class probability progress indicators, confidence rating badges, sample query chips, in-memory session history, and live evaluation metric views.
4. **Production-Ready FastAPI Service (`src/api.py`):** Asynchronous REST API with Pydantic validation, structured error responses, automated Swagger documentation, and health monitoring endpoints.
5. **Model Persistence & Deserialization (`src/model_persistence.py`):** Serializes fitted pipelines with `joblib` for rapid deployment without cold-start training overhead.

---

## 🎯 The Core Problem & Motivation

Enterprise support desks handle thousands of inbound requests daily. Defaulting to closed-source commercial LLM APIs for simple classification introduces:
- High recurring per-token cost
- Substantial network latency overhead (300ms–1500ms vs <1ms for classical models)
- Privacy risks associated with sending un-redacted user text to third-party endpoints
- Rate-limiting bottlenecks during traffic surges

TaskTune demonstrates that a well-calibrated, lightweight classical model achieves high accuracy and sub-millisecond latency on operational routing tasks with zero external API fees.

### Target Classification Intents
- **`billing`** (33.3%): Invoices, unexpected charges, refund requests, tax exemption, payment retries, discount codes.
- **`technical_support`** (33.3%): Application crashes, HTTP 500/504 errors, sync timeouts, UI glitches, API failures.
- **`account_access`** (33.3%): Password resets, 2FA/MFA token recovery, account lockouts, SSO errors, seat provisioning.

---

## 📊 Evaluation Benchmarks & Experimental Results

All experiments are deterministic (random seed `42`) and evaluated on the curated 300-example benchmark dataset:

### 1. Held-Out Stratified Test Set (225 Train / 75 Test Examples)

| Model Configuration | Test Accuracy | Macro F1-Score | Inference Latency |
| :--- | :---: | :---: | :---: |
| **Majority-Class Heuristic** | 33.33% | 0.1667 | <0.01 ms |
| **Multinomial Naive Bayes** | **93.33%** | **0.9335** | **~0.06 ms** |
| **TF-IDF + Logistic Regression (Baseline)** | **92.00%** | **0.9199** | **~0.08 ms** |

### 2. Per-Class Test Set Performance (Logistic Regression)

| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **`account_access`** | 95.65% | 88.00% | 0.9167 | 25 |
| **`billing`** | 88.46% | 92.00% | 0.9020 | 25 |
| **`technical_support`** | 92.31% | 96.00% | 0.9412 | 25 |
| **Macro Average** | **92.14%** | **92.00%** | **0.9199** | 75 |

### 3. Confusion Matrix (Rows = Actual, Columns = Predicted)

```text
                     Predicted: account_access   Predicted: billing   Predicted: technical_support
Actual: account_access                   22                    2                          1
Actual: billing                           1                   23                          1
Actual: technical_support                 0                    1                         24
```

### 4. Stratified 5-Fold Cross-Validation Summary

| Fold Metric | Mean | Population Std Dev |
| :--- | :---: | :---: |
| **CV Accuracy** | **92.67%** | ± 2.49% |
| **CV Macro-F1** | **92.64%** | ± 2.52% |
| `account_access` F1 | 91.99% | ± 2.95% |
| `billing` F1 | 91.48% | ± 2.99% |
| `technical_support` F1 | 94.46% | ± 3.82% |

*Reports generated reproducibly via `python src/evaluate_cross_validation.py`.*

---

## 🚀 Quick Start Guide

### 1. Installation & Environment Setup

```bash
# Clone the repository
git clone https://github.com/deepikajdev/TaskTune.git
cd TaskTune

# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Activate on Linux/macOS:
source .venv/bin/activate

# Install core dependencies:
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Application

```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser. The dashboard provides:
- **Live Classifier:** Test inquiries, view predicted categories, confidence ratings (🟢 High, 🟡 Moderate, 🔴 Low), and probability distributions.
- **Evaluation Metrics:** Interactive views of accuracy, macro-F1, per-class tables, and 5-fold CV statistics.
- **Session History:** In-memory query log with timestamps, sequential query counters, and clear-history action.

### 3. Launch the FastAPI Prediction Microservice

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```
Interactive API docs are available at **`http://127.0.0.1:8000/docs`**.

#### API Usage Examples

**Classify an Inquiry:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"message": "I was charged twice on my invoice for subscription renewal.", "include_probabilities": true}'
```

**Response:**
```json
{
  "message": "I was charged twice on my invoice for subscription renewal.",
  "predicted_label": "billing",
  "confidence": 0.8872,
  "probabilities": {
    "account_access": 0.0524,
    "billing": 0.8872,
    "technical_support": 0.0604
  },
  "disclaimer": "Confidence scores are model estimates, not guaranteed correctness.",
  "timestamp_utc": "2026-11-27T18:15:00.000000Z"
}
```

**Health Check:**
```bash
curl http://127.0.0.1:8000/health
```

---

## 🛠️ CLI Tooling & Pipeline Commands

```bash
# Validate dataset format, required fields, and duplicate detection
python src/validate_dataset.py --file data/sample_dataset.jsonl

# Profile dataset distributions, vocabulary count, and text length stats
python src/profile_dataset.py --output results/dataset_profile.json

# Train baseline model and display held-out confusion matrix
python src/train_baseline.py

# Compare Logistic Regression against Majority-Class predictor
python src/compare_baselines.py --output results/baseline_comparison.json

# Compare Logistic Regression against Multinomial Naive Bayes
python src/compare_models.py --output results/model_comparison.json

# Perform granular error analysis on test split misclassifications
python src/analyze_errors.py --output results/error_analysis.json

# Run leakage-free stratified 5-fold cross-validation
python src/evaluate_cross_validation.py --folds 5 --output results/cross_validation.json

# Predict single message from CLI with estimated confidence
python src/predict_message.py "I forgot my password and cannot sign in." --scores

# Train and serialize model to disk
python src/model_persistence.py --dataset data/sample_dataset.jsonl --output models/baseline_model.joblib

# Predict using pre-serialized model artifact
python src/predict_message.py "Desktop client crashed on startup." --model models/baseline_model.joblib --scores
```

---

## 🧪 Testing & Code Quality

TaskTune includes a comprehensive automated test suite with **over 80 unit and integration tests**:

```bash
# Run the full test suite
python -m unittest discover -s tests -v
```

Covered test modules:
- `test_validate_dataset.py`: Schema validation, missing fields, duplicates, Unicode.
- `test_profile_dataset.py`: Vocabulary metrics, character/word length distributions.
- `test_train_baseline.py`: Reproducible stratified splitting, data leakage prevention.
- `test_compare_baselines.py`: Majority-class baseline comparison.
- `test_compare_models.py`: Logistic Regression vs. Naive Bayes, latency recording.
- `test_analyze_errors.py`: Error rate diagnosis, confusion matrices, classification reports.
- `test_evaluate_cross_validation.py`: K-fold determinism, across-fold variance metrics.
- `test_predict_message.py`: CLI predictions, probability scoring, error handling.
- `test_model_persistence.py`: Joblib round-trip serialization and trust assumptions.
- `test_input_validation.py`: Empty strings, whitespace, type enforcement, 500-char bounds.
- `test_prediction_history.py`: In-memory session logging, FIFO capping, clear action.
- `test_report_loader.py`: Safe JSON report parsing and missing-file fallbacks.
- `test_ui_helpers.py`: Confidence rating categories, badges, text truncation.
- `test_api.py`: FastAPI endpoints, status codes, input validation, Pydantic schemas.

---

## ⚠️ Limitations & Real-World Considerations

1. **Curated Synthetic Dataset:** The 300 benchmark inquiries are synthetically generated and curated to exercise ML pipeline stability; they are not real customer records.
2. **Lexical Feature Limitations:** Bag-of-words and n-gram TF-IDF representations rely on surface vocabulary. Highly cross-cutting language or severe spelling errors can lead to misclassifications.
3. **Probability Calibration:** Softmax probabilities reflect the model's confidence over the trained linear hyperplane, not calibrated ground-truth correctness.
4. **Session State Scope:** The Streamlit web interface logs history in `st.session_state` only. No persistent external database is added.
5. **Model Deserialization Security:** `joblib` artifacts execute arbitrary Python bytecode. Models should only be loaded from verified, trusted local storage.

---

## 📜 License
This project is open-source under the MIT License.
