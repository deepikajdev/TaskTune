# TaskTune: Small Language Model Fine-Tuning & Evaluation

TaskTune is an open-source evaluation framework and developer toolkit for customer-support text classification. It provides classical machine-learning baselines, offline evaluation benchmarks, an interactive Streamlit web dashboard, and serialized model workflows.

1. **Classical baseline comparison:** Majority-class prediction versus TF-IDF + Logistic Regression, evaluated on the same held-out, stratified test split.
2. **Interactive web application:** Lightweight Streamlit dashboard with real-time intent prediction, class probability breakdown, input validation, in-memory session query logs, and benchmark reports.
3. **Future experiments:** Prompted and fine-tuned small language models (SLMs) will be explored in future milestones.

---

## 📌 Project Status

> **Important:** The classical baseline experiment and interactive Streamlit web application are fully implemented. The bundled dataset contains 75 illustrative examples (25 per class); it is designed as a pipeline sanity check and developer smoke test—not an estimate of real-world or production performance.

---

## 🎯 The Core Problem & Motivation

Enterprise applications often default to massive closed-source APIs for simple text classification. However, large models introduce recurring costs, privacy concerns, network latency overhead, and API rate limits.

**Small Language Models (SLMs)** (e.g., under 3 billion parameters) and highly optimized classical baselines offer lightweight alternatives. TaskTune establishes reproducible benchmarks and developer interfaces to evaluate these trade-offs honestly.

### Target Classification Task
Categorizing inbound customer inquiries into three core operational intents:
- **`billing`**: Invoices, unexpected charges, refund requests, payment retries, and discount codes.
- **`technical_support`**: Software bugs, application crashes, 500/504 errors, broken UI, and sync timeouts.
- **`account_access`**: Forgotten passwords, 2FA authenticator recovery, account lockouts, and SSO failures.

### Dataset Format: Why JSON Lines (`.jsonl`)?
We format datasets as **JSON Lines (`.jsonl`)**, where every line is a standalone JSON object:
```json
{"text": "The web application keeps crashing with error code 500.", "label": "technical_support"}
```
- **Stream-friendly:** Can be processed line-by-line without loading entire files into memory.
- **Robust:** A single corrupted record can be easily flagged by line number without breaking a massive JSON document.
- **Standard:** Directly compatible with Hugging Face `datasets` and modern LLM fine-tuning pipelines.

---

## 🖥️ Streamlit Web Application

TaskTune includes a lightweight, clean web dashboard built with Streamlit (`app.py`).

### Key Features:
- **💬 Live Classifier:** Predict customer intent categories with estimated class probabilities, confidence ratings (🟢 High, 🟡 Moderate, 🔴 Low), and illustrative example prompt chips.
- **🛡️ Input Validation:** Handles empty messages, whitespace padding, unsupported types, and enforces a sensible 500-character maximum input length with friendly error notices.
- **📊 Offline Evaluation Metrics:** Renders held-out test split metrics (accuracy, macro-F1, per-class table) and 5-fold cross-validation results loaded directly from actual saved reports in `results/`. Missing reports are handled gracefully with helpful generation instructions.
- **📜 In-Memory Session History:** Keeps a session log of recent queries with sequence numbers, timestamps, and predicted categories. Includes a one-click "Clear History" action without any external database or background service.

### Launching the Web App:

From Windows PowerShell or terminal:

```powershell
streamlit run app.py
```

Or run directly via Python module:
```powershell
python -m streamlit run app.py
```

The web dashboard opens at `http://localhost:8501`.

---

## 📁 Project Structure

```text
tasktune/
├── .gitignore                   # Ignores virtual envs, models, checkpoints, and cached files
├── README.md                    # Project documentation, guides, and roadmap
├── requirements.txt             # Scikit-learn, joblib, and Streamlit dependencies
├── app.py                       # Interactive Streamlit prediction and metrics dashboard
├── data/
│   ├── README.md                # Data directory guide and schema specification
│   └── sample_dataset.jsonl     # 75 illustrative inquiries, balanced across 3 classes
├── docs/
│   └── QUICKSTART.md            # Windows PowerShell quick-start guide
├── notebooks/                   # Exploratory analysis and training notebooks
├── results/                     # Reproducible metrics, logs, and diagnostic artifacts
│   ├── README.md                # Guide to evaluation JSON artifacts and dashboard integration
│   ├── baseline_comparison.json # Majority-class vs Logistic Regression comparison
│   ├── cross_validation.json    # 5-fold stratified cross-validation metrics
│   ├── error_analysis.json      # Per-class error counts and misclassified examples
│   ├── model_comparison.json    # Logistic Regression vs Naive Bayes comparison
│   └── dataset_profile.json     # Record health, class balance, and text statistics
├── src/
│   ├── __init__.py              # Package initializer
│   ├── validate_dataset.py      # Dataset schema, label, and duplicate validator
│   ├── profile_dataset.py       # Dataset profiling (lengths, vocabulary, anomalies)
│   ├── train_baseline.py        # TF-IDF + Logistic Regression training and evaluation
│   ├── compare_baselines.py     # Majority-class versus Logistic Regression comparison
│   ├── compare_models.py        # Logistic Regression versus Naive Bayes fair comparison
│   ├── analyze_errors.py        # Granular classifier error analysis and misclassifications
│   ├── predict_message.py       # Command-line prediction with confidence scores & model loading
│   ├── evaluate_cross_validation.py # Stratified 5-fold cross-validation evaluation
│   ├── model_persistence.py     # Model training, joblib serialization, and deserialization
│   ├── input_validation.py      # Shared message validation and friendly error handling
│   ├── prediction_history.py    # In-memory session state history manager
│   ├── report_loader.py         # Graceful JSON evaluation report parser
│   └── ui_helpers.py            # Dashboard badges, confidence ratings, and sample queries
└── tests/
    ├── test_validate_dataset.py # Automated unit tests for data validation
    ├── test_profile_dataset.py  # Tests for dataset profiling and statistics
    ├── test_train_baseline.py   # Automated unit tests for the baseline
    ├── test_compare_baselines.py # Tests for majority vs baseline comparison
    ├── test_compare_models.py   # Tests for Logistic Regression vs Naive Bayes
    ├── test_analyze_errors.py   # Tests for classification error analysis
    ├── test_predict_message.py  # Tests for prediction CLI & confidence scores
    ├── test_evaluate_cross_validation.py # Cross-validation tests
    ├── test_model_persistence.py # Tests for serialization round-trip & trust assumptions
    ├── test_input_validation.py # Tests for message length, type, and whitespace checks
    ├── test_prediction_history.py # Tests for in-memory session history
    ├── test_report_loader.py    # Tests for saved report parsing and missing file fallbacks
    └── test_ui_helpers.py       # Tests for UI emojis, ratings, and text truncation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Git

### 2. Install Dependencies (Windows PowerShell)
From the repository root, create and activate a virtual environment, then install project dependencies:

```powershell
# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell):
.venv\Scripts\Activate.ps1

# (If script execution is restricted on Windows):
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

# Install dependencies:
pip install -r requirements.txt
```

---

## 🛠️ CLI Usage & Tooling Guide

### 1. Predict Category with Confidence Scores
Classify a message using TF-IDF + Logistic Regression with estimated class probabilities:
```powershell
python src/predict_message.py "I was charged twice for my subscription renewal." --scores
```
Output:
```text
Predicted category: billing
Confidence score: 0.8542
Class probabilities:
  account_access: 0.0812
  billing: 0.8542
  technical_support: 0.0646
Note: Confidence scores are model estimates, not guaranteed correctness.
```

### 2. Model Persistence: Train & Save
Train the baseline pipeline on the validated dataset and serialize it with `joblib`:
```powershell
python src/model_persistence.py --dataset data/sample_dataset.jsonl --output models/baseline_model.joblib
```

### 3. Predict Using Pre-Saved Model
Predict instantly without retraining on each invocation:
```powershell
python src/predict_message.py "I cannot reset my password." --model models/baseline_model.joblib --scores
```

### 4. Validate Dataset Integrity
Checks JSON syntax, required fields, allowed labels, and duplicates:
```powershell
python src/validate_dataset.py --file data/sample_dataset.jsonl
```

### 5. Profile Dataset Distribution & Health
```powershell
python src/profile_dataset.py --file data/sample_dataset.jsonl --output results/dataset_profile.json
```

### 6. Train & Evaluate Baseline
```powershell
python src/train_baseline.py
```

### 7. Compare Baseline Against Majority-Class
```powershell
python src/compare_baselines.py --output results/baseline_comparison.json
```

### 8. Compare Logistic Regression vs. Naive Bayes
```powershell
python src/compare_models.py --output results/model_comparison.json
```

### 9. Error Analysis
```powershell
python src/analyze_errors.py --output results/error_analysis.json
```

### 10. Stratified 5-Fold Cross-Validation
```powershell
python src/evaluate_cross_validation.py --folds 5 --output results/cross_validation.json
```

### 11. Run Automated Unit Test Suite
Run the full test suite with full verbose discovery:
```powershell
python -m unittest discover -s tests -v
```

---

## ⚠️ Interpreting Results & Realistic Limitations

When developing or evaluating models with TaskTune, keep the following considerations in mind:

1. **Small illustrative dataset (75 records total):**
   - The bundled dataset contains 25 synthetic examples for each of the 3 classes (`billing`, `technical_support`, `account_access`).
   - It is intended as a **pipeline sanity check and tooling smoke test**, not as an accurate simulation of production customer support traffic.
2. **High sensitivity to split variance:**
   - In a 19-sample test split, a single misclassified message swings overall accuracy by **over 5.2 percentage points** ($1 / 19 \approx 5.26\%$). Small test set metrics should not be treated as statistically decisive.
3. **Lexical overlap vs. semantic understanding:**
   - Bag-of-words and TF-IDF representations rely on shared n-grams and vocabulary. Unfamiliar phrasing, typos, or cross-cutting language (e.g., "login failed due to an error code") can confuse classical linear models.
4. **Estimated confidence scores are not ground truth:**
   - Probability outputs reflect softmax logistic regression estimates on training features, not calibrated real-world probabilities. High confidence does not guarantee correctness.
5. **In-memory session state only:**
   - The Streamlit web app stores prediction history strictly in `st.session_state`. Logs reset upon page reload or server restart; no external databases or analytics trackers are used.
6. **Model persistence trust assumptions:**
   - Deserialization via `joblib` / `pickle` can execute arbitrary code. Only load model artifacts from trusted local paths and never deserialize untrusted files received over insecure channels.
7. **No production readiness claims:**
   - High test accuracy on a curated 75-record seed dataset does not guarantee real-world generalization. Production deployment requires expanding the dataset to thousands of verified interactions, tracking drift, measuring latency under load, and validating edge cases.

---

## 📜 License
This project is open-source under the MIT License.
