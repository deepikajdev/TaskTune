# TaskTune: Small Language Model Fine-Tuning & Evaluation

TaskTune is a staged project for evaluating customer-support text classification models. Its first modeling experiment is a lightweight classical machine-learning baseline:

1. **Classical baseline comparison:** Majority-class prediction versus TF-IDF + Logistic Regression, evaluated on the same held-out, stratified test split.
2. **Future experiments:** Prompted and fine-tuned small language models may be explored in later milestones.

---

## 📌 Project Status

> **Important:** A first baseline experiment is implemented. The sample dataset has 75 illustrative examples (25 per class); it is not representative of real customers, and its metrics are a pipeline smoke test only—not a reliable estimate of real-world or production performance.

---

## 🎯 The Core Problem & Motivation

Enterprise applications often default to massive closed-source APIs for simple text classification. However, large models introduce high recurring costs, privacy concerns, latency overhead, and API rate limits.

**Small Language Models (SLMs)** (e.g., under 3 billion parameters) may offer an alternative. TaskTune will explore model trade-offs over time; the tiny seed dataset is not sufficient to establish production-level performance.

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

The bundled sample contains 25 examples for each of `billing`, `technical_support`, and `account_access`. These hand-written examples help exercise the validator and baseline; they are not a representative customer sample. Validation checks JSON syntax, required fields and their types, non-empty text and labels, allowed label values, and duplicate messages.

---

## 🧠 Essential Concepts (Plain English)

- **SLM (Small Language Model):** A language model designed to have fewer parameters (typically 1B to 3B parameters) than frontier models (70B+ parameters). SLMs are fast, lightweight, and feasible to run and fine-tune on free or low-cost compute (like Google Colab T4).
- **Prompt Engineering:** Teaching a model what to do solely through the input text prompt (system instructions and few-shot examples) without changing any underlying model weights.
- **Fine-Tuning:** Updating the model's internal weights using labeled examples so it specializes in a specific task.
- **LoRA (Low-Rank Adaptation):** Instead of updating all billions of model weights, LoRA freezes the original model and injects tiny trainable adapter matrices into attention layers. This drastically reduces GPU memory and training time while preserving performance.
- **QLoRA (Quantized LoRA):** Compresses the frozen base model weights into 4-bit precision before attaching LoRA adapters, allowing modern models to be fine-tuned on single consumer GPUs or free Colab instances.

---

## 📊 Evaluation Methodology

To ensure scientific rigor and eliminate bias, all model configurations will be evaluated against the **exact same held-out test split**:

| Metric | Why It Matters |
| :--- | :--- |
| **Accuracy** | Overall proportion of correct predictions across all categories. |
| **Macro F1-Score** | Balances precision and recall evenly across classes, penalizing models that ignore underrepresented categories. |
| **Precision & Recall** | Measures false positives (e.g. routing a billing issue to technical support) vs false negatives. |
| **Confusion Matrix** | Visualizes which specific categories the models confuse with each other. |
| **Inference Latency** | Time taken per prediction (critical for real-time customer support routing). |

---

## 🗺️ Build Roadmap

- [x] **Milestone 1:** Initialize repository, beginner-friendly architecture, and documentation.
- [x] **Milestone 2:** Dataset curation and validation pipeline (`validate_dataset.py`, automated tests).
- [x] **Milestone 3:** Stratified train/test split and classical baseline classifier (TF-IDF + Logistic Regression).
- [ ] **Milestone 4:** Model selection & feasibility check (selecting an open SLM with permissive license and free GPU compatibility).
- [ ] **Milestone 5:** Zero-shot and few-shot prompted model evaluation on the test set.
- [ ] **Milestone 6:** LoRA / QLoRA fine-tuning implementation with validation tracking.
- [ ] **Milestone 7:** Final comparative benchmark report (Base vs Prompted vs LoRA vs Baseline), confusion matrices, and latency comparison.
- [ ] **Milestone 8:** Artifact publishing (adapter weights and model card on Hugging Face).

---

## 📁 Project Structure

```text
tasktune/
├── .gitignore                   # Ignores virtual envs, checkpoints, and cached files
├── README.md                    # Project documentation, guides, and roadmap
├── requirements.txt             # Project dependencies with descriptive annotations
├── data/
│   ├── README.md                # Data directory guide and schema specification
│   └── sample_dataset.jsonl     # 75 illustrative inquiries, balanced across 3 classes
├── notebooks/                   # Exploratory analysis and training notebooks
├── results/                     # Reproducible metrics, logs, and diagnostic artifacts
│   ├── README.md                # Guide to evaluation JSON artifacts
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
│   ├── predict_message.py       # Command-line prediction for one customer message
│   └── evaluate_cross_validation.py # Stratified cross-validation evaluation
└── tests/
    ├── test_validate_dataset.py # Automated unit tests for data validation
    ├── test_profile_dataset.py  # Tests for dataset profiling and statistics
    ├── test_train_baseline.py   # Automated unit tests for the baseline
    ├── test_compare_baselines.py # Tests for majority vs baseline comparison
    ├── test_compare_models.py   # Tests for Logistic Regression vs Naive Bayes
    ├── test_analyze_errors.py   # Tests for classification error analysis
    ├── test_predict_message.py  # Tests for the prediction CLI command
    └── test_evaluate_cross_validation.py # Cross-validation tests
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Git

### 2. Install Dependencies
From the repository root, create and activate a virtual environment, then install project dependencies:

```bash
# Create and activate a virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🛠️ CLI Usage & Tooling Guide

### 1. Validate Dataset Integrity
Checks JSON syntax, required fields (`text`, `label`), allowed labels, non-empty text, and duplicates:
```bash
python src/validate_dataset.py --file data/sample_dataset.jsonl
```

### 2. Profile Dataset Distribution & Health
Analyzes class distribution percentages, duplicate records, blank/invalid lines, and character/word length statistics:
```bash
python src/profile_dataset.py --file data/sample_dataset.jsonl --output results/dataset_profile.json
```

### 3. Train & Evaluate the Baseline Model
Fits TF-IDF and Logistic Regression on a stratified training split and prints test metrics with a confusion matrix:
```bash
python src/train_baseline.py
```

### 4. Compare Baseline against Majority-Class Predictor
Compares majority-class guessing against TF-IDF + Logistic Regression on the exact same held-out test split:
```bash
python src/compare_baselines.py --file data/sample_dataset.jsonl --test-size 0.25 --random-state 42 --output results/baseline_comparison.json
```

### 5. Fair Comparison Between Two Classification Models
Fairly compares Logistic Regression against Multinomial Naive Bayes using identical stratified splits, identical vectorizer fitting, and identical evaluation metrics:
```bash
python src/compare_models.py --file data/sample_dataset.jsonl --test-size 0.25 --random-state 42 --output results/model_comparison.json
```

### 6. Error Analysis Report
Diagnoses classifier mistakes on the held-out test set, identifying misclassified examples and error counts per category:
```bash
python src/analyze_errors.py --file data/sample_dataset.jsonl --test-size 0.25 --random-state 42 --output results/error_analysis.json
```

### 7. Stratified Cross-Validation
Evaluates stability across 5 stratified folds with TF-IDF fitted inside each fold to eliminate data leakage:
```bash
python src/evaluate_cross_validation.py --file data/sample_dataset.jsonl --folds 5 --random-state 42 --output results/cross_validation.json
```

### 8. Predict Category for a Single Customer Message
Quickly routes a single inbound message using the trained pipeline:
```bash
python src/predict_message.py "I was charged twice for my subscription renewal."
```

### 9. Run the Automated Unit Test Suite
Runs all unit tests with full discovery:
```bash
python -m unittest discover -s tests -v
```

---

## 🔬 Reproducing Experiments Step-by-Step

All experiments in TaskTune are deterministic and can be reproduced with these exact commands from the repository root:

1. **Profile the sample dataset:**
   ```bash
   python src/profile_dataset.py --file data/sample_dataset.jsonl --output results/dataset_profile.json
   ```
   *Generates:* `results/dataset_profile.json` documenting 75 total records, zero duplicates, and exact 33.33% balance across the 3 classes.

2. **Evaluate the baseline vs. majority class:**
   ```bash
   python src/compare_baselines.py --file data/sample_dataset.jsonl --test-size 0.25 --random-state 42 --output results/baseline_comparison.json
   ```
   *Generates:* `results/baseline_comparison.json` with a 56-train / 19-test stratified split.

3. **Compare Logistic Regression vs. Multinomial Naive Bayes:**
   ```bash
   python src/compare_models.py --file data/sample_dataset.jsonl --test-size 0.25 --random-state 42 --output results/model_comparison.json
   ```
   *Generates:* `results/model_comparison.json` comparing both classifiers on identical test indices.

4. **Run error analysis:**
   ```bash
   python src/analyze_errors.py --file data/sample_dataset.jsonl --test-size 0.25 --random-state 42 --output results/error_analysis.json
   ```
   *Generates:* `results/error_analysis.json` identifying the 4 misclassified test examples and per-class error distribution.

5. **Run stratified cross-validation:**
   ```bash
   python src/evaluate_cross_validation.py --file data/sample_dataset.jsonl --folds 5 --random-state 42 --output results/cross_validation.json
   ```
   *Generates:* `results/cross_validation.json` containing 5 fold results and summary mean/standard deviation.

---

## ⚠️ Interpreting Results & Realistic Limitations

When reading the metrics in `results/`, keep the following real-world considerations in mind:

1. **Small illustrative dataset (75 records total):**
   - The bundled dataset contains 25 synthetic examples for each of the 3 classes (`billing`, `technical_support`, `account_access`).
   - It is intended as a **pipeline sanity check and tooling smoke test**, not as an accurate simulation of production customer support traffic.
2. **High sensitivity to split variance:**
   - In a 19-sample test split, a single misclassified message swings the overall accuracy score by **over 5.2 percentage points** ($1 / 19 \approx 5.26\%$). Small test set metrics should not be treated as statistically decisive.
3. **Lexical overlap vs. semantic understanding:**
   - Bag-of-words and TF-IDF representations rely on shared n-grams and vocabulary. When a customer uses unfamiliar phrasing, typos, or cross-cutting language (e.g., "login failed due to an error code"), classical linear models can easily become confused.
4. **No production readiness claims:**
   - A high score on this curated seed dataset does not guarantee real-world generalization. Production deployment requires expanding the dataset to thousands of verified customer interactions, tracking drift, measuring latency under load, and validating edge cases.

---

## 📜 License
This project is open-source under the MIT License.
