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
├── .gitignore               # Ignores virtual envs, checkpoints, and cached files
├── README.md                # Project documentation and roadmap
├── requirements.txt         # Project dependencies with descriptive annotations
├── data/
│   ├── README.md            # Data directory guide and schema specification
│   └── sample_dataset.jsonl # 75 illustrative inquiries, balanced across 3 classes
├── notebooks/               # Exploratory analysis and training notebooks
├── results/                 # Reproducible metrics, logs, and confusion matrix plots
├── src/
│   ├── __init__.py          # Package initializer
│   ├── validate_dataset.py  # Dataset schema, label, and duplicate validator
│   ├── train_baseline.py    # TF-IDF + Logistic Regression training and evaluation
│   ├── compare_baselines.py # Majority-class versus Logistic Regression comparison
│   ├── predict_message.py   # Command-line prediction for one customer message
│   └── evaluate_cross_validation.py # Stratified cross-validation evaluation
└── tests/
    ├── test_validate_dataset.py  # Automated unit tests for data validation
    ├── test_train_baseline.py    # Automated unit tests for the baseline
    ├── test_compare_baselines.py # Tests for fair baseline comparison
    ├── test_predict_message.py   # Tests for the prediction command
    └── test_evaluate_cross_validation.py # Cross-validation tests
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Git

### 2. Install Dependencies
From the repository root, create and activate a virtual environment, then install the baseline dependency:

```bash
# Create and activate a virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Validate, Test, and Compare the Baselines
```bash
# Validate the sample dataset
python src/validate_dataset.py --file data/sample_dataset.jsonl

# Compare both baselines (default: 25% held out, seed 42)
python src/compare_baselines.py --file data/sample_dataset.jsonl

# Choose the test fraction, seed, or output JSON path
python src/compare_baselines.py --file data/sample_dataset.jsonl --test-size 0.25 --random-state 42 --output results/baseline_comparison.json

# Classify one new customer message (run from the repository root)
python src/predict_message.py "I was charged twice for my subscription."

# Use a different validated dataset
python src/predict_message.py "I cannot reset my password." --file data/sample_dataset.jsonl

# Evaluate across five stratified folds and save JSON metrics
python src/evaluate_cross_validation.py --file data/sample_dataset.jsonl

# Set the fold count, seed, or output path
python src/evaluate_cross_validation.py --file data/sample_dataset.jsonl --folds 5 --random-state 42 --output results/cross_validation.json

# Run automated unit tests
python -m unittest discover -s tests -v
```

The comparison validates the JSONL data, creates one fixed stratified split, and evaluates both models on the same held-out examples. The majority-class baseline learns its most common label from the training split only. TF-IDF is fitted on training text only, then applied to test text. Accuracy, macro-F1, per-class precision/recall/F1, and confusion matrices use a consistent label order. Actual metrics and reproducibility metadata (including the split indices, seed, versions, and dataset SHA-256) are saved to `results/baseline_comparison.json`. The bundled dataset is illustrative rather than representative; its results are not a meaningful performance benchmark.

The prediction command validates and trains the same TF-IDF + Logistic Regression pipeline on the labeled dataset, then predicts one supplied message. Prediction uses the labeled examples for fitting; the separate comparison command remains responsible for held-out evaluation. A trained model is not saved to disk. The CLI reports an error for empty messages or missing/invalid datasets.

Cross-validation divides the dataset into several stratified folds. Each fold trains on all but one part and evaluates on the held-out part; every example is held out once. The TF-IDF vectorizer is inside the training pipeline, so each fold learns its vocabulary from that fold's training examples only. The report includes fold-level accuracy, macro-F1, per-class precision/recall/F1, and the mean and population standard deviation across folds. The five-fold default requires at least five examples in every class. Cross-validation on this small illustrative dataset is only an initial estimate, not proof of real-world performance.

---

## 📜 License
This project is open-source under the MIT License.
