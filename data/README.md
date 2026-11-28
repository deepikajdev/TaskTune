# Data Directory

This directory stores customer support classification datasets.

## Files
- `sample_dataset.jsonl`: Curated synthetic dataset containing 300 diverse customer-inquiry examples, balanced at 100 examples per intent (`billing`, `technical_support`, `account_access`).
- `train.csv` / `val.csv` / `test.csv`: No static split files are generated. The evaluation pipeline creates stratified train/test splits in memory for each run to eliminate data leakage.

## Format Specification
Dataset files use JSON Lines (`.jsonl`), where each line is an independent JSON object:
```json
{"text": "Customer message string", "label": "billing"}
```

## Synthetic Data Disclosure & Integrity
The dataset consists of **300 synthetically generated and human-curated examples** designed for evaluating pipeline stability, training behavior, and model comparison across:
- **`billing`** (100 examples): Invoices, credit card charges, refund requests, tax forms, payment method updates.
- **`technical_support`** (100 examples): App crashes, HTTP 500/504 errors, sync failures, UI glitches, API timeouts.
- **`account_access`** (100 examples): Password recovery, MFA/2FA authenticator issues, account lockout, SSO tokens.

All records are validated using `src/validate_dataset.py` for JSON syntax, required fields, allowed labels, whitespace normalization, and duplicate detection. This dataset is synthetic and for benchmarking; it is not real customer data.
