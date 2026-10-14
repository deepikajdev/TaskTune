# Data Directory

This directory stores customer support classification datasets.

## Files
- `sample_dataset.jsonl`: Small, illustrative dataset containing 75 manually written customer-inquiry examples, balanced at 25 examples per intent (`billing`, `technical_support`, `account_access`).
- `train.csv` / `val.csv` / `test.csv`: No split files are generated. The baseline creates a stratified train/test split in memory for each run.

## Format Specification
Dataset files use JSON Lines (`.jsonl`), where each line is an independent JSON object:
```json
{"text": "Customer message string", "label": "billing"}
```
The sample is for learning and exercising the pipeline; it is not a representative sample of real customer inquiries and should not support claims about production performance. The validator checks JSON syntax, object records, required text and label fields, field types, non-empty text and labels, allowed categories, and duplicate messages (case-insensitive with normalized whitespace). Blank lines are reported as warnings. Dataset records are validated before modeling in `src/train_baseline.py`.
