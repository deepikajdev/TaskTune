# Results & Evaluation Artifacts

Run `python src/compare_baselines.py` from the repository root to compare the majority-class predictor with TF-IDF + Logistic Regression. The script prints accuracy, macro-F1, per-class precision/recall/F1, and labeled confusion matrices, and writes the actual metrics and reproducibility metadata to `baseline_comparison.json`. The JSON includes the fixed stratified split indices, random seed, package versions, label ordering, and dataset SHA-256.

Run `python src/evaluate_cross_validation.py` to evaluate the TF-IDF + Logistic Regression pipeline with stratified cross-validation. This writes fold metrics and across-fold means and population standard deviations to `cross_validation.json`, without overwriting the comparison results.

The sample contains 75 illustrative messages, not a representative customer dataset. Treat its held-out metrics as a pipeline check, not as evidence of generalization or production performance.
