"""TaskTune Streamlit web application.

Provides a simple, clean interface for predicting customer inquiry categories
using the trained TF-IDF + Logistic Regression classifier.

Usage:
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

# Allow imports from src/ whether running from repo root or via streamlit
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict_message import predict_with_confidence
from src.validate_dataset import ALLOWED_LABELS
from src.input_validation import MAX_MESSAGE_LENGTH, friendly_validation_error, validate_message
from src.prediction_history import add_history_record, clear_history
from src.report_loader import get_baseline_metrics, get_cross_validation_metrics

BASELINE_REPORT_PATH = PROJECT_ROOT / "results" / "baseline_comparison.json"
CV_REPORT_PATH = PROJECT_ROOT / "results" / "cross_validation.json"

DATASET_PATH = PROJECT_ROOT / "data" / "sample_dataset.jsonl"
LABEL_EMOJI = {
    "billing": "💳",
    "technical_support": "🔧",
    "account_access": "🔑",
}

# ── Session State for Prediction History ──────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TaskTune — Message Classifier",
    page_icon="🎯",
    layout="centered",
)

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🎯 TaskTune")
st.caption("Customer inquiry classifier · TF-IDF + Logistic Regression baseline")
st.divider()

# ── Input section ─────────────────────────────────────────────────────────────
st.subheader("📝 Classify a message")

message = st.text_area(
    "Enter a customer message",
    placeholder="e.g. I was charged twice for my subscription this month.",
    max_chars=MAX_MESSAGE_LENGTH,
    height=120,
    help=f"Maximum {MAX_INPUT_LENGTH} characters. Must not be blank.",
)

predict_clicked = st.button("🔍 Predict", type="primary", use_container_width=True)

# ── Validation & Prediction ───────────────────────────────────────────────────
if predict_clicked:
    try:
        stripped = validate_message(message)
    except (TypeError, ValueError) as validation_exc:
        st.error(friendly_validation_error(validation_exc))
        stripped = None

    if stripped:
        with st.spinner("Classifying…"):
            try:
                result = predict_with_confidence(stripped, DATASET_PATH)
            except FileNotFoundError:
                st.error("❌ Training dataset not found. Run the app from the repository root.")
                result = None
            except ValueError as exc:
                st.error(f"❌ Prediction error: {exc}")
                result = None

        if result:
            label = result["predicted_label"]
            emoji = LABEL_EMOJI.get(label, "📌")
            confidence = result["confidence"]

            # Save to in-memory session state history
            add_history_record(st.session_state.history, stripped, label, confidence)

            st.divider()
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"### {emoji} `{label}`")
                st.caption("Predicted category")
            with col2:
                st.metric("Confidence", f"{confidence:.1%}")

            st.subheader("📊 Class probabilities")
            for cls in sorted(result["probabilities"].keys()):
                prob = result["probabilities"][cls]
                bar_emoji = LABEL_EMOJI.get(cls, "📌")
                st.progress(prob, text=f"{bar_emoji} {cls}: {prob:.1%}")

            st.caption(
                "⚠️ Confidence scores are model estimates, not guaranteed correctness. "
                "This model is trained on 75 illustrative examples — results may not "
                "reflect real-world performance."
            )

# ── Model Metrics Dashboard Section ──────────────────────────────────────────
st.divider()
st.subheader("📊 Offline Evaluation Metrics")
st.caption(
    "Metrics loaded from saved benchmark reports (`results/baseline_comparison.json` & `results/cross_validation.json`). "
    "These reflect offline evaluation on the 75-example sample dataset, not live production telemetry."
)

baseline = get_baseline_metrics(BASELINE_REPORT_PATH)
if baseline:
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric("Test Accuracy", f"{baseline['accuracy']:.1%}")
    with m_col2:
        st.metric("Macro F1", f"{baseline['macro_f1']:.3f}")
    with m_col3:
        st.metric("Dataset Split", f"{baseline['train_examples']} train / {baseline['test_examples']} test")

    if baseline.get("per_class"):
        st.markdown("**Per-Class Breakdown (Held-Out Test Set):**")
        class_table = []
        for cls_name, vals in sorted(baseline["per_class"].items()):
            class_table.append({
                "Category": f"{LABEL_EMOJI.get(cls_name, '📌')} {cls_name}",
                "Precision": f"{vals.get('precision', 0.0):.1%}",
                "Recall": f"{vals.get('recall', 0.0):.1%}",
                "F1 Score": f"{vals.get('f1', 0.0):.3f}",
                "Support": vals.get("support", 0),
            })
        st.table(class_table)
else:
    st.warning("⚠️ Baseline evaluation report not found. Run `python src/compare_baselines.py` to generate it.")

cv = get_cross_validation_metrics(CV_REPORT_PATH)
if cv:
    with st.expander("📈 5-Fold Cross-Validation Summary"):
        cv_col1, cv_col2 = st.columns(2)
        with cv_col1:
            st.metric("Mean CV Accuracy", f"{cv['mean_accuracy']:.1%}", delta=f"±{cv['std_accuracy']:.1%} std")
        with cv_col2:
            st.metric("Mean CV Macro F1", f"{cv['mean_macro_f1']:.3f}", delta=f"±{cv['std_macro_f1']:.3f} std")
        st.caption("Fitted strictly within each fold's training split to eliminate data leakage.")
elif not baseline:
    st.info("Cross-validation report not found. Run `python src/evaluate_cross_validation.py` to generate it.")

# ── Prediction History Section ───────────────────────────────────────────────
st.divider()
history_header_col, clear_btn_col = st.columns([3, 1])
with history_header_col:
    st.subheader("📜 Session Prediction History")
with clear_btn_col:
    if st.session_state.history:
        if st.button("🗑️ Clear History", key="clear_history_btn"):
            st.session_state.history = clear_history()
            st.rerun()

if st.session_state.history:
    for item in reversed(st.session_state.history):
        item_emoji = LABEL_EMOJI.get(item["predicted_label"], "📌")
        with st.container(border=True):
            col_meta, col_lbl = st.columns([3, 1])
            with col_meta:
                st.markdown(f"**#{item['sequence']}** · `{item['timestamp']}`")
                st.write(item["message"])
            with col_lbl:
                st.markdown(f"### {item_emoji} `{item['predicted_label']}`")
                st.caption(f"Confidence: {item['confidence']:.1%}")
else:
    st.info("No predictions recorded in this session yet. Enter a message above and click Predict!")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "TaskTune · Supported labels: "
    + " · ".join(f"{LABEL_EMOJI.get(l, '')} {l}" for l in sorted(ALLOWED_LABELS))
)
