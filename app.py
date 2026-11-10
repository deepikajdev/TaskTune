"""TaskTune Streamlit web application.

Provides a clean, interactive dashboard for predicting customer inquiry categories
using the trained TF-IDF + Logistic Regression classifier and inspecting
saved evaluation benchmark reports.

Usage:
    streamlit run app.py
"""

from pathlib import Path
import sys

import streamlit as st

# Allow imports from src/ whether running from repo root or via streamlit
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.input_validation import MAX_MESSAGE_LENGTH, friendly_validation_error, validate_message
from src.predict_message import predict_with_confidence
from src.prediction_history import add_history_record, clear_history
from src.report_loader import get_baseline_metrics, get_cross_validation_metrics
from src.ui_helpers import (
    SAMPLE_QUERIES,
    get_confidence_rating,
    get_label_emoji,
    truncate_text,
)
from src.validate_dataset import ALLOWED_LABELS

DATASET_PATH = PROJECT_ROOT / "data" / "sample_dataset.jsonl"
BASELINE_REPORT_PATH = PROJECT_ROOT / "results" / "baseline_comparison.json"
CV_REPORT_PATH = PROJECT_ROOT / "results" / "cross_validation.json"

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TaskTune · Customer Inquiry Classifier",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session State Initialization ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

if "input_text" not in st.session_state:
    st.session_state.input_text = ""


def _set_sample_query(text: str) -> None:
    st.session_state.input_text = text


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎯 TaskTune")
    st.caption("Classical ML baseline for inquiry routing")
    st.markdown("---")

    st.markdown("### 🏷️ Target Categories")
    for label in sorted(ALLOWED_LABELS):
        emoji = get_label_emoji(label)
        st.markdown(f"- **{emoji} `{label}`**")

    st.markdown("---")
    st.markdown("### ⚙️ Baseline Architecture")
    st.markdown(
        "- **Vectorization:** TF-IDF (word n-grams)\n"
        "- **Classifier:** Logistic Regression\n"
        "- **Training seed:** 42 (deterministic)\n"
        "- **Dataset:** 75 illustrative inquiries"
    )

    st.markdown("---")
    st.caption(
        "TaskTune is an open-source evaluation framework. "
        "Baseline results are pipeline sanity checks, not production guarantees."
    )

# ── Main Header ───────────────────────────────────────────────────────────────
st.title("🎯 TaskTune Customer Inquiry Classifier")
st.markdown(
    "Route customer support inquiries into **Billing**, **Technical Support**, "
    "or **Account Access** with confidence estimates and benchmark diagnostics."
)

# ── Tab Navigation ────────────────────────────────────────────────────────────
tab_predict, tab_metrics, tab_history = st.tabs([
    "💬 Live Classifier",
    "📊 Evaluation Metrics",
    f"📜 Session History ({len(st.session_state.history)})",
])

# ── TAB 1: Live Classifier ────────────────────────────────────────────────────
with tab_predict:
    st.subheader("Classify a Customer Message")
    st.caption("Select an illustrative sample or write your own inquiry below.")

    # Quick sample prompt chips
    st.markdown("**Try a sample inquiry:**")
    chip_cols = st.columns(len(SAMPLE_QUERIES))
    for col, (chip_label, sample_text) in zip(chip_cols, SAMPLE_QUERIES.items()):
        with col:
            if st.button(chip_label, use_container_width=True, key=f"chip_{chip_label}"):
                _set_sample_query(sample_text)

    # Input form
    message_input = st.text_area(
        "Customer message text:",
        value=st.session_state.input_text,
        placeholder="e.g. My monthly invoice was billed twice and I would like a refund.",
        max_chars=MAX_MESSAGE_LENGTH,
        height=130,
        help=f"Enter up to {MAX_MESSAGE_LENGTH} characters. Blank or whitespace-only inputs are rejected.",
    )

    action_col1, action_col2 = st.columns([1, 4])
    with action_col1:
        predict_btn = st.button("🚀 Predict Intent", type="primary", use_container_width=True)
    with action_col2:
        if st.button("🧹 Clear Input", key="clear_input_btn"):
            st.session_state.input_text = ""
            st.rerun()

    if predict_btn:
        try:
            cleaned_message = validate_message(message_input)
        except (TypeError, ValueError) as val_exc:
            st.error(friendly_validation_error(val_exc))
            cleaned_message = None

        if cleaned_message:
            with st.spinner("Analyzing text and estimating class probabilities..."):
                try:
                    result = predict_with_confidence(cleaned_message, DATASET_PATH)
                except FileNotFoundError:
                    st.error("❌ Training dataset file not found. Ensure `data/sample_dataset.jsonl` exists.")
                    result = None
                except ValueError as err:
                    st.error(f"❌ Prediction failed: {err}")
                    result = None

            if result:
                label = result["predicted_label"]
                conf = result["confidence"]
                rating_text, rating_icon = get_confidence_rating(conf)
                emoji = get_label_emoji(label)

                # Append to session history
                add_history_record(st.session_state.history, cleaned_message, label, conf)

                st.success("Classification complete!")
                res_box = st.container(border=True)
                with res_box:
                    rc1, rc2, rc3 = st.columns([2, 1, 1])
                    with rc1:
                        st.markdown(f"### {emoji} `{label}`")
                        st.caption("Predicted Intent Category")
                    with rc2:
                        st.metric("Estimated Confidence", f"{conf:.1%}")
                    with rc3:
                        st.metric("Confidence Rating", f"{rating_icon} {rating_text}")

                    st.markdown("#### Class Probability Distribution")
                    for cls_name in sorted(result["probabilities"].keys()):
                        prob = result["probabilities"][cls_name]
                        cls_emoji = get_label_emoji(cls_name)
                        st.progress(prob, text=f"{cls_emoji} {cls_name}: {prob:.1%}")

                    st.caption(
                        "ℹ️ **Confidence disclaimer:** Scores are model estimates from softmax "
                        "logistic regression, not guaranteed real-world correctness. "
                        "The model is trained on a 75-example illustrative seed dataset."
                    )

# ── TAB 2: Evaluation Metrics ─────────────────────────────────────────────────
with tab_metrics:
    st.subheader("Offline Evaluation & Benchmark Performance")
    st.caption(
        "These performance metrics are parsed directly from reproducible evaluation reports "
        "stored in `results/`. They represent controlled offline benchmarks, not live production telemetry."
    )

    baseline_data = get_baseline_metrics(BASELINE_REPORT_PATH)
    if baseline_data:
        b_col1, b_col2, b_col3 = st.columns(3)
        with b_col1:
            st.metric("Held-Out Accuracy", f"{baseline_data['accuracy']:.1%}")
        with b_col2:
            st.metric("Macro F1-Score", f"{baseline_data['macro_f1']:.3f}")
        with b_col3:
            st.metric(
                "Stratified Split",
                f"{baseline_data['train_examples']} train / {baseline_data['test_examples']} test",
            )

        if baseline_data.get("per_class"):
            st.markdown("### Per-Class Test Set Performance")
            metric_rows = []
            for c_name, scores in sorted(baseline_data["per_class"].items()):
                c_emoji = get_label_emoji(c_name)
                metric_rows.append({
                    "Category": f"{c_emoji} {c_name}",
                    "Precision": f"{scores.get('precision', 0.0):.1%}",
                    "Recall": f"{scores.get('recall', 0.0):.1%}",
                    "F1-Score": f"{scores.get('f1', 0.0):.3f}",
                    "Support": scores.get("support", 0),
                })
            st.dataframe(metric_rows, use_container_width=True, hide_index=True)

        st.caption(f"Report source: `results/baseline_comparison.json` · Generated: {baseline_data.get('created_at')}")
    else:
        st.warning(
            "⚠️ Baseline comparison report not found at `results/baseline_comparison.json`. "
            "Run `python src/compare_baselines.py` to produce this artifact."
        )

    st.markdown("---")
    cv_data = get_cross_validation_metrics(CV_REPORT_PATH)
    if cv_data:
        st.subheader("Stratified 5-Fold Cross-Validation")
        st.caption("Vectorization and model training evaluated across 5 stratified folds to prevent data leakage.")
        cv_c1, cv_c2 = st.columns(2)
        with cv_c1:
            st.metric("Mean CV Accuracy", f"{cv_data['mean_accuracy']:.1%}", delta=f"±{cv_data['std_accuracy']:.1%} std")
        with cv_c2:
            st.metric("Mean CV Macro F1", f"{cv_data['mean_macro_f1']:.3f}", delta=f"±{cv_data['std_macro_f1']:.3f} std")
    else:
        st.info("Cross-validation artifact not found at `results/cross_validation.json`.")

# ── TAB 3: Prediction History ─────────────────────────────────────────────────
with tab_history:
    hist_top_col1, hist_top_col2 = st.columns([3, 1])
    with hist_top_col1:
        st.subheader("Session Query Log")
        st.caption("Kept in in-memory session state only. No data is stored externally.")
    with hist_top_col2:
        if st.session_state.history:
            if st.button("🗑️ Clear History", key="tab_clear_hist_btn", use_container_width=True):
                st.session_state.history = clear_history()
                st.rerun()

    if not st.session_state.history:
        st.info(
            "💡 No queries recorded in this session yet. "
            "Switch to the **Live Classifier** tab, enter an inquiry, and click Predict to start logging."
        )
    else:
        for entry in reversed(st.session_state.history):
            e_emoji = get_label_emoji(entry["predicted_label"])
            with st.container(border=True):
                hc1, hc2 = st.columns([3, 1])
                with hc1:
                    st.markdown(f"**Query #{entry['sequence']}** · `{entry['timestamp']}`")
                    st.write(entry["message"])
                with hc2:
                    st.markdown(f"### {e_emoji} `{entry['predicted_label']}`")
                    st.caption(f"Confidence: {entry['confidence']:.1%}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "TaskTune · Staged evaluation framework for customer-support text classification · "
    + " · ".join(f"{get_label_emoji(l)} {l}" for l in sorted(ALLOWED_LABELS))
)
