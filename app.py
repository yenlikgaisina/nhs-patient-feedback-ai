"""Patient Feedback Intelligence App — public demo (portfolio prototype)."""
import pandas as pd
import streamlit as st

from predict import CONFIDENCE_THRESHOLD, predict_batch, predict_feedback

# ---------------------------------------------------------------------------
# Theme display labels
# ---------------------------------------------------------------------------

THEME_LABELS = {
    "appointment_access": "Appointment Access",
    "communication": "Communication",
    "staff_attitude": "Staff Attitude",
    "waiting_time": "Waiting Time",
    "treatment_quality": "Treatment Quality",
    "administration": "Administration",
    "facilities": "Facilities",
    "other": "Other",
}


def pretty(theme: str) -> str:
    return THEME_LABELS.get(theme, theme)


# ---------------------------------------------------------------------------
# Page configuration — NHS-inspired palette, no official NHS logo
# ---------------------------------------------------------------------------

NHS_BLUE = "#005EB8"
NHS_DARK_BLUE = "#003087"
NHS_LIGHT_BG = "#F0F4F5"

st.set_page_config(
    page_title="Patient Feedback Intelligence App",
    page_icon="💬",
    layout="centered",
)

# Inject NHS-inspired colour styling — no official NHS logo or identity used
st.markdown(
    f"""
    <style>
        [data-testid="stHeader"] {{
            background-color: {NHS_BLUE};
        }}
        .stButton > button[kind="primary"] {{
            background-color: {NHS_BLUE};
            border-color: {NHS_DARK_BLUE};
            color: #ffffff;
        }}
        .stButton > button[kind="primary"]:hover {{
            background-color: {NHS_DARK_BLUE};
            border-color: {NHS_DARK_BLUE};
        }}
        [data-testid="stSidebar"] {{
            background-color: {NHS_LIGHT_BG};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — demo safety notes
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        f"<h2 style='color:{NHS_BLUE};'>Patient Feedback Intelligence App</h2>",
        unsafe_allow_html=True,
    )
    st.caption("NHS-style service improvement prototype")

    st.divider()

    st.subheader("⚠️ Demo safety notes")
    st.markdown(
        "- **Do not enter** names, NHS numbers, phone numbers, addresses, "
        "or medical record details.\n"
        "- **Not for diagnosis**, urgent care, or clinical decisions.\n"
        "- Outputs are for **service-improvement triage only**.\n"
        "- **Low-confidence outputs** require human review before any "
        "operational action.\n"
        "- Use **fictional or anonymised** comments only."
    )

    st.divider()

    st.subheader("About")
    st.markdown(
        "An end-to-end NLP portfolio project: TF-IDF + Logistic Regression "
        "trained on synthetic NHS-style data, served with Streamlit.\n\n"
        "**Skills:** Python · scikit-learn · NLP · Streamlit · MLflow · "
        "pytest · GitHub Actions · responsible AI"
    )

    st.divider()

    st.caption(
        "**Independent portfolio prototype.** "
        "Not affiliated with, endorsed by, or approved by the NHS."
    )

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------

st.markdown(
    f"<h1 style='color:{NHS_BLUE};'>Patient Feedback Intelligence App</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='font-size:1.1rem; color:#333;'>"
    "NHS-style service improvement prototype for classifying patient feedback "
    "into operational themes."
    "</p>",
    unsafe_allow_html=True,
)

# Top-level disclaimer banner
st.warning(
    "⚠️ **Independent portfolio prototype.** Not affiliated with, endorsed by, "
    "or approved by the NHS. **Do not enter personal, identifiable, urgent, or "
    "clinical information.** Use fictional or anonymised comments only."
)

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_single, tab_batch, tab_about = st.tabs(
    ["💬 Single comment", "📋 Batch CSV", "🔍 Model & responsible AI"]
)

# ---------------------------------------------------------------------------
# Tab 1 — Single comment analysis
# ---------------------------------------------------------------------------

with tab_single:
    st.markdown(
        "<p style='color:#555;'>Paste a fictional or anonymised patient comment "
        "to classify it into a service-improvement theme.</p>",
        unsafe_allow_html=True,
    )

    comment = st.text_area(
        "Patient comment:",
        height=180,
        placeholder=(
            "Use fictional or anonymised comments only — e.g. "
            "'I waited weeks for an appointment and nobody explained the delay.'"
        ),
    )

    st.caption(
        "🔒 Privacy reminder: do not enter names, NHS numbers, contact details, "
        "medical record details, or urgent clinical information."
    )

    if st.button("Analyse feedback", type="primary"):
        if not comment.strip():
            st.warning("Please paste a comment first.")
        else:
            result = predict_feedback(comment)

            col1, col2, col3 = st.columns(3)
            col1.metric("Theme", pretty(result["theme"]))
            col2.metric("Sentiment", result["sentiment"])
            if result["confidence"] is not None:
                col3.metric("Confidence", f"{result['confidence']:.0%}")

            if result["needs_review"]:
                st.warning(
                    f"⚠️ **Needs human review** — confidence below "
                    f"{CONFIDENCE_THRESHOLD:.0%} threshold."
                )
            else:
                st.success("✅ Confidence above review threshold.")

            st.subheader("Top 3 themes")
            top3_df = pd.DataFrame(result["top_3"])
            top3_df["theme"] = top3_df["theme"].map(pretty)
            st.bar_chart(top3_df.set_index("theme"))

            st.subheader("Suggested operational action")
            st.info(result["suggested_action"])

# ---------------------------------------------------------------------------
# Tab 2 — Batch CSV upload
# ---------------------------------------------------------------------------

with tab_batch:
    st.markdown(
        "<p style='color:#555;'>Upload a CSV of fictional or anonymised patient "
        "comments to classify in bulk. The file must contain a column with the "
        "comment text.</p>",
        unsafe_allow_html=True,
    )

    st.caption(
        "🔒 Privacy reminder: ensure all comments are fictional or fully "
        "anonymised before uploading. Do not upload identifiable patient data."
    )

    uploaded = st.file_uploader("CSV file", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read CSV: {exc}")
            df = None

        if df is not None and len(df) > 0:
            st.write(f"Loaded **{len(df):,}** rows.")
            st.dataframe(df.head(5))

            text_col = st.selectbox(
                "Which column contains the patient comment?",
                options=list(df.columns),
            )

            if st.button("Run batch predictions", type="primary"):
                with st.spinner(f"Classifying {len(df):,} comments..."):
                    texts = df[text_col].fillna("").astype(str).tolist()
                    results = predict_batch(texts)

                out = df.copy()
                out["predicted_theme"] = [r["theme"] for r in results]
                out["predicted_theme_label"] = [
                    pretty(r["theme"]) for r in results
                ]
                out["sentiment"] = [r["sentiment"] for r in results]
                out["confidence"] = [r["confidence"] for r in results]
                out["needs_review"] = [r["needs_review"] for r in results]
                out["suggested_action"] = [
                    r["suggested_action"] for r in results
                ]

                st.subheader("Results")
                st.dataframe(out)

                st.subheader("Theme distribution")
                st.bar_chart(out["predicted_theme_label"].value_counts())

                review_count = int(out["needs_review"].sum())
                st.metric(
                    "Rows flagged for human review",
                    f"{review_count} / {len(out)}",
                )

                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download results as CSV",
                    data=csv_bytes,
                    file_name="feedback_predictions.csv",
                    mime="text/csv",
                )
        elif df is not None:
            st.warning("CSV is empty.")

# ---------------------------------------------------------------------------
# Tab 3 — Model & responsible AI
# ---------------------------------------------------------------------------

with tab_about:
    st.subheader("Model")
    st.markdown(
        "- **Algorithm:** TF-IDF (unigrams + bigrams, max 10k features) "
        "+ Logistic Regression with balanced class weights\n"
        "- **Themes:** appointment access, communication, staff attitude, "
        "waiting time, treatment quality, administration, facilities\n"
        "- **Sentiment:** TextBlob polarity (positive / negative / neutral)\n"
        f"- **Review threshold:** predictions below {CONFIDENCE_THRESHOLD:.0%} "
        "confidence are flagged for human review\n"
        "- **Training data:** synthetic NHS-style dataset generated for "
        "reproducible portfolio deployment"
    )

    st.subheader("Responsible AI")
    st.markdown(
        "- **Synthetic development data.** The included dataset is a synthetic, "
        "NHS-style dataset. The reported 1.00 accuracy reflects memorisation of "
        "a small fixed phrase pool — not a real-world performance estimate. "
        "Expected accuracy on real NHS data: ~0.65–0.75.\n"
        "- **Weak labels.** Training labels are rule-based starter labels, not "
        "expert-annotated clinical categories.\n"
        "- **Not a clinical tool.** Do not use for diagnosis, patient safety "
        "decisions, or automated complaint handling without human review.\n"
        "- **Human review required.** Low-confidence predictions are flagged "
        "and should be triaged by a person before any operational action.\n"
        "- **No demographic fairness testing yet** — performance may differ "
        "across patient groups.\n"
        "- **Independent portfolio prototype.** Not affiliated with, endorsed "
        "by, or approved by the NHS."
    )

    st.subheader("Clinical & data safety")
    st.markdown(
        "This prototype has **not** undergone NHS DTAC assessment, DCB0129 "
        "clinical safety review, or Data Security and Protection Toolkit "
        "evaluation. It is not suitable for live NHS deployment without "
        "completing those governance processes."
    )

    st.caption(
        "Full model card: `reports/model_card.md` · "
        "Risk assessment: `reports/responsible_ai_risk_assessment.md`"
    )
