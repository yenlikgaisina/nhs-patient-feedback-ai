"""Streamlit app for the NHS Patient Feedback Intelligence prototype."""
import pandas as pd
import streamlit as st

from predict import CONFIDENCE_THRESHOLD, predict_batch, predict_feedback

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


st.set_page_config(
    page_title="NHS Patient Feedback Intelligence App",
    page_icon="🏥",
    layout="centered",
)

st.title("🏥 NHS Patient Feedback Intelligence App")
st.write(
    "Classify NHS-style patient feedback into service-improvement themes. "
    "Portfolio prototype only — not a clinical decision tool."
)

tab_single, tab_batch, tab_about = st.tabs(
    ["Single comment", "Batch CSV", "Model & responsible AI"]
)


# ---------------------------------------------------------------------------
# Tab 1 — Single comment analysis
# ---------------------------------------------------------------------------

with tab_single:
    comment = st.text_area(
        "Paste a patient comment:",
        height=180,
        placeholder=(
            "Example: I waited weeks for an appointment and nobody explained the delay."
        ),
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
    st.write(
        "Upload a CSV of patient comments to classify in bulk. "
        "The file must contain a column with the comment text."
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
                out["predicted_theme_label"] = [pretty(r["theme"]) for r in results]
                out["sentiment"] = [r["sentiment"] for r in results]
                out["confidence"] = [r["confidence"] for r in results]
                out["needs_review"] = [r["needs_review"] for r in results]
                out["suggested_action"] = [r["suggested_action"] for r in results]

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
                    file_name="nhs_feedback_predictions.csv",
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
        "- **Algorithm:** TF-IDF (unigrams + bigrams) + Logistic Regression\n"
        "- **Themes:** appointment access, communication, staff attitude, "
        "waiting time, treatment quality, administration, facilities\n"
        "- **Sentiment:** TextBlob polarity (positive / negative / neutral)\n"
        f"- **Review threshold:** predictions below {CONFIDENCE_THRESHOLD:.0%} "
        "confidence are flagged for human review"
    )

    st.subheader("Responsible AI")
    st.markdown(
        "- **Synthetic development data.** The included dataset is a synthetic, "
        "NHS-style dataset for reproducible portfolio deployment. The reported "
        "1.00 accuracy reflects memorisation of a small fixed phrase pool and "
        "is not a real-world performance estimate.\n"
        "- **Weak labels.** Training labels are rule-based starter labels, not "
        "expert-annotated clinical categories.\n"
        "- **Not a clinical tool.** Do not use for diagnosis, patient safety "
        "decisions, or automated complaint handling.\n"
        "- **Human review required.** Low-confidence predictions are flagged "
        "and should be triaged by a person before any operational action.\n"
        "- **No demographic fairness testing yet** — performance may differ "
        "across patient groups."
    )

    st.caption(
        "Full model card: `reports/model_card.md` · "
        "Risk assessment: `reports/responsible_ai_risk_assessment.md`"
    )
