"""
predict.py — single-comment and batch prediction for the NHS patient
feedback theme classifier.

Public API:
    predict_feedback(text)        -> dict
    predict_batch(texts)          -> list[dict]

Each result dict contains:
    theme              str   most likely service-improvement theme
    sentiment          str   "positive" | "negative" | "neutral"
    confidence         float top probability (None if model lacks predict_proba)
    needs_review       bool  True when confidence is below CONFIDENCE_THRESHOLD
    top_3              list  [{theme, probability}, ...] sorted by probability
    suggested_action   str   operational suggestion for the predicted theme
"""
from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

import joblib
from textblob import TextBlob

MODEL_PATH = "models/model.pkl"
CONFIDENCE_THRESHOLD = 0.55

ACTION_MAP = {
    "appointment_access": "Review appointment availability, phone access, and booking pathways.",
    "communication": "Improve explanation quality, updates, and patient-facing communication.",
    "staff_attitude": "Review staff training, empathy, and complaint-handling processes.",
    "waiting_time": "Investigate delays, queue management, and appointment scheduling.",
    "treatment_quality": "Escalate for clinical review and assess care-quality concerns.",
    "administration": "Review referrals, forms, prescriptions, test results, and admin workflows.",
    "facilities": "Review cleanliness, parking, rooms, equipment, and patient environment.",
    "other": "Route for manual review because the theme is unclear.",
}


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)


def get_sentiment(text: str) -> str:
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0.1:
        return "positive"
    if polarity < -0.1:
        return "negative"
    return "neutral"


def predict_batch(texts: Iterable[str]) -> list[dict]:
    """Predict themes for a list of comments. One result dict per input row."""
    text_list = [str(t) for t in texts]
    if not text_list:
        return []

    model = load_model()
    predictions = model.predict(text_list)

    has_proba = hasattr(model, "predict_proba")
    proba_matrix = model.predict_proba(text_list) if has_proba else None
    classes = list(model.classes_) if has_proba else None

    results = []
    for i, text in enumerate(text_list):
        prediction = predictions[i]

        if has_proba:
            probs = proba_matrix[i]
            ranked = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
            top_3 = [
                {"theme": label, "probability": float(p)} for label, p in ranked[:3]
            ]
            confidence = float(probs.max())
        else:
            top_3 = [{"theme": prediction, "probability": 1.0}]
            confidence = None

        needs_review = confidence is None or confidence < CONFIDENCE_THRESHOLD

        results.append(
            {
                "theme": prediction,
                "sentiment": get_sentiment(text),
                "confidence": confidence,
                "needs_review": needs_review,
                "top_3": top_3,
                "suggested_action": ACTION_MAP.get(prediction, ACTION_MAP["other"]),
            }
        )

    return results


def predict_feedback(text: str) -> dict:
    """Predict the theme for a single patient comment."""
    return predict_batch([text])[0]
