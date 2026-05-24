"""Tests for the prediction module."""
import pytest

from predict import (
    ACTION_MAP,
    CONFIDENCE_THRESHOLD,
    predict_batch,
    predict_feedback,
)

# ---------------------------------------------------------------------------
# Single-comment predictions
# ---------------------------------------------------------------------------


def test_predict_feedback_returns_expected_keys(sample_text):
    result = predict_feedback(sample_text)
    expected = {
        "theme",
        "sentiment",
        "confidence",
        "needs_review",
        "top_3",
        "suggested_action",
    }
    assert expected.issubset(result.keys())


def test_confidence_is_between_0_and_1(sample_text):
    result = predict_feedback(sample_text)
    if result["confidence"] is not None:
        assert 0.0 <= result["confidence"] <= 1.0


def test_sentiment_is_valid_value(sample_text):
    result = predict_feedback(sample_text)
    assert result["sentiment"] in ("positive", "negative", "neutral")


def test_suggested_action_is_known_string(sample_text):
    result = predict_feedback(sample_text)
    assert isinstance(result["suggested_action"], str)
    assert result["suggested_action"] in ACTION_MAP.values()


# ---------------------------------------------------------------------------
# Top-3 + needs_review
# ---------------------------------------------------------------------------


def test_top_3_has_three_entries(sample_text):
    result = predict_feedback(sample_text)
    assert len(result["top_3"]) == 3
    for entry in result["top_3"]:
        assert {"theme", "probability"} <= entry.keys()
        assert 0.0 <= entry["probability"] <= 1.0


def test_top_3_is_sorted_descending(sample_text):
    result = predict_feedback(sample_text)
    probs = [e["probability"] for e in result["top_3"]]
    assert probs == sorted(probs, reverse=True)


def test_top_3_top_entry_matches_main_prediction(sample_text):
    result = predict_feedback(sample_text)
    assert result["top_3"][0]["theme"] == result["theme"]


def test_needs_review_flag_is_boolean(sample_text):
    result = predict_feedback(sample_text)
    assert isinstance(result["needs_review"], bool)


def test_needs_review_consistent_with_threshold(sample_text):
    result = predict_feedback(sample_text)
    if result["confidence"] is not None:
        assert result["needs_review"] == (result["confidence"] < CONFIDENCE_THRESHOLD)


# ---------------------------------------------------------------------------
# Edge cases — should not crash
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["a", "ok", "no"])
def test_short_input_does_not_crash(text):
    result = predict_feedback(text)
    assert "theme" in result


def test_long_input_does_not_crash():
    text = "I waited and waited. " * 500  # ~10k chars
    result = predict_feedback(text)
    assert "theme" in result


def test_special_characters_do_not_crash():
    text = "Appointment was @#$%^&*()!! 😡😡😡 — and nobody helped!"
    result = predict_feedback(text)
    assert "theme" in result


def test_empty_input_does_not_raise():
    result = predict_feedback("")
    assert "theme" in result


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------


def test_predict_batch_returns_one_row_per_input(sample_comments):
    texts = list(sample_comments.values())
    results = predict_batch(texts)
    assert len(results) == len(texts)


def test_predict_batch_empty_list_returns_empty_list():
    assert predict_batch([]) == []


def test_predict_batch_each_row_has_expected_keys(sample_comments):
    results = predict_batch(list(sample_comments.values()))
    expected = {
        "theme",
        "sentiment",
        "confidence",
        "needs_review",
        "top_3",
        "suggested_action",
    }
    for r in results:
        assert expected <= r.keys()


def test_predict_batch_matches_single_predictions(sample_comments):
    texts = list(sample_comments.values())
    single = [predict_feedback(t) for t in texts]
    batch = predict_batch(texts)
    for s, b in zip(single, batch):
        assert s["theme"] == b["theme"]
        assert s["suggested_action"] == b["suggested_action"]
