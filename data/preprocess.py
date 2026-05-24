"""
preprocess.py — turn the raw NHS-style XLS files into the processed CSV
that train_model.py consumes.

Inputs:
    data/raw/hospital_comments.xls
    data/raw/gp_comments.xls

Output:
    data/processed/nhs_patient_feedback_processed.csv

Pipeline:
    load XLS -> concat with service_type -> clean text -> rule-based theme
    labels -> TextBlob sentiment -> write CSV.

Usage:
    python data/preprocess.py
"""
from __future__ import annotations

import re
import string
from pathlib import Path

import nltk
import pandas as pd
from textblob import TextBlob

RAW_DIR = Path(__file__).parent / "raw"
PROCESSED_DIR = Path(__file__).parent / "processed"
HOSPITAL_PATH = RAW_DIR / "hospital_comments.xls"
GP_PATH = RAW_DIR / "gp_comments.xls"
OUT_PATH = PROCESSED_DIR / "nhs_patient_feedback_processed.csv"

THEME_KEYWORDS = {
    "appointment_access": [
        "appointment", "book", "booking", "access", "available",
        "phone", "call", "reception", "gp appointment",
    ],
    "communication": [
        "communication", "explain", "explained", "told", "listen",
        "listened", "information", "contact", "update",
    ],
    "staff_attitude": [
        "rude", "kind", "helpful", "unhelpful", "staff", "nurse",
        "doctor", "receptionist", "attitude", "care",
    ],
    "waiting_time": [
        "wait", "waiting", "delay", "delayed", "late", "queue",
        "hours", "minutes",
    ],
    "treatment_quality": [
        "treatment", "diagnosis", "diagnosed", "medicine", "medication",
        "pain", "care", "procedure", "symptoms",
    ],
    "administration": [
        "letter", "referral", "form", "admin", "paperwork",
        "record", "prescription", "test results", "results",
    ],
    "facilities": [
        "clean", "dirty", "parking", "room", "ward",
        "toilet", "building", "facility", "equipment",
    ],
}


def _ensure_nltk():
    for pkg in ("stopwords", "punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}" if pkg.startswith("punkt") else f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


def _load_raw() -> pd.DataFrame:
    hosp = pd.read_excel(HOSPITAL_PATH, engine="openpyxl")
    gp = pd.read_excel(GP_PATH, engine="openpyxl")
    hosp["service_type"] = "Hospital"
    gp["service_type"] = "GP"
    df = pd.concat([hosp, gp], ignore_index=True)
    return df.rename(
        columns={
            "OrganisationName": "organisation",
            "Comment": "comment",
            "Rating": "rating",
        }
    )[["organisation", "comment", "rating", "service_type"]]


def _clean_text(text: str, stop_words: set[str]) -> str:
    from nltk.tokenize import word_tokenize

    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    return " ".join(
        w for w in tokens if w.isalpha() and w not in stop_words and len(w) > 2
    )


def _assign_theme(text: str) -> str:
    text = str(text).lower()
    scores = {
        theme: sum(1 for kw in keywords if kw in text)
        for theme, keywords in THEME_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return "other" if scores[best] == 0 else best


def _sentiment(text: str) -> str:
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0.1:
        return "positive"
    if polarity < -0.1:
        return "negative"
    return "neutral"


def main() -> None:
    _ensure_nltk()
    from nltk.corpus import stopwords

    stop_words = set(stopwords.words("english"))

    df = _load_raw()
    print(f"Loaded {len(df):,} raw rows")

    df["clean_comment"] = df["comment"].apply(lambda t: _clean_text(t, stop_words))
    df = df[df["clean_comment"].str.len() > 0].copy()
    df["theme"] = df["comment"].apply(_assign_theme)
    df["sentiment"] = df["comment"].apply(_sentiment)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}  ({len(df):,} rows)")
    print("\nTheme distribution:")
    print(df["theme"].value_counts())


if __name__ == "__main__":
    main()
