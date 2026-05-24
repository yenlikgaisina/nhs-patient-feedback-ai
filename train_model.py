"""
train_model.py — Run the full training pipeline from processed data.

Usage:
    python train_model.py

Expects:  data/processed/nhs_patient_feedback_processed.csv
Outputs:  models/model.pkl
"""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROCESSED_DATA_PATH = "data/processed/nhs_patient_feedback_processed.csv"
MODEL_PATH = "models/model.pkl"


def load_data():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    model_df = df[df["theme"] != "other"].copy()
    X = model_df["comment"]
    y = model_df["theme"]
    return X, y


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=10000
        )),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])


def train():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = build_pipeline()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    # --- MLflow tracking ---
    mlflow.set_experiment("nhs_patient_feedback_theme_classifier")
    with mlflow.start_run():
        mlflow.log_param("model_type", "TF-IDF + Logistic Regression")
        mlflow.log_param("ngram_range", "(1,2)")
        mlflow.log_param("max_features", 10000)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("macro_f1", f1)
        mlflow.sklearn.log_model(clf, "model")

    # --- Save model ---
    Path("models").mkdir(exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
