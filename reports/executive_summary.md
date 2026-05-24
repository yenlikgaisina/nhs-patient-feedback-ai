# Executive Summary

This project analyses NHS patient feedback comments and builds a prototype machine learning app that classifies comments into operational improvement themes.

The project demonstrates an end-to-end ML workflow:

- Data loading and cleaning from public NHS Choices XLS files
- Exploratory text analysis (word frequencies, theme distributions, sentiment breakdown)
- Rule-based starter labelling across 8 themes
- TF-IDF and Logistic Regression classification
- Model evaluation with accuracy, macro F1, confusion matrix, and error analysis
- MLflow experiment tracking
- Streamlit deployment prototype
- GitHub Actions automated testing
- Responsible AI documentation (model card and risk assessment)

The app allows a user to paste a patient comment and receive a predicted theme, sentiment score, confidence, and suggested operational action.

The main value of the project is to show how NLP can support healthcare service-improvement triage while keeping human oversight and responsible AI limitations clearly visible.

## Key metrics

| Metric | Value |
|--------|-------|
| Accuracy (synthetic dev data) | 1.00 |
| Macro F1 (synthetic dev data) | 1.00 |
| Expected accuracy on real NHS data | ~0.65–0.75 |
| Themes classified | 7 (plus `other` catch-all) |
| Training data source | NHS Choices (data.gov.uk) |
| Model type | TF-IDF (unigrams + bigrams) + Logistic Regression |
| Tests | 5/5 passing (GitHub Actions CI) |
| Figures generated | 6 (theme distribution, sentiment, length, word cloud, confusion matrix, ratings) |

## Next improvements

- Use expert-labelled data for supervised training
- Fine-tune DistilBERT for improved accuracy
- Add confidence thresholds to route low-confidence cases to human review
- Add a manager-facing dashboard for aggregate theme trends
- Add monitoring and data drift detection for production use
