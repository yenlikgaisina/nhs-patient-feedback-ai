# NHS Patient Feedback Intelligence App

[![CI](https://github.com/yenlikgaisina/nhs-patient-feedback-ai/actions/workflows/tests.yml/badge.svg)](https://github.com/yenlikgaisina/nhs-patient-feedback-ai/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Live demo

Try the public demo on Hugging Face Spaces:
https://huggingface.co/spaces/yenlikg/patient-feedback-intelligence-app

Independent portfolio prototype. Not affiliated with, endorsed by, or approved by the NHS. Use fictional or anonymised comments only.

An end-to-end NLP portfolio project: an NLP classifier for NHS-style patient feedback, served through a Streamlit app with single-comment analysis, batch CSV prediction, and a built-in responsible-AI summary.

> **Important.** This project uses a synthetic, NHS-style development dataset bundled in the repo for reproducible portfolio deployment. The model is not a clinical tool and should not be used for diagnosis, patient safety decisions, or automated complaint handling without human review.
>
> ## What the app does
>
> - **Single comment tab** — paste a patient comment and get the predicted service-improvement theme, sentiment, confidence score, a `needs human review` flag for low-confidence predictions, a top-3 theme bar chart, and a suggested operational action.
> - - **Batch CSV tab** — upload a CSV of comments, pick the text column, run predictions across all rows, see the theme distribution and the count of rows flagged for review, then download the labelled CSV.
>   - - **Model & responsible AI tab** — a plain-language summary of the model, its limitations, and the responsible-AI considerations.
>    
>     - ## Business question
>    
>     - What are the main drivers of negative patient experience across NHS services, and can an AI tool classify new patient comments into actionable service-improvement themes — while keeping a human in the loop?
>    
>     - ## Methods
>
> - Synthetic NHS-style XLS generation (`data/make_dataset.py`) so the project is reproducible without downloading the public dataset.
> - - Text cleaning with NLTK (tokenisation, stopword removal, punctuation strip).
>   - - Rule-based starter labels across 7 themes (`data/preprocess.py`).
>     - - TF-IDF (unigrams + bigrams, up to 10k features) + Logistic Regression with balanced class weights.
>       - - TextBlob polarity for sentiment.
>         - - Single-row and batched prediction API with top-3 probabilities and a configurable `needs_review` confidence threshold (default 0.55).
>           - - MLflow experiment tracking (`mlruns/`).
>             - - Streamlit deployment with a 3-tab interface.
>               - - GitHub Actions CI on Python 3.10 and 3.11, running the full pipeline: data → train → lint → test.
>                 - - Pytest suite (19 tests) covering the prediction contract, top-3 ordering, the review threshold, batch prediction, and edge cases.
>                   - - Responsible AI documentation: model card and risk assessment.
>                    
>                     - ## Model performance
>                    
>                     - | Metric | Value |
>                     - |--------|-------|
>                     - | Accuracy (synthetic dev data) | 1.00 |
> | Macro F1 (synthetic dev data) | 1.00 |
> | Expected accuracy on real NHS data | ~0.65–0.75 |
> | Themes classified | 7 (appointment_access, communication, staff_attitude, waiting_time, treatment_quality, administration, facilities) |
> | Review threshold | predictions below 55% confidence are flagged for human review |
> | Training set | 80% of labelled comments (excluding the other catch-all) |
> | Test set | 20% stratified holdout |
> | Model | TF-IDF (unigrams + bigrams, max 10k features) + Logistic Regression (balanced class weights) |
>
> > The synthetic development dataset uses a fixed pool of example sentences, so the model memorises them perfectly and scores 1.00. The interesting portfolio story is the full reproducible pipeline, not the score — swap in real NHS Choices XLS files and accuracy drops to the realistic range above.
> >
> > ## Repository structure
> >
> > ```
> > nhs-patient-feedback-ai/
> > ├── README.md
> > ├── LICENSE
> > ├── Makefile                  # make install / data / train / lint / test / app / clean
> > ├── pyproject.toml            # pytest + ruff config
> > ├── requirements.txt          # pinned dependencies
> > ├── app.py                    # Streamlit app (single + batch + RAI tabs)
> > ├── train_model.py            # Training pipeline (TF-IDF + LR + MLflow)
> > ├── predict.py                # predict_feedback() + predict_batch()
> > ├── data/
> > │   ├── make_dataset.py       # generates synthetic raw XLS files
> > │   ├── preprocess.py         # raw XLS → processed CSV
> > │   ├── raw/                  # generated XLS files (gitignored)
> > │   └── processed/            # processed CSV (gitignored)
> > ├── notebooks/
> > │   └── 01_exploration_and_modelling.ipynb
> > ├── models/
> > │   └── model.pkl             # trained model (committed for HF deploy)
> > ├── reports/
> > │   ├── figures/              # EDA and evaluation charts (PNG)
> > │   ├── model_card.md
> > │   ├── responsible_ai_risk_assessment.md
> > │   └── executive_summary.md
> > ├── tests/
> > │   ├── conftest.py           # shared fixtures + missing-model guard
> > │   └── test_predict.py       # 19 tests
> > └── .github/
> >     └── workflows/
> >         └── tests.yml         # CI matrix: Python 3.10 + 3.11
> > ```
> >
> > ## How to run locally
> >
> > ```bash
> > # Set up environment (any of: venv, conda, pyenv — Python 3.9+)
> > python -m venv .venv && source .venv/bin/activate
> >
> > # Run the full pipeline
> > make install   # pip install -r requirements.txt
> > make data      # synthetic raw + processed CSV
> > make train     # build models/model.pkl + log to MLflow
> > make lint      # ruff
> > make test      # pytest -q
> > make app       # launch Streamlit on http://localhost:8501
> > ```
> >
> > `make clean` removes the model, processed data, MLflow runs, and tooling caches.
> >
> > To use real NHS Choices data instead of the synthetic set, drop the original XLS files into `data/raw/` (renamed to `hospital_comments.xls` and `gp_comments.xls`) and re-run `make train`.
> >
> > ## Continuous integration
> >
> > `.github/workflows/tests.yml` runs on every push and pull request. The matrix runs Python 3.10 and 3.11 and executes:
> >
> > `make data` → `make train` → `make lint` → `make test`
> >
> > so the entire pipeline is rebuilt from the synthetic data on every commit.
> >
> > ## Responsible AI
> >
> > This is a portfolio prototype, not a clinical tool. Labels are rule-based starter labels, not expert-annotated clinical categories. A human reviewer should validate outputs before any operational action. Low-confidence predictions (below the 55% threshold) are explicitly flagged in the app via the `needs_review` boolean.
> >
> > See `reports/model_card.md` and `reports/responsible_ai_risk_assessment.md`.
> >
> > ## Deployment
> >
> > Live demo: https://huggingface.co/spaces/yenlikg/patient-feedback-intelligence-app
> >
> > The repo is ready to deploy as-is — `models/model.pkl` is committed so the Space can launch the Streamlit app without retraining.
> >
> > ## Next improvements
> >
> > - Use expert-labelled data for supervised training instead of rule-based starter labels
> > - - Fine-tune DistilBERT for improved accuracy on real NHS comments
> >   - - Add active learning to surface low-confidence batch rows for expert labelling
> >     - - Add a manager-facing dashboard for aggregate theme trends over time
> >       - - Add monitoring and data-drift detection for production use
> >        
> >         - ## Skills demonstrated
> >        
> >         - Python · pandas · scikit-learn · NLP · TF-IDF · Logistic Regression · Streamlit · MLflow · pytest · ruff · Makefile · GitHub Actions CI · batch prediction · responsible AI · healthcare analytics · model deployment
> >        
> >         - ## Author
> >
> > Yenlik Gaisina
