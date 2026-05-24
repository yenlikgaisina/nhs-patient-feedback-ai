# Model Card: NHS Patient Feedback Theme Classifier

## Model purpose
This model classifies patient feedback comments into service-improvement themes such as appointment access, communication, staff attitude, waiting time, treatment quality, administration, and facilities.

## Intended use
The model is intended as a portfolio prototype for service-improvement triage and operational insight.

## Not intended use
This model must not be used for clinical diagnosis, patient safety decisions, automated complaint resolution, or replacing human review.

## Data
The model uses public NHS patient comments from NHS Choices datasets available on data.gov.uk. The project uses rule-based starter labels created from theme keywords — these are not professionally annotated clinical labels.

## Model
Baseline model: TF-IDF vectorisation with Logistic Regression.

Future iteration: DistilBERT fine-tuned on expert-labelled data.

## Evaluation
Metrics: accuracy, macro F1, classification report, confusion matrix.

## Limitations
- Labels are weak labels, not professionally annotated ground truth
- The model may misclassify comments with sarcasm, mixed themes, vague wording, or multiple issues
- Not tested on demographic subgroups — potential for performance disparity
- TextBlob sentiment scoring is basic and not domain-adapted for healthcare language

## Responsible AI considerations
A human reviewer should validate outputs before operational action. Sensitive comments should be treated carefully and not exposed in public demos.
