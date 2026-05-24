# Responsible AI Risk Assessment

## Project
NHS Patient Feedback Intelligence App

## Main risks

### 1. Misclassification
The model may assign the wrong theme, especially when a comment contains several issues.

**Mitigation:** Show confidence score and recommend human review for low-confidence cases.

### 2. Weak labels
The training labels are rule-based starter labels, not expert-labelled clinical categories.

**Mitigation:** Clearly document this limitation and avoid clinical claims.

### 3. Sensitive patient content
Patient comments may contain personal or emotional details.

**Mitigation:** Do not publish raw identifiable comments in the deployed app or README.

### 4. Over-trust by users
A user may assume the model is clinically authoritative.

**Mitigation:** Include visible disclaimer that the tool is for operational triage only.

### 5. Bias
Some patient groups may use different language to describe similar experiences. If certain demographics are under-represented in the training data, the model may perform worse for those groups.

**Mitigation:** Monitor model performance across subgroups. Avoid making decisions without human oversight.

### 6. Data staleness
The NHS Choices dataset covers a specific time period. Language patterns may have shifted.

**Mitigation:** Document training data date range in the model card and plan for retraining when newer data becomes available.

## Final position
This prototype is suitable for portfolio demonstration and service-improvement exploration. It is not suitable for real NHS deployment without governance, privacy review, expert labelling, and human oversight.
