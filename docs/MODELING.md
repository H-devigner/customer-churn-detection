# Modeling Notes

## Model Candidates

The project trains several common tabular classifiers:

- Logistic Regression: a strong, interpretable baseline for tabular churn data.
- Balanced Logistic Regression: the same model with class weights to pay more attention to churners.
- Random Forest: a nonlinear model that can capture interactions between customer attributes.
- Extra Trees: a tree ensemble with more randomness, often useful as a robust benchmark.
- Histogram Gradient Boosting: a compact gradient-boosted tree model.
- Gradient Boosting and AdaBoost: additional boosted baselines.
- Soft Voting: a challenger ensemble that averages several model probabilities.

All candidates use the same preprocessing pipeline:

- Median imputation for missing numeric values.
- Standard scaling for numeric values.
- Most-frequent imputation for missing categorical values.
- One-hot encoding for categorical values.

## Validation

The split is stratified into train, validation, and test sets so the churn rate stays similar in each split.

The validation set is used for:

- Choosing a model.
- Tuning the probability threshold for F1.

The final test set is used only after selection. This keeps the reported metrics more honest than selecting and reporting on the same holdout set.

The model selector is churn-focused: among models with near-best validation ROC-AUC and acceptable precision, it chooses the model with the strongest validation recall. This matches the common business goal of catching more likely churners before they leave.

Saved metrics include:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- Average precision

## Figures

The generated figures are meant to answer practical questions:

- Is the dataset imbalanced?
- Which customer segments churn more?
- Does the model improve as it sees more data?
- How good is threshold-independent ranking?
- What precision/recall/F1 trade-off do we get at different thresholds?
- Which features drive the prediction?

For real business use, the decision threshold should be chosen based on the cost of false positives and false negatives, not only the default `0.50` threshold.
