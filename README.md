# Customer Churn Detection

A small, reproducible machine learning project for predicting whether a customer is likely to churn. The project can train on a Kaggle-style churn CSV, such as the common Telco Customer Churn dataset, and also includes a realistic synthetic fallback dataset so the full pipeline runs immediately.

## What It Does

- Loads customer churn data from `data/raw/*.csv` when available.
- Generates a synthetic customer churn dataset if no compatible CSV is found.
- Cleans numeric and categorical fields.
- Adds simple churn-focused features such as tenure flags and activity rates.
- Trains and compares Logistic Regression, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, and a soft-voting challenger.
- Tunes the churn probability threshold on a validation split.
- Validates the selected model on a separate stratified test set.
- Saves metrics, the trained model, and helpful figures.

## Project Structure

```text
customer-churn-detection/
├── data/
│   ├── raw/                 # Kaggle CSVs or generated synthetic data
│   └── processed/           # Cleaned modeling data
├── docs/
│   ├── DATASET.md
│   └── MODELING.md
├── models/                  # Trained model artifact
├── reports/
│   ├── figures/             # Learning curve, ROC, confusion matrix, etc.
│   └── metrics/             # JSON metrics and classification report
├── scripts/
│   └── setup_env.sh          # Create .venv and install requirements
├── static/                   # Flask UI styles
├── templates/                # Flask UI templates
├── src/churn_detection/
│   ├── data.py
│   ├── features.py
│   └── plots.py
├── app.py                    # Flask prediction app
├── requirements.txt
└── train.py
```

## Quick Start

```bash
cd /Users/houcine/Desktop/Random/customer-churn-detection
bash scripts/setup_env.sh
source .venv/bin/activate
python train.py
```

Manual environment setup:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
```

After training, open the generated figures in `reports/figures/` and metrics in `reports/metrics/`.

## Run The Flask App

Train once, then start the web UI:

```bash
python train.py
python app.py
```

Open `http://127.0.0.1:5001` and enter a customer profile to get a churn probability, model decision, and retention suggestions.

## Optional Kaggle Dataset

If you want to use a Kaggle dataset, place a churn CSV inside `data/raw/` before running `python train.py`.

For example, with the Kaggle CLI configured:

```bash
kaggle datasets download -d blastchar/telco-customer-churn -p data/raw --unzip
python train.py
```

The loader looks for target columns such as `Churn`, `churn`, `Exited`, or `Attrition_Flag`. If it does not find one, it falls back to the generated dataset.

## Outputs

- `models/customer_churn_model.joblib`: trained scikit-learn model bundle with the selected threshold.
- `reports/metrics/model_metrics.json`: validation metrics and model comparison.
- `reports/metrics/classification_report.txt`: precision, recall, and F1 by class.
- `reports/figures/class_balance.png`: churn class distribution.
- `reports/figures/churn_rate_by_contract.png`: churn rate by contract type.
- `reports/figures/monthly_charges_by_churn.png`: monthly charge distribution by churn.
- `reports/figures/learning_curve.png`: train vs validation ROC-AUC by sample size.
- `reports/figures/confusion_matrix.png`: holdout confusion matrix.
- `reports/figures/roc_curve.png`: ROC curve.
- `reports/figures/precision_recall_curve.png`: precision-recall curve.
- `reports/figures/threshold_tradeoff.png`: precision, recall, and F1 by threshold.
- `reports/figures/feature_importance.png`: strongest model drivers.

## Web App

The Flask UI loads `models/customer_churn_model.joblib`, applies the same feature engineering as training, and scores a single customer profile. The saved model is ignored by git, so run `python train.py` after cloning before launching `python app.py`.

## Notes

This is intentionally simple and readable. It is meant as a starter project for ML workflow practice, not a production churn system.

## Latest Demo Result

Using the generated synthetic dataset, the current pipeline selects tuned balanced Logistic Regression with a threshold of `0.48`.

- Accuracy: `0.7520`
- Precision: `0.6013`
- Recall: `0.8158`
- F1: `0.6923`
- ROC-AUC: `0.8279`
- Average precision: `0.7040`
