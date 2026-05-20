# Dataset Notes

## Supported Data

The training script can use either:

1. A Kaggle-style customer churn CSV placed in `data/raw/`.
2. A generated synthetic dataset if no compatible CSV is found.

The loader searches CSV files in `data/raw/` and looks for one of these churn target columns:

- `Churn`
- `churn`
- `Exited`
- `Attrition_Flag`
- `Customer_Status`

For the common Telco Customer Churn dataset, the script also cleans `TotalCharges`, which sometimes appears as text because of blank values.

## Synthetic Fallback

The fallback dataset is created with realistic churn signals:

- Month-to-month contracts tend to churn more often.
- Longer tenure reduces churn risk.
- More support calls and late payments increase churn risk.
- Automatic payment methods reduce churn risk.
- Higher monthly charges can increase churn risk.

This gives the model enough structure to learn from while keeping the project fully runnable without external credentials.

## Replacing The Data

To use your own data:

1. Put a CSV in `data/raw/`.
2. Make sure it has a churn target column.
3. Run `python train.py`.

The code automatically detects numeric and categorical columns, so you usually do not need to edit the feature pipeline.
