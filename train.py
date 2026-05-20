"""Train and validate a customer churn classifier.

Run from the project root:

    python train.py

The script prefers CSV files in data/raw/ and falls back to a synthetic dataset
so the project can be executed without Kaggle credentials.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
(PROJECT_ROOT / ".matplotlib-cache").mkdir(exist_ok=True)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_detection.data import load_or_create_dataset, save_processed_dataset
from churn_detection.features import add_feature_engineering
from churn_detection.plots import (
    plot_churn_rate_by_category,
    plot_class_balance,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_learning_curve,
    plot_numeric_by_churn,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_threshold_metrics,
)


RANDOM_STATE = 42
ROC_AUC_TOLERANCE = 0.005
MIN_SELECTION_PRECISION = 0.55


def main() -> None:
    """Run the complete modeling workflow."""

    _ensure_directories()

    df, target_column, source = load_or_create_dataset(PROJECT_ROOT, random_state=RANDOM_STATE)
    df = add_feature_engineering(df, target_column)
    save_processed_dataset(df, PROJECT_ROOT)

    X = df.drop(columns=target_column)
    y = df[target_column].astype(int)

    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=0.25,
        stratify=y_development,
        random_state=RANDOM_STATE,
    )

    fitted_models = {}
    validation_comparison = {}
    test_comparison = {}
    for model_name, estimator in build_candidate_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X_train)),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)

        threshold, validation_metrics = tune_threshold_for_f1(pipeline, X_validation, y_validation)
        fitted_models[model_name] = pipeline
        validation_comparison[model_name] = validation_metrics
        test_comparison[model_name] = evaluate_model(pipeline, X_test, y_test, threshold=threshold)

    best_name = select_best_model(validation_comparison)
    best_model = fitted_models[best_name]
    best_metrics = test_comparison[best_name]

    save_outputs(
        df=df,
        target_column=target_column,
        source=source,
        X_development=X_development,
        X_test=X_test,
        y_development=y_development,
        y_test=y_test,
        best_name=best_name,
        best_model=best_model,
        best_metrics=best_metrics,
        validation_comparison=validation_comparison,
        test_comparison=test_comparison,
        split_sizes={
            "train": int(len(X_train)),
            "validation": int(len(X_validation)),
            "test": int(len(X_test)),
        },
    )

    print(f"Dataset source: {source}")
    print(f"Rows: {len(df):,} | Churn rate: {y.mean():.1%}")
    print(f"Best model: {best_name}")
    print(json.dumps(best_metrics, indent=2))


def build_candidate_models() -> dict[str, object]:
    """Return diverse model candidates for the validation sweep."""

    return {
        "logistic_regression": LogisticRegression(max_iter=2_000),
        "logistic_regression_balanced_tuned": LogisticRegression(
            max_iter=2_000,
            class_weight="balanced",
            C=0.7,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=350,
            max_depth=10,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=450,
            max_depth=12,
            min_samples_leaf=8,
            class_weight="balanced",
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.045,
            max_leaf_nodes=15,
            l2_regularization=0.05,
            random_state=RANDOM_STATE,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.045,
            max_depth=2,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
        ),
        "ada_boost": AdaBoostClassifier(
            n_estimators=120,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
        ),
        "soft_voting_challenger": VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(max_iter=2_000)),
                (
                    "balanced_lr",
                    LogisticRegression(max_iter=2_000, class_weight="balanced", C=0.7),
                ),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=350,
                        max_depth=10,
                        min_samples_leaf=8,
                        class_weight="balanced_subsample",
                        n_jobs=1,
                        random_state=RANDOM_STATE,
                    ),
                ),
                (
                    "hgb",
                    HistGradientBoostingClassifier(
                        max_iter=180,
                        learning_rate=0.045,
                        max_leaf_nodes=15,
                        l2_regularization=0.05,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ],
            voting="soft",
            weights=[1, 2, 1, 1],
            n_jobs=1,
        ),
    }


def select_best_model(validation_comparison: dict[str, dict[str, float]]) -> str:
    """Select a churn-focused model without using the final test set.

    Churn programs usually prefer catching more likely churners, as long as the
    model still has near-best ranking quality and reasonable precision.
    """

    best_roc_auc = max(metrics["roc_auc"] for metrics in validation_comparison.values())
    eligible_models = {
        name: metrics
        for name, metrics in validation_comparison.items()
        if metrics["roc_auc"] >= best_roc_auc - ROC_AUC_TOLERANCE
        and metrics["precision"] >= MIN_SELECTION_PRECISION
    }
    if not eligible_models:
        eligible_models = validation_comparison

    return max(
        eligible_models,
        key=lambda name: (
            eligible_models[name]["recall"],
            eligible_models[name]["f1"],
            eligible_models[name]["average_precision"],
        ),
    )


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing based on the dataframe's column types."""

    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        verbose_feature_names_out=False,
    )


def tune_threshold_for_f1(model: Pipeline, X_validation: pd.DataFrame, y_validation: pd.Series) -> tuple[float, dict[str, float]]:
    """Choose the probability threshold with the best validation F1 score."""

    y_probability = model.predict_proba(X_validation)[:, 1]
    candidate_thresholds = np.linspace(0.10, 0.90, 161)

    best_threshold = 0.50
    best_score = -1.0
    for threshold in candidate_thresholds:
        y_prediction = (y_probability >= threshold).astype(int)
        score = f1_score(y_validation, y_prediction, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, evaluate_model(model, X_validation, y_validation, threshold=best_threshold)


def predict_with_threshold(model: Pipeline, X: pd.DataFrame, threshold: float) -> np.ndarray:
    """Predict churn labels with a configurable probability threshold."""

    return (model.predict_proba(X)[:, 1] >= threshold).astype(int)


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.50) -> dict[str, float]:
    """Evaluate one fitted model on the holdout set."""

    y_probability = model.predict_proba(X_test)[:, 1]
    y_prediction = (y_probability >= threshold).astype(int)

    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(accuracy_score(y_test, y_prediction), 4),
        "precision": round(precision_score(y_test, y_prediction, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_prediction, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_prediction, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_probability), 4),
        "average_precision": round(average_precision_score(y_test, y_probability), 4),
    }


def save_outputs(
    *,
    df: pd.DataFrame,
    target_column: str,
    source: str,
    X_development: pd.DataFrame,
    X_test: pd.DataFrame,
    y_development: pd.Series,
    y_test: pd.Series,
    best_name: str,
    best_model: Pipeline,
    best_metrics: dict[str, float],
    validation_comparison: dict[str, dict[str, float]],
    test_comparison: dict[str, dict[str, float]],
    split_sizes: dict[str, int],
) -> None:
    """Save model artifacts, metrics, and figures."""

    figure_dir = PROJECT_ROOT / "reports" / "figures"
    metrics_dir = PROJECT_ROOT / "reports" / "metrics"
    model_dir = PROJECT_ROOT / "models"

    best_threshold = best_metrics["threshold"]
    y_test_probability = best_model.predict_proba(X_test)[:, 1]
    y_test_prediction = (y_test_probability >= best_threshold).astype(int)

    joblib.dump(
        {
            "model": best_model,
            "threshold": best_threshold,
            "target_column": target_column,
            "feature_columns": X_test.columns.tolist(),
        },
        model_dir / "customer_churn_model.joblib",
    )

    metrics_payload = {
        "dataset_source": source,
        "rows": int(len(df)),
        "churn_rate": round(float(df[target_column].mean()), 4),
        "split_sizes": split_sizes,
        "selection_metric": "validation_recall_with_near_best_roc_auc",
        "selection_rule": {
            "roc_auc_tolerance": ROC_AUC_TOLERANCE,
            "minimum_precision": MIN_SELECTION_PRECISION,
            "tie_breakers": ["recall", "f1", "average_precision"],
        },
        "best_model": best_name,
        "selected_threshold": best_threshold,
        "best_model_metrics": best_metrics,
        "validation_model_comparison": validation_comparison,
        "test_model_comparison_at_validation_threshold": test_comparison,
    }
    with (metrics_dir / "model_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics_payload, file, indent=2)

    report = classification_report(
        y_test,
        y_test_prediction,
        target_names=["Not churned", "Churned"],
        zero_division=0,
    )
    with (metrics_dir / "classification_report.txt").open("w", encoding="utf-8") as file:
        file.write(report)

    plot_class_balance(df, target_column, figure_dir / "class_balance.png")
    if "contract" in df.columns:
        plot_churn_rate_by_category(df, target_column, "contract", figure_dir / "churn_rate_by_contract.png")
    if "monthly_charges" in df.columns:
        plot_numeric_by_churn(df, target_column, "monthly_charges", figure_dir / "monthly_charges_by_churn.png")

    plot_learning_curve(best_model, X_development, y_development, figure_dir / "learning_curve.png", random_state=RANDOM_STATE)
    plot_confusion_matrix(y_test, y_test_prediction, figure_dir / "confusion_matrix.png")
    plot_roc_curve(best_model, X_test, y_test, figure_dir / "roc_curve.png")
    plot_precision_recall_curve(best_model, X_test, y_test, figure_dir / "precision_recall_curve.png")
    plot_threshold_metrics(y_test, y_test_probability, best_threshold, figure_dir / "threshold_tradeoff.png")
    plot_feature_importance(best_model, figure_dir / "feature_importance.png", X=X_test, y=y_test)


def _ensure_directories() -> None:
    for relative_path in [
        "data/raw",
        "data/processed",
        "models",
        "reports/figures",
        "reports/metrics",
    ]:
        (PROJECT_ROOT / relative_path).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
