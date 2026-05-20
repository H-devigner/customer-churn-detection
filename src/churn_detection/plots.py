"""Plotting helpers for the churn modeling workflow."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import StratifiedKFold, learning_curve


FIGURE_DPI = 140


def plot_class_balance(df: pd.DataFrame, target_column: str, output_path: Path) -> None:
    counts = df[target_column].value_counts().sort_index()
    labels = ["Not churned", "Churned"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, counts.values, color=["#2f6f73", "#d36b45"])
    ax.set_title("Class Balance")
    ax.set_ylabel("Customers")
    _annotate_bars(ax)
    _save(fig, output_path)


def plot_churn_rate_by_category(
    df: pd.DataFrame,
    target_column: str,
    category_column: str,
    output_path: Path,
) -> None:
    rates = df.groupby(category_column)[target_column].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(rates.index.astype(str), rates.values, color="#5b7f95")
    ax.set_title(f"Churn Rate By {category_column.replace('_', ' ').title()}")
    ax.set_ylabel("Churn rate")
    ax.set_ylim(0, max(0.1, rates.max() * 1.25))
    ax.tick_params(axis="x", rotation=20)
    _annotate_bars(ax, percentage=True)
    _save(fig, output_path)


def plot_numeric_by_churn(
    df: pd.DataFrame,
    target_column: str,
    numeric_column: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    df.boxplot(column=numeric_column, by=target_column, ax=ax, grid=False, color="#354f52")
    ax.set_title(f"{numeric_column.replace('_', ' ').title()} By Churn")
    ax.set_xlabel("Churn")
    ax.set_ylabel(numeric_column.replace("_", " ").title())
    ax.set_xticklabels(["Not churned", "Churned"])
    fig.suptitle("")
    _save(fig, output_path)


def plot_learning_curve(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    output_path: Path,
    random_state: int,
) -> None:
    """Plot train and validation ROC-AUC as training size grows."""

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
        scoring="roc_auc",
        train_sizes=np.linspace(0.15, 1.0, 6),
        # Keep this single-process so the project runs in restricted notebooks,
        # sandboxes, and classroom machines without multiprocessing surprises.
        n_jobs=1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    validation_mean = validation_scores.mean(axis=1)
    validation_std = validation_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_sizes, train_mean, marker="o", label="Training ROC-AUC", color="#2f6f73")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#2f6f73", alpha=0.15)
    ax.plot(train_sizes, validation_mean, marker="o", label="Validation ROC-AUC", color="#c44e52")
    ax.fill_between(
        train_sizes,
        validation_mean - validation_std,
        validation_mean + validation_std,
        color="#c44e52",
        alpha=0.15,
    )
    ax.set_title("Learning Curve")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.5, 1.02)
    ax.legend()
    _save(fig, output_path)


def plot_confusion_matrix(y_test: pd.Series, y_prediction: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_prediction,
        display_labels=["Not churned", "Churned"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Confusion Matrix")
    _save(fig, output_path)


def plot_roc_curve(estimator, X_test: pd.DataFrame, y_test: pd.Series, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    display = RocCurveDisplay.from_estimator(estimator, X_test, y_test, ax=ax)
    display.line_.set_color("#2f6f73")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1)
    ax.set_title("ROC Curve")
    _save(fig, output_path)


def plot_precision_recall_curve(estimator, X_test: pd.DataFrame, y_test: pd.Series, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    display = PrecisionRecallDisplay.from_estimator(estimator, X_test, y_test, ax=ax)
    display.line_.set_color("#d36b45")
    ax.set_title("Precision-Recall Curve")
    _save(fig, output_path)


def plot_threshold_metrics(
    y_true: pd.Series,
    y_probability: np.ndarray,
    selected_threshold: float,
    output_path: Path,
) -> None:
    """Plot precision, recall, and F1 across possible churn thresholds."""

    from sklearn.metrics import f1_score, precision_score, recall_score

    thresholds = np.linspace(0.05, 0.95, 181)
    precision_values = []
    recall_values = []
    f1_values = []

    for threshold in thresholds:
        y_prediction = (y_probability >= threshold).astype(int)
        precision_values.append(precision_score(y_true, y_prediction, zero_division=0))
        recall_values.append(recall_score(y_true, y_prediction, zero_division=0))
        f1_values.append(f1_score(y_true, y_prediction, zero_division=0))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, precision_values, label="Precision", color="#5b7f95")
    ax.plot(thresholds, recall_values, label="Recall", color="#d36b45")
    ax.plot(thresholds, f1_values, label="F1", color="#2f6f73")
    ax.axvline(selected_threshold, color="#333333", linestyle="--", linewidth=1, label="Selected threshold")
    ax.set_title("Threshold Trade-Off")
    ax.set_xlabel("Churn probability threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.legend()
    _save(fig, output_path)


def plot_feature_importance(
    estimator,
    output_path: Path,
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    top_n: int = 15,
) -> None:
    """Plot model drivers from native importance, coefficients, or permutation."""

    preprocessing = estimator.named_steps["preprocess"]
    model = estimator.named_steps["model"]
    feature_names = preprocessing.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        x_label = "Importance"
        title = "Top Feature Importances"
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
        x_label = "Absolute coefficient"
        title = "Top Absolute Coefficients"
    else:
        if X is None or y is None:
            return
        result = permutation_importance(
            estimator,
            X,
            y,
            n_repeats=8,
            random_state=42,
            scoring="roc_auc",
            n_jobs=1,
        )
        values = result.importances_mean
        feature_names = np.array(X.columns)
        x_label = "Mean ROC-AUC decrease"
        title = "Top Permutation Importances"

    top_index = np.argsort(values)[-top_n:]
    sorted_names = feature_names[top_index]
    sorted_values = values[top_index]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(sorted_names, sorted_values, color="#5b7f95")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.tick_params(axis="y", labelsize=8)
    _save(fig, output_path)


def _annotate_bars(ax: plt.Axes, percentage: bool = False) -> None:
    for patch in ax.patches:
        value = patch.get_height()
        label = f"{value:.1%}" if percentage else f"{value:,.0f}"
        ax.annotate(
            label,
            (patch.get_x() + patch.get_width() / 2, value),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 3),
            textcoords="offset points",
        )


def _save(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
