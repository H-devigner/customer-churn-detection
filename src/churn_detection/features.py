"""Feature engineering shared by training and the Flask prediction app."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_feature_engineering(df: pd.DataFrame, target_column: str | None = None) -> pd.DataFrame:
    """Add simple, explainable churn features when source columns exist."""

    df = df.copy()
    tenure_column = _first_existing_column(df, ["tenure_months", "tenure"])
    monthly_column = _first_existing_column(df, ["monthly_charges", "MonthlyCharges", "monthlycharges"])
    total_column = _first_existing_column(df, ["total_charges", "TotalCharges", "totalcharges"])

    if tenure_column is not None:
        safe_tenure = df[tenure_column].clip(lower=1)
        df["is_new_customer"] = (df[tenure_column] <= 6).astype(int)
        df["is_long_tenure_customer"] = (df[tenure_column] >= 48).astype(int)

        if total_column is not None:
            df["average_monthly_spend"] = df[total_column] / safe_tenure

        if "support_calls" in df.columns:
            df["support_calls_per_year"] = df["support_calls"] / (safe_tenure / 12)

        if "late_payments" in df.columns:
            df["late_payments_per_year"] = df["late_payments"] / (safe_tenure / 12)

    if monthly_column is not None and total_column is not None:
        df["charge_to_total_ratio"] = df[monthly_column] / df[total_column].clip(lower=1)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find a column by trying exact names and case-insensitive names."""

    lowered = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None
