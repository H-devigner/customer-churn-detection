"""Data loading and generation utilities for customer churn detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TARGET_CANDIDATES = [
    "Churn",
    "churn",
    "Exited",
    "Attrition_Flag",
    "Customer_Status",
]


def find_target_column(columns: list[str]) -> str | None:
    """Return the first known churn target column from a CSV schema."""

    normalized = {column.lower(): column for column in columns}
    for candidate in TARGET_CANDIDATES:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    return None


def load_or_create_dataset(project_root: Path, random_state: int = 42) -> tuple[pd.DataFrame, str, str]:
    """Load a churn CSV from data/raw or create a synthetic fallback dataset.

    Returns:
        A tuple of dataframe, target column name, and a short source label.
    """

    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    csv_path = _find_compatible_csv(raw_dir)
    if csv_path is not None:
        df = pd.read_csv(csv_path)
        df = normalize_churn_dataframe(df)
        source_label = (
            f"generated/reused:{csv_path.name}"
            if csv_path.name == "synthetic_customer_churn.csv"
            else f"csv:{csv_path.name}"
        )
        return df, "churn", source_label

    df = generate_synthetic_churn_data(n_rows=5_000, random_state=random_state)
    synthetic_path = raw_dir / "synthetic_customer_churn.csv"
    df.to_csv(synthetic_path, index=False)
    return df, "churn", "generated:synthetic_customer_churn.csv"


def normalize_churn_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common churn datasets to a binary `churn` target column."""

    df = df.copy()
    target_column = find_target_column(list(df.columns))
    if target_column is None:
        raise ValueError("No supported churn target column found.")

    # Common Kaggle Telco data stores TotalCharges as text with blank strings.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Customer IDs are useful keys, but they should not become model features.
    for id_column in ["customerID", "CustomerId", "RowNumber", "Surname"]:
        if id_column in df.columns:
            df = df.drop(columns=id_column)

    df["churn"] = _target_to_binary(df[target_column])
    if target_column != "churn":
        df = df.drop(columns=target_column)

    return df


def generate_synthetic_churn_data(n_rows: int = 5_000, random_state: int = 42) -> pd.DataFrame:
    """Generate a realistic tabular customer churn dataset.

    The relationships are deliberately simple and documented so the project is
    easy to understand: short tenure, monthly contracts, support issues, late
    payments, and high charges all increase churn risk.
    """

    rng = np.random.default_rng(random_state)

    tenure_months = np.clip(rng.gamma(shape=2.2, scale=14.0, size=n_rows), 0, 72).round().astype(int)
    senior_citizen = rng.binomial(1, 0.17, size=n_rows)
    partner = rng.choice(["Yes", "No"], size=n_rows, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n_rows, p=[0.31, 0.69])

    contract = _choose_contract(rng, tenure_months)
    internet_service = rng.choice(["DSL", "Fiber optic", "No"], size=n_rows, p=[0.36, 0.47, 0.17])
    online_security = _choose_online_security(rng, internet_service)
    tech_support = _choose_tech_support(rng, internet_service)
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n_rows,
        p=[0.35, 0.18, 0.24, 0.23],
    )

    support_calls = rng.poisson(lam=0.8 + (contract == "Month-to-month") * 0.7, size=n_rows)
    late_payments = rng.poisson(lam=0.35 + (payment_method == "Electronic check") * 0.35, size=n_rows)

    monthly_charges = (
        24
        + (internet_service == "DSL") * 25
        + (internet_service == "Fiber optic") * 45
        + (online_security == "Yes") * 7
        + (tech_support == "Yes") * 8
        + rng.normal(loc=0, scale=8, size=n_rows)
    )
    monthly_charges = np.clip(monthly_charges, 18, 130).round(2)
    total_charges = (monthly_charges * np.maximum(tenure_months, 1) + rng.normal(0, 60, n_rows)).round(2)
    total_charges = np.clip(total_charges, 0, None)

    churn_probability = _estimate_churn_probability(
        tenure_months=tenure_months,
        senior_citizen=senior_citizen,
        contract=contract,
        internet_service=internet_service,
        online_security=online_security,
        tech_support=tech_support,
        payment_method=payment_method,
        support_calls=support_calls,
        late_payments=late_payments,
        monthly_charges=monthly_charges,
    )
    churn = rng.binomial(1, churn_probability)

    return pd.DataFrame(
        {
            "gender": rng.choice(["Female", "Male"], size=n_rows, p=[0.51, 0.49]),
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "tenure_months": tenure_months,
            "contract": contract,
            "internet_service": internet_service,
            "online_security": online_security,
            "tech_support": tech_support,
            "payment_method": payment_method,
            "support_calls": support_calls,
            "late_payments": late_payments,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "churn": churn,
        }
    )


def save_processed_dataset(df: pd.DataFrame, project_root: Path) -> Path:
    """Save the normalized modeling dataset for inspection."""

    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "modeling_dataset.csv"
    df.to_csv(output_path, index=False)
    return output_path


def _find_compatible_csv(raw_dir: Path) -> Path | None:
    """Find the first CSV in data/raw with a supported churn target."""

    for csv_path in sorted(raw_dir.glob("*.csv")):
        try:
            sample = pd.read_csv(csv_path, nrows=5)
        except Exception:
            continue
        if find_target_column(list(sample.columns)) is not None:
            return csv_path
    return None


def _target_to_binary(series: pd.Series) -> pd.Series:
    """Convert common churn labels to 0/1 integers."""

    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)

    normalized = series.astype(str).str.strip().str.lower()
    positive_values = {
        "yes",
        "true",
        "1",
        "churned",
        "attrited customer",
        "closed",
        "lost",
    }
    return normalized.isin(positive_values).astype(int)


def _choose_contract(rng: np.random.Generator, tenure_months: np.ndarray) -> np.ndarray:
    """Make longer-tenured customers more likely to hold longer contracts."""

    month_to_month_probability = np.clip(0.72 - tenure_months / 130, 0.24, 0.72)
    one_year_probability = np.clip(0.18 + tenure_months / 260, 0.18, 0.38)
    draws = rng.random(len(tenure_months))

    return np.where(
        draws < month_to_month_probability,
        "Month-to-month",
        np.where(draws < month_to_month_probability + one_year_probability, "One year", "Two year"),
    )


def _choose_online_security(rng: np.random.Generator, internet_service: np.ndarray) -> np.ndarray:
    values = rng.choice(["Yes", "No"], size=len(internet_service), p=[0.42, 0.58])
    return np.where(internet_service == "No", "No internet service", values)


def _choose_tech_support(rng: np.random.Generator, internet_service: np.ndarray) -> np.ndarray:
    values = rng.choice(["Yes", "No"], size=len(internet_service), p=[0.39, 0.61])
    return np.where(internet_service == "No", "No internet service", values)


def _estimate_churn_probability(
    *,
    tenure_months: np.ndarray,
    senior_citizen: np.ndarray,
    contract: np.ndarray,
    internet_service: np.ndarray,
    online_security: np.ndarray,
    tech_support: np.ndarray,
    payment_method: np.ndarray,
    support_calls: np.ndarray,
    late_payments: np.ndarray,
    monthly_charges: np.ndarray,
) -> np.ndarray:
    """Create churn probabilities from known, explainable risk drivers."""

    log_odds = (
        -1.25
        - 0.035 * tenure_months
        + 0.62 * senior_citizen
        + 1.05 * (contract == "Month-to-month")
        - 0.55 * (contract == "Two year")
        + 0.62 * (internet_service == "Fiber optic")
        + 0.42 * (online_security == "No")
        + 0.35 * (tech_support == "No")
        + 0.55 * (payment_method == "Electronic check")
        - 0.38 * np.isin(payment_method, ["Bank transfer", "Credit card"])
        + 0.22 * support_calls
        + 0.30 * late_payments
        + 0.012 * (monthly_charges - 70)
    )
    return 1 / (1 + np.exp(-log_odds))
