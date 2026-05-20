"""Flask UI for scoring customer churn risk."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "customer_churn_model.joblib"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from churn_detection.features import add_feature_engineering


app = Flask(__name__)


FORM_DEFAULTS = {
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "No",
    "tenure_months": 18,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "tech_support": "No",
    "payment_method": "Electronic check",
    "support_calls": 2,
    "late_payments": 1,
    "monthly_charges": 84.9,
    "total_charges": 1528.2,
}


def load_model_bundle() -> dict:
    """Load the trained model bundle saved by train.py."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found. Run `python train.py` before starting the app.")

    bundle = joblib.load(MODEL_PATH)
    if isinstance(bundle, dict):
        return bundle

    return {
        "model": bundle,
        "threshold": 0.5,
        "feature_columns": None,
        "target_column": "churn",
    }


def parse_customer(form_data) -> dict:
    """Convert form values into one raw customer row."""

    tenure_months = _to_int(form_data.get("tenure_months"), FORM_DEFAULTS["tenure_months"])
    monthly_charges = _to_float(form_data.get("monthly_charges"), FORM_DEFAULTS["monthly_charges"])
    total_charges = _to_float(form_data.get("total_charges"), monthly_charges * max(tenure_months, 1))

    return {
        "gender": form_data.get("gender", FORM_DEFAULTS["gender"]),
        "senior_citizen": _to_int(form_data.get("senior_citizen"), FORM_DEFAULTS["senior_citizen"]),
        "partner": form_data.get("partner", FORM_DEFAULTS["partner"]),
        "dependents": form_data.get("dependents", FORM_DEFAULTS["dependents"]),
        "tenure_months": tenure_months,
        "contract": form_data.get("contract", FORM_DEFAULTS["contract"]),
        "internet_service": form_data.get("internet_service", FORM_DEFAULTS["internet_service"]),
        "online_security": form_data.get("online_security", FORM_DEFAULTS["online_security"]),
        "tech_support": form_data.get("tech_support", FORM_DEFAULTS["tech_support"]),
        "payment_method": form_data.get("payment_method", FORM_DEFAULTS["payment_method"]),
        "support_calls": _to_int(form_data.get("support_calls"), FORM_DEFAULTS["support_calls"]),
        "late_payments": _to_int(form_data.get("late_payments"), FORM_DEFAULTS["late_payments"]),
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
    }


def prepare_features(customer: dict, feature_columns: list[str] | None) -> pd.DataFrame:
    """Apply feature engineering and align to the training feature order."""

    features = add_feature_engineering(pd.DataFrame([customer]))
    if feature_columns is None:
        return features

    for column in feature_columns:
        if column not in features.columns:
            features[column] = np.nan

    return features[feature_columns]


def score_customer(customer: dict) -> dict:
    """Return churn probability, class, and practical retention hints."""

    bundle = load_model_bundle()
    model = bundle["model"]
    threshold = float(bundle.get("threshold", 0.5))
    features = prepare_features(customer, bundle.get("feature_columns"))

    probability = float(model.predict_proba(features)[:, 1][0])
    will_churn = probability >= threshold

    return {
        "probability": probability,
        "probability_percent": round(probability * 100, 1),
        "threshold_percent": round(threshold * 100, 1),
        "label": "Likely to churn" if will_churn else "Likely to stay",
        "band": risk_band(probability),
        "will_churn": will_churn,
        "recommendations": build_recommendations(customer, probability),
    }


def risk_band(probability: float) -> str:
    if probability >= 0.70:
        return "High risk"
    if probability >= 0.45:
        return "Watch list"
    return "Stable"


def build_recommendations(customer: dict, probability: float) -> list[str]:
    """Create short, practical suggestions from the entered values."""

    recommendations = []
    if customer["contract"] == "Month-to-month":
        recommendations.append("Offer a one-year plan with a small loyalty credit.")
    if customer["support_calls"] >= 3:
        recommendations.append("Route to a senior support follow-up before the next billing cycle.")
    if customer["late_payments"] >= 2:
        recommendations.append("Suggest autopay or a softer billing reminder cadence.")
    if customer["tenure_months"] <= 6:
        recommendations.append("Send an onboarding check-in and confirm the customer is getting value.")
    if customer["payment_method"] == "Electronic check":
        recommendations.append("Promote card or bank transfer payment with a one-time incentive.")
    if customer["monthly_charges"] >= 90:
        recommendations.append("Review plan fit and remove unused add-ons.")

    if not recommendations and probability < 0.45:
        recommendations.append("Keep the customer in the standard nurture flow.")
    elif not recommendations:
        recommendations.append("Schedule a light-touch retention call.")

    return recommendations[:4]


def _to_float(value: str | None, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value: str | None, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


@app.get("/")
@app.post("/")
def index():
    form_values = FORM_DEFAULTS.copy()
    result = None
    error = None

    if request.method == "POST":
        form_values = parse_customer(request.form)
        try:
            result = score_customer(form_values)
        except FileNotFoundError as exc:
            error = str(exc)

    return render_template("index.html", form_values=form_values, result=result, error=error)


@app.get("/health")
def health():
    return {"status": "ok", "model_available": MODEL_PATH.exists()}


if __name__ == "__main__":
    app.run(debug=True, port=5002)
