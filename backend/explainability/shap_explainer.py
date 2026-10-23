"""
SHAP Explainability Module
===========================
Provides SHAP-based explanations for demand and solar forecasts.

Outputs:
    - Technical SHAP values (for charts)
    - Plain-language officer-friendly explanations
    - Top contributing factors
    - Feature importance rankings

IMPORTANT:
    - Only XGBoost baseline uses TreeExplainer (fast, exact)
    - GRU uses KernelExplainer (approximate — computationally expensive,
      so a cached version is used at inference time)
"""

import numpy as np
import pandas as pd
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import MODELS_DIR

logger = logging.getLogger(__name__)

# Natural-language descriptions of features for officer explanations
FEATURE_PLAIN_NAMES = {
    "temperature_c":              "Temperature",
    "humidity_pct":               "Humidity",
    "rainfall_mm":                "Rainfall",
    "hour_sin":                   "Time of Day",
    "hour_cos":                   "Time of Day",
    "month_sin":                  "Season",
    "month_cos":                  "Season",
    "day_sin":                    "Day of Week",
    "day_cos":                    "Day of Week",
    "is_weekend":                 "Weekend",
    "is_holiday":                 "Holiday",
    "lag_1h_scaled":              "Previous Hour Demand",
    "lag_24h_scaled":             "Previous Day Same Hour",
    "rolling_mean_24h_scaled":    "24-Hour Demand Average",
    "zone_encoded":               "Zone Type",
    "solar_lag_1h_scaled":        "Previous Hour Solar",
    "solar_rolling_mean_6h_scaled": "Recent Solar Average",
    "hour":                       "Time of Day",
    "month":                      "Month",
    "season_encoded":             "Season",
}

# Explanation templates
IMPACT_DIRECTIONS = {
    "positive": "increased demand",
    "negative": "reduced demand",
}


def explain_xgboost_demand(
    zone: str,
    sample_df: pd.DataFrame,
    top_n: int = 6,
) -> Dict:
    """
    Generate SHAP explanations for XGBoost demand baseline predictions.
    Uses TreeExplainer (fast, exact for tree models).
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP required. pip install shap")

    model_path = MODELS_DIR / f"xgboost_demand_baseline_{zone.lower()}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"XGBoost demand model not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    demand_features = [
        "temperature_c", "humidity_pct", "rainfall_mm",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "day_sin", "day_cos", "is_weekend", "is_holiday",
        "lag_1h_scaled", "lag_24h_scaled",
        "rolling_mean_24h_scaled", "zone_encoded",
    ]

    zone_df = sample_df[sample_df["zone"] == zone].copy()
    available = [f for f in demand_features if f in zone_df.columns]
    X = zone_df[available].fillna(0).tail(1)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    feature_names = available
    shap_row = shap_values[0] if len(shap_values.shape) > 1 else shap_values

    return _build_explanation_dict(feature_names, shap_row, X.iloc[0], top_n, zone)


def explain_solar(
    sample_df: pd.DataFrame,
    top_n: int = 5,
) -> Dict:
    """
    Generate SHAP explanations for XGBoost solar model.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP required. pip install shap")

    model_path = MODELS_DIR / "xgboost_solar.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Solar model not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    feature_path = MODELS_DIR / "xgboost_solar_features.pkl"
    with open(feature_path, "rb") as f:
        features = pickle.load(f)

    if "zone" in sample_df.columns:
        solar_df = sample_df[sample_df["zone"] == "Residential"].copy()
    else:
        solar_df = sample_df.copy()

    available = [f for f in features if f in solar_df.columns]
    X = solar_df[available].fillna(0).tail(1)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_row = shap_values[0] if len(shap_values.shape) > 1 else shap_values

    return _build_explanation_dict(available, shap_row, X.iloc[0], top_n, "Solar")


def _build_explanation_dict(
    feature_names: List[str],
    shap_values: np.ndarray,
    feature_values: pd.Series,
    top_n: int,
    context: str,
) -> Dict:
    """
    Build a structured explanation dictionary from SHAP values.
    Merges cyclical features (hour_sin + hour_cos → Time of Day).
    """
    # Aggregate SHAP values for cyclical feature pairs
    aggregated: Dict[str, float] = {}
    aggregated_values: Dict[str, float] = {}
    merged_features = set()

    pairs = [
        ("hour_sin", "hour_cos", "Time of Day"),
        ("month_sin", "month_cos", "Season"),
        ("day_sin", "day_cos", "Day of Week"),
    ]

    for f1, f2, label in pairs:
        if f1 in feature_names and f2 in feature_names:
            i1 = feature_names.index(f1)
            i2 = feature_names.index(f2)
            combined_shap = float(shap_values[i1]) + float(shap_values[i2])
            aggregated[label] = combined_shap
            merged_features.update([f1, f2])

    for i, fname in enumerate(feature_names):
        if fname in merged_features:
            continue
        plain = FEATURE_PLAIN_NAMES.get(fname, fname)
        if plain in aggregated:
            aggregated[plain] += float(shap_values[i])
        else:
            aggregated[plain] = float(shap_values[i])

    # Sort by absolute SHAP value
    sorted_factors = sorted(
        aggregated.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_n]

    # Build result
    factors = []
    for name, shap_val in sorted_factors:
        direction = "positive" if shap_val > 0 else "negative"
        impact_label = "High" if abs(shap_val) > 10 else ("Medium" if abs(shap_val) > 3 else "Low")
        factors.append({
            "feature": name,
            "shap_value": round(shap_val, 4),
            "direction": direction,
            "impact": impact_label,
            "plain_explanation": _plain_language(name, shap_val, context),
        })

    # Peak cause explanation (officer-friendly)
    peak_explanation = _build_peak_explanation(factors, context)

    return {
        "context": context,
        "top_factors": factors,
        "peak_cause_explanation": peak_explanation,
        "technical_note": "SHAP values represent the marginal contribution of each feature "
                          "to the model prediction relative to the baseline expected output.",
    }


def _plain_language(feature: str, shap_val: float, context: str) -> str:
    """Generate a plain-language explanation sentence for one feature."""
    direction = "increasing" if shap_val > 0 else "reducing"
    templates = {
        "Temperature":            f"High temperature is {direction} {'demand' if context != 'Solar' else 'solar output'}.",
        "Humidity":               f"Elevated humidity is {direction} {'demand' if context != 'Solar' else 'solar generation'}.",
        "Rainfall":               f"Rainfall (cloud cover) is {direction} {'demand' if context != 'Solar' else 'solar generation'}.",
        "Time of Day":            f"Current time of day is {direction} {'demand' if context != 'Solar' else 'solar generation'}.",
        "Season":                 f"Current season is {direction} {'demand' if context != 'Solar' else 'solar output'}.",
        "Day of Week":            f"Day of week pattern is {direction} demand.",
        "Weekend":                f"Weekend effect is {direction} demand.",
        "Holiday":                f"Holiday effect is {direction} demand.",
        "Previous Hour Demand":   f"Recent demand level is {direction} the forecast.",
        "Previous Day Same Hour": f"Previous day's demand at this hour is {direction} today's forecast.",
        "24-Hour Demand Average": f"Average demand over the past 24 hours is {direction} the forecast.",
        "Zone Type":              f"Zone characteristics are {direction} demand.",
        "Previous Hour Solar":    f"Recent solar output is {direction} the solar forecast.",
        "Recent Solar Average":   f"Average solar generation recently is {direction} the forecast.",
    }
    return templates.get(feature, f"{feature} is {direction} the forecast.")


def _build_peak_explanation(factors: List[Dict], context: str) -> str:
    """Compose a concise officer-friendly peak cause explanation."""
    if not factors:
        return "Insufficient data to explain the forecast."

    positive = [f for f in factors if f["direction"] == "positive" and f["impact"] in ("High", "Medium")]
    negative = [f for f in factors if f["direction"] == "negative" and f["impact"] in ("High", "Medium")]

    lines = []
    if positive:
        drivers = ", ".join(f['feature'] for f in positive[:3])
        lines.append(f"Main factors driving {'demand' if context != 'Solar' else 'solar generation'} up: {drivers}.")
    if negative:
        reducers = ", ".join(f['feature'] for f in negative[:2])
        lines.append(f"Factors offsetting {'demand' if context != 'Solar' else 'solar generation'}: {reducers}.")
    if not lines:
        top = factors[0]
        lines.append(top["plain_explanation"])

    return " ".join(lines)


def get_feature_importance(zone: str) -> List[Dict]:
    """
    Return overall feature importance for a zone's XGBoost model.
    Based on XGBoost's built-in feature importance (gain).
    """
    model_path = MODELS_DIR / f"xgboost_demand_baseline_{zone.lower()}.pkl"
    if not model_path.exists():
        logger.warning(f"No XGBoost model for zone '{zone}' — cannot return importance")
        return []

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    importance = model.get_booster().get_score(importance_type="gain")
    total = sum(importance.values()) or 1
    result = []
    for feat, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
        plain = FEATURE_PLAIN_NAMES.get(feat, feat)
        result.append({
            "feature": feat,
            "plain_name": plain,
            "importance": round(score / total * 100, 2),  # as % of total gain
            "raw_gain": round(score, 4),
        })
    return result
