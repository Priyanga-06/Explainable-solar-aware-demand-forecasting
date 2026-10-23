"""
Inference Engine
=================
Loads saved models and generates all dashboard-ready outputs
for a given input scenario.

This is the ONLY file that should be called at dashboard/API request time.
Models are loaded from saved files — not retrained on each call.

Output structure:
    {
        zone_forecasts: {Residential, Commercial, Industrial},
        solar_forecast: {...},
        net_demand: {...},
        peak_severity: {...},
        anomalies: [...],
        shap_explanation: {...},
        solar_load_match: {...},
        recommendations: [...],
        model_comparison: {...},
    }
"""

import numpy as np
import pandas as pd
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    MODELS_DIR, ZONES, GRU_SEQUENCE_LENGTH, CONFIDENCE_BAND_FACTOR
)

logger = logging.getLogger(__name__)


def load_scalers() -> dict:
    """Load the fitted scalers saved during training."""
    path = MODELS_DIR / "demand_scalers.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Scalers not found: {path}. "
            "Please run backend/training/train_all.py first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def prepare_inference_window(
    recent_data: pd.DataFrame,
    scalers: dict,
    already_engineered: bool = False,
) -> pd.DataFrame:
    """
    Apply feature engineering and scaling to recent data for inference.
    Uses saved scalers — does NOT refit.
    Skips feature engineering if lag features already present (avoids row loss).

    Args:
        recent_data: Raw or already-engineered dataframe
        scalers: Pre-fitted scalers
        already_engineered: If True, skip engineer_features (data loaded from
                             full_processed.csv which already has all lag features)
    """
    from backend.data.preprocessing.pipeline import engineer_features, apply_scalers

    # Check if lag features already exist (e.g., from processed CSV) — either
    # via the explicit flag or by column presence.
    if already_engineered or "lag_1h" in recent_data.columns:
        df = recent_data.copy()
    else:
        df = engineer_features(recent_data)

    # Apply scalers only if scaled columns are not already present
    if "demand_mw_scaled" not in df.columns:
        df = apply_scalers(df, scalers)

    return df


def run_full_inference(
    recent_df: pd.DataFrame,
    current_hour: int = None,
) -> Dict:
    """
    Run all inference modules and return a complete dashboard payload.

    Args:
        recent_df: Preprocessed dataframe with at least GRU_SEQUENCE_LENGTH
                   rows per zone (recent historical window)
        current_hour: Current hour for solar-load matching (default: now)

    Returns:
        Dict with all dashboard-ready outputs, clearly typed by data_type
    """
    if current_hour is None:
        current_hour = datetime.now().hour

    logger.info("Running inference pipeline...")

    scalers = load_scalers()
    # If the dataframe already has lag features (loaded from full_processed.csv),
    # skip re-engineering to avoid the NaN-drop problem on short windows.
    already_engineered = "rolling_mean_24h" in recent_df.columns and "lag_24h" in recent_df.columns
    processed_df = prepare_inference_window(recent_df, scalers, already_engineered=already_engineered)

    # Generate future timestamps for forecast window
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    forecast_timestamps = [
        (now + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M") for i in range(24)
    ]
    forecast_hours = [(current_hour + i) % 24 for i in range(24)]

    # ── Zone-wise GRU demand forecasts ───────────────────────────────────────
    zone_forecasts = {}
    total_demand_24h = np.zeros(24)

    for zone in ZONES:
        try:
            from backend.models.gru.model import predict_24h
            forecast = predict_24h(zone, processed_df, scalers)
            zone_forecasts[zone] = forecast
            zone_demand_arr = np.array(forecast["predicted_demand_mw"])
            total_demand_24h += zone_demand_arr
        except ImportError:
            # TensorFlow not available — use XGBoost multi-step fallback
            logger.info(f"TensorFlow unavailable — using XGBoost fallback for zone {zone}")
            try:
                from backend.models.gru.fallback_model import predict_24h_fallback
                zone_forecast = predict_24h_fallback(zone, processed_df)
                zone_forecasts[zone] = zone_forecast
                total_demand_24h += np.array(zone_forecast["predicted_demand_mw"])
            except Exception as e2:
                logger.error(f"Fallback also failed for zone {zone}: {e2}")
        except Exception as e:
            logger.warning(f"GRU forecast failed for zone {zone}: {e}")
            # Try XGBoost multi-step fallback next
            try:
                from backend.models.gru.fallback_model import predict_24h_fallback
                zone_forecast = predict_24h_fallback(zone, processed_df)
                zone_forecasts[zone] = zone_forecast
                total_demand_24h += np.array(zone_forecast["predicted_demand_mw"])
            except Exception as e2:
                logger.warning(f"Fallback also failed for zone {zone}: {e2}")
                # Final fallback to XGBoost baseline (1-step)
                try:
                    zone_forecast = _xgboost_demand_fallback(zone, processed_df)
                    zone_forecasts[zone] = zone_forecast
                    total_demand_24h += np.array(zone_forecast["predicted_demand_mw"])
                except Exception as e3:
                    logger.error(f"All demand fallbacks failed for zone {zone}: {e3}")

    total_demand_list = total_demand_24h.tolist()

    # ── Solar forecast ────────────────────────────────────────────────────────
    try:
        from backend.models.xgboost.solar_model import predict_solar_24h

        # Build a 24-hour future feature dataframe for solar prediction
        future_df = _build_future_feature_df(processed_df, forecast_hours, forecast_timestamps)
        solar_result = predict_solar_24h(future_df, scalers)
        solar_24h = solar_result["predicted_solar_mw"]
    except Exception as e:
        logger.warning(f"Solar forecast failed: {e}")
        solar_24h = [0.0] * 24
        solar_result = {"predicted_solar_mw": solar_24h, "peak_solar_mw": 0.0}

    # ── Net grid demand ───────────────────────────────────────────────────────
    from backend.forecasting.net_demand import calculate_net_demand, calculate_solar_load_match
    net_demand_24h = calculate_net_demand(total_demand_list, solar_24h)
    solar_load_match = calculate_solar_load_match(
        total_demand_list, solar_24h,
        hour_offsets=forecast_hours,
    )

    # ── Peak severity ─────────────────────────────────────────────────────────
    from backend.severity.peak_severity import get_peak_severity
    severity_result = get_peak_severity(
        total_demand_list, solar_24h, timestamps=forecast_timestamps
    )

    # ── Anomaly detection ─────────────────────────────────────────────────────
    from backend.anomaly.detector import detect_anomalies, get_anomaly_summary
    try:
        anomaly_df = detect_anomalies(processed_df)
        anomaly_summary = get_anomaly_summary(anomaly_df)
        recent_anomalies = anomaly_summary[:10]  # Top 10 most anomalous
    except Exception as e:
        logger.warning(f"Anomaly detection failed: {e}")
        recent_anomalies = []

    # ── SHAP explanation ──────────────────────────────────────────────────────
    shap_explanations = {}
    for zone in ZONES:
        try:
            from backend.explainability.shap_explainer import explain_xgboost_demand
            shap_exp = explain_xgboost_demand(zone, processed_df)
            shap_explanations[zone] = shap_exp
        except Exception as e:
            logger.warning(f"SHAP explanation failed for zone {zone}: {e}")
            shap_explanations[zone] = {"error": str(e)}

    # Combined peak cause explanation (use highest-demand zone)
    peak_zone = max(zone_forecasts, key=lambda z: zone_forecasts[z].get("peak_mw", 0))
    peak_explanation = shap_explanations.get(peak_zone, {})

    # ── Recommendations ───────────────────────────────────────────────────────
    from backend.recommendations.engine import generate_recommendations
    zone_peak_demands = {
        zone: float(zone_forecasts[zone].get("peak_mw", 0))
        for zone in zone_forecasts
    }
    recommendations = generate_recommendations(
        severity_level=severity_result["severity"]["level"],
        peak_hour_index=severity_result["peak_hour_index"],
        peak_net_demand_mw=severity_result["peak_net_demand_mw"],
        current_hour=current_hour,
        solar_generation_mw=solar_result.get("peak_solar_mw", 0),
        zone_demands=zone_peak_demands,
        anomalies=recent_anomalies,
        solar_load_match_pct=solar_load_match.get("match_pct"),
    )

    # ── Model comparison ──────────────────────────────────────────────────────
    from backend.evaluation.model_comparison import get_full_comparison
    model_comparison = get_full_comparison()

    # ── Forecast confidence ───────────────────────────────────────────────────
    # Confidence band is labelled as "Estimated Range" — not a stat interval
    peak_demand = max(total_demand_list) if total_demand_list else 0
    confidence = {
        "expected_peak_mw": round(peak_demand, 2),
        "estimated_lower_mw": round(peak_demand * (1 - CONFIDENCE_BAND_FACTOR), 2),
        "estimated_upper_mw": round(peak_demand * (1 + CONFIDENCE_BAND_FACTOR), 2),
        "label": "Estimated Range",
        "note": (
            "Range is ±7.5% of the predicted peak. "
            "This is an approximate uncertainty estimate, "
            "not a statistically derived confidence interval."
        ),
    }

    # ── Assemble final payload ────────────────────────────────────────────────
    payload = {
        "data_type": "predicted",
        "generated_at": datetime.utcnow().isoformat(),
        "forecast_timestamps": forecast_timestamps,
        "forecast_hours": forecast_hours,
        "zone_forecasts": zone_forecasts,
        "total_demand_24h": [round(v, 2) for v in total_demand_list],
        "solar_forecast": solar_result,
        "solar_24h": [round(v, 2) for v in solar_24h],
        "net_demand_24h": net_demand_24h,
        "solar_load_match": solar_load_match,
        "peak_severity": severity_result,
        "forecast_confidence": confidence,
        "anomalies": recent_anomalies,
        "shap_explanations": shap_explanations,
        "peak_cause_explanation": peak_explanation,
        "recommendations": recommendations,
        "model_comparison": model_comparison,
        "kpi": {
            "current_load_mw": round(
                processed_df["demand_mw"].tail(len(ZONES)).sum(), 2
            ),
            "next_hour_demand_mw": round(
                sum(zone_forecasts[z]["predicted_demand_mw"][0]
                    for z in zone_forecasts if "predicted_demand_mw" in zone_forecasts[z]),
                2
            ),
            "peak_24h_mw": round(max(total_demand_list) if total_demand_list else 0, 2),
            "solar_generation_mw": round(solar_24h[0] if solar_24h else 0, 2),
            "net_grid_demand_mw": round(net_demand_24h[0] if net_demand_24h else 0, 2),
            "severity": severity_result["severity"]["level"],
        },
    }

    return payload


def _xgboost_demand_fallback(zone: str, processed_df: pd.DataFrame) -> Dict:
    """Fallback: use XGBoost baseline for demand forecast if GRU unavailable."""
    import xgboost as xgb

    model_path = MODELS_DIR / f"xgboost_demand_baseline_{zone.lower()}.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    demand_features = [
        "temperature_c", "humidity_pct", "rainfall_mm",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "day_sin", "day_cos", "is_weekend", "is_holiday",
        "lag_1h_scaled", "lag_24h_scaled",
        "rolling_mean_24h_scaled", "zone_encoded",
    ]
    zone_df = processed_df[processed_df["zone"] == zone].tail(24)
    available = [f for f in demand_features if f in zone_df.columns]
    X = zone_df[available]
    pred = model.predict(X)
    pred = np.maximum(pred, 0)
    peak_idx = int(np.argmax(pred))
    band = CONFIDENCE_BAND_FACTOR * pred
    return {
        "zone": zone,
        "predicted_demand_mw": pred.tolist(),
        "peak_mw": round(float(pred[peak_idx]), 2),
        "peak_hour_offset": peak_idx,
        "confidence_lower_mw": (pred - band).tolist(),
        "confidence_upper_mw": (pred + band).tolist(),
        "model_used": "XGBoost_Baseline_Fallback",
    }


def _build_future_feature_df(
    processed_df: pd.DataFrame,
    forecast_hours: List[int],
    forecast_timestamps: List[str],
) -> pd.DataFrame:
    """
    Build a feature dataframe for the next 24 hours using the last
    known weather/state values from processed_df.

    For each future hour we carry forward the last known weather state
    and update all hour-dependent features correctly.
    """
    # Use Residential zone rows as the canonical reference (solar is zone-agnostic)
    ref_df = processed_df[processed_df["zone"] == "Residential"].sort_values("timestamp")
    if ref_df.empty:
        ref_df = processed_df.sort_values("timestamp")

    last_row = ref_df.tail(1).iloc[0].to_dict()

    rows = []
    for i, (hour, ts) in enumerate(zip(forecast_hours, forecast_timestamps)):
        row = last_row.copy()
        ts_dt = pd.Timestamp(ts)
        row["timestamp"] = ts_dt
        row["zone"] = "Residential"
        row["hour"] = hour
        row["month"] = ts_dt.month
        row["day_of_week"] = ts_dt.dayofweek
        row["is_weekend"] = int(ts_dt.dayofweek >= 5)

        # Update cyclical encodings for the actual future hour/month
        row["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        row["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        row["month_sin"] = np.sin(2 * np.pi * ts_dt.month / 12)
        row["month_cos"] = np.cos(2 * np.pi * ts_dt.month / 12)
        row["day_sin"] = np.sin(2 * np.pi * ts_dt.dayofweek / 7)
        row["day_cos"] = np.cos(2 * np.pi * ts_dt.dayofweek / 7)

        # Solar lag features: carry forward from last known solar values
        # (these will be overwritten if the model uses them; using last-known is safe)
        rows.append(row)

    return pd.DataFrame(rows)
