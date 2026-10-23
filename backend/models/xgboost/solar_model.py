"""
XGBoost Solar Generation Forecasting Model
============================================
Trains an XGBoost regressor to forecast solar generation (MW)
using weather, temporal, and historical solar features.

Supports:
    - Training on the full dataset (not per-zone; solar is zone-aggregate)
    - Real evaluation metrics (MAE, RMSE, MAPE, R²)
    - SHAP explainability compatible
    - Model saving/loading
"""

import numpy as np
import pandas as pd
import logging
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from config.settings import (
    XGBOOST_N_ESTIMATORS, XGBOOST_MAX_DEPTH,
    XGBOOST_LEARNING_RATE, XGBOOST_SUBSAMPLE,
    XGBOOST_COLSAMPLE_BYTREE, MODELS_DIR, METRICS_DIR
)

logger = logging.getLogger(__name__)

# ─── Features used by the XGBoost solar model ────────────────────────────────
SOLAR_INPUT_FEATURES = [
    "hour",
    "month",
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "temperature_c",
    "humidity_pct",
    "rainfall_mm",
    "solar_lag_1h_scaled",
    "solar_rolling_mean_6h_scaled",
    "is_weekend",
    "is_holiday",
    "season_encoded",
]

SOLAR_TARGET = "solar_generation_mw"


def prepare_solar_data(
    df: pd.DataFrame,
    feature_cols: List[str] = SOLAR_INPUT_FEATURES,
    target_col: str = SOLAR_TARGET,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract features and target for the solar model.
    Deduplicates timestamps (solar is not per-zone).
    """
    # Solar generation is the same across zones for the same timestamp
    # Use the Residential zone rows as the canonical solar source
    if "zone" in df.columns:
        solar_df = df[df["zone"] == "Residential"].copy()
    else:
        solar_df = df.copy()

    solar_df = solar_df.sort_values("timestamp").reset_index(drop=True)

    available = [f for f in feature_cols if f in solar_df.columns]
    missing = [f for f in feature_cols if f not in solar_df.columns]
    if missing:
        logger.warning(f"Solar model: missing features {missing} — skipping them")

    X = solar_df[available]
    y = solar_df[target_col]
    return X, y, available


def train_solar_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict:
    """
    Train XGBoost solar model and evaluate on test set.
    Saves model and evaluation metrics.
    """
    try:
        import xgboost as xgb
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    except ImportError:
        raise ImportError("XGBoost required. pip install xgboost")

    logger.info("Training XGBoost solar generation model...")

    X_train, y_train, features = prepare_solar_data(train_df)
    X_val, y_val, _ = prepare_solar_data(val_df, feature_cols=features)
    X_test, y_test, _ = prepare_solar_data(test_df, feature_cols=features)

    logger.info(f"Solar model — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    logger.info(f"Features: {features}")

    model = xgb.XGBRegressor(
        n_estimators=XGBOOST_N_ESTIMATORS,
        max_depth=XGBOOST_MAX_DEPTH,
        learning_rate=XGBOOST_LEARNING_RATE,
        subsample=XGBOOST_SUBSAMPLE,
        colsample_bytree=XGBOOST_COLSAMPLE_BYTREE,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0.0)   # Solar cannot be negative

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    nonzero = y_test.values != 0
    mape = (np.mean(np.abs((y_test.values[nonzero] - y_pred[nonzero]) /
                            y_test.values[nonzero])) * 100
            if nonzero.sum() > 0 else None)

    metrics = {
        "MAE":  round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "MAPE": round(float(mape), 4) if mape is not None else None,
        "R2":   round(float(r2), 4),
    }
    logger.info(f"Solar model metrics: {metrics}")

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "xgboost_solar.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Save feature list
    feature_path = MODELS_DIR / "xgboost_solar_features.pkl"
    with open(feature_path, "wb") as f:
        pickle.dump(features, f)

    # Save metrics
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = METRICS_DIR / "xgboost_solar_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"model": "XGBoost_Solar", "metrics": metrics}, f, indent=2)

    return {
        "model_path": str(model_path),
        "features": features,
        "metrics": metrics,
    }


def load_solar_model():
    """Load the saved XGBoost solar model."""
    model_path = MODELS_DIR / "xgboost_solar.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Solar model not found: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def predict_solar_24h(
    recent_df: pd.DataFrame,
    scalers: dict = None,
) -> Dict:
    """
    Generate 24-hour solar generation forecast using the saved XGBoost model.

    Args:
        recent_df: Next 24 hours of feature data (with weather forecasts)
        scalers: Optional dict of scalers (not needed for XGBoost but passed for consistency)

    Returns:
        Dict with predicted_solar_mw list, peak_solar_mw, peak_hour
    """
    model = load_solar_model()

    feature_path = MODELS_DIR / "xgboost_solar_features.pkl"
    with open(feature_path, "rb") as f:
        features = pickle.load(f)

    if "zone" in recent_df.columns:
        solar_df = recent_df[recent_df["zone"] == "Residential"].copy()
    else:
        solar_df = recent_df.copy()

    solar_df = solar_df.sort_values("timestamp").head(24)
    available = [f for f in features if f in solar_df.columns]
    X = solar_df[available]

    predictions = model.predict(X)
    predictions = np.maximum(predictions, 0.0)

    peak_idx = int(np.argmax(predictions))

    return {
        "predicted_solar_mw": predictions.tolist(),
        "peak_solar_mw": round(float(predictions[peak_idx]), 2),
        "peak_solar_hour_offset": peak_idx,
    }


def train_xgboost_baseline_for_demand(
    zone: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict:
    """
    Train an XGBoost baseline for demand forecasting (used for model comparison).
    This is a simple one-step-ahead regression (not multi-step).
    Used ONLY for comparison against the GRU model.
    """
    try:
        import xgboost as xgb
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    except ImportError:
        raise ImportError("XGBoost required. pip install xgboost")

    logger.info(f"Training XGBoost demand baseline for zone: {zone}")

    demand_features = [
        "temperature_c", "humidity_pct", "rainfall_mm",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "day_sin", "day_cos", "is_weekend", "is_holiday",
        "lag_1h_scaled", "lag_24h_scaled",
        "rolling_mean_24h_scaled", "zone_encoded",
    ]

    def prep(df, zone):
        z = df[df["zone"] == zone].copy()
        avail = [f for f in demand_features if f in z.columns]
        return z[avail], z["demand_mw"]

    X_train, y_train = prep(train_df, zone)
    X_val, y_val = prep(val_df, zone)
    X_test, y_test = prep(test_df, zone)

    avail_features = X_train.columns.tolist()

    model = xgb.XGBRegressor(
        n_estimators=XGBOOST_N_ESTIMATORS,
        max_depth=XGBOOST_MAX_DEPTH,
        learning_rate=XGBOOST_LEARNING_RATE,
        subsample=XGBOOST_SUBSAMPLE,
        colsample_bytree=XGBOOST_COLSAMPLE_BYTREE,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_pred = model.predict(X_test[avail_features])

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    nonzero = y_test.values != 0
    mape = (np.mean(np.abs((y_test.values[nonzero] - y_pred[nonzero]) /
                            y_test.values[nonzero])) * 100
            if nonzero.sum() > 0 else None)

    metrics = {
        "MAE":  round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "MAPE": round(float(mape), 4) if mape is not None else None,
        "R2":   round(float(r2), 4),
    }

    # Save
    model_path = MODELS_DIR / f"xgboost_demand_baseline_{zone.lower()}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = METRICS_DIR / f"xgboost_baseline_{zone.lower()}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"zone": zone, "model": "XGBoost_Baseline", "metrics": metrics}, f, indent=2)

    logger.info(f"XGBoost baseline {zone}: {metrics}")
    return {"zone": zone, "model": "XGBoost_Baseline", "metrics": metrics}
