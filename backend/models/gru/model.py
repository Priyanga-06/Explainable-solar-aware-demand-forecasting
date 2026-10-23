"""
GRU Demand Forecasting Model
==============================
Implements a GRU-based neural network for zone-wise 24-hour
electricity demand forecasting.

Architecture:
    Input → GRU Layer 1 → Dropout → GRU Layer 2 → Dropout
          → Dense → Output (24-hour forecast)

Supports:
    - Per-zone training (Residential / Commercial / Industrial)
    - Multi-step 24-hour ahead forecasting
    - Real evaluation metrics (MAE, RMSE, MAPE, R²)
    - Model saving/loading (no retraining on dashboard load)
"""

import numpy as np
import pandas as pd
import logging
import json
import pickle
import sys
from pathlib import Path
from typing import Tuple, Dict, Optional, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from config.settings import (
    GRU_SEQUENCE_LENGTH, GRU_FORECAST_HORIZON,
    GRU_HIDDEN_UNITS, GRU_DROPOUT, GRU_EPOCHS,
    GRU_BATCH_SIZE, GRU_LEARNING_RATE,
    MODELS_DIR, METRICS_DIR, ZONES
)

logger = logging.getLogger(__name__)

# ─── GRU features used as model input ────────────────────────────────────────
GRU_INPUT_FEATURES = [
    "demand_mw_scaled",
    "temperature_c_scaled",
    "humidity_pct_scaled",
    "solar_generation_mw_scaled",
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "day_sin", "day_cos",
    "is_weekend", "is_holiday",
    "lag_1h_scaled", "lag_24h_scaled",
    "rolling_mean_24h_scaled",
    "zone_encoded",
]


def build_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "demand_mw_scaled",
    seq_len: int = GRU_SEQUENCE_LENGTH,
    horizon: int = GRU_FORECAST_HORIZON,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) sequences for the GRU model.

    X shape: (n_samples, seq_len, n_features)
    y shape: (n_samples, horizon)

    Only uses past values to predict future values — no look-ahead.
    """
    available = [c for c in feature_cols if c in df.columns]
    data = df[available].values
    target_idx = df.columns.get_loc(target_col) if target_col in df.columns else None

    # Rebuild from available columns
    feat_data = df[available].values
    if target_col in df.columns:
        tgt_data = df[target_col].values
    else:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    X_list, y_list = [], []
    n = len(feat_data)
    for i in range(seq_len, n - horizon + 1):
        X_list.append(feat_data[i - seq_len:i])
        y_list.append(tgt_data[i:i + horizon])

    if not X_list:
        raise ValueError("Not enough data to build sequences. "
                         f"Need at least {seq_len + horizon} rows.")

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def build_gru_model(
    n_features: int,
    seq_len: int = GRU_SEQUENCE_LENGTH,
    horizon: int = GRU_FORECAST_HORIZON,
    hidden_units: int = GRU_HIDDEN_UNITS,
    dropout: float = GRU_DROPOUT,
    learning_rate: float = GRU_LEARNING_RATE,
):
    """
    Build a two-layer GRU model for multi-step forecasting.
    Returns a compiled Keras model.
    """
    # Import TensorFlow/Keras here to avoid loading at import time
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        raise ImportError(
            "TensorFlow is required for GRU model. "
            "Install it with: pip install tensorflow"
        )

    tf.random.set_seed(42)

    model = keras.Sequential([
        keras.layers.Input(shape=(seq_len, n_features)),
        keras.layers.GRU(hidden_units, return_sequences=True,
                         dropout=dropout, recurrent_dropout=0.1),
        keras.layers.GRU(hidden_units // 2, return_sequences=False,
                         dropout=dropout),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(horizon),
    ], name="GRU_DemandForecaster")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )
    logger.info(f"GRU model built: {model.count_params():,} parameters")
    return model


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scaler=None,
    target_col: str = "demand_mw"
) -> Dict[str, float]:
    """
    Calculate real evaluation metrics on inverse-transformed predictions.
    Metrics:
        MAE   — Mean Absolute Error (MW)
        RMSE  — Root Mean Squared Error (MW)
        MAPE  — Mean Absolute Percentage Error (%)
        R²    — Coefficient of Determination
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Inverse transform if scaler provided
    if scaler is not None:
        y_true_mw = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred_mw = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    else:
        y_true_mw = y_true.flatten()
        y_pred_mw = y_pred.flatten()

    mae = mean_absolute_error(y_true_mw, y_pred_mw)
    rmse = np.sqrt(mean_squared_error(y_true_mw, y_pred_mw))
    r2 = r2_score(y_true_mw, y_pred_mw)

    # MAPE — guard against zero division
    nonzero = y_true_mw != 0
    mape = np.mean(np.abs((y_true_mw[nonzero] - y_pred_mw[nonzero]) /
                           y_true_mw[nonzero])) * 100 if nonzero.sum() > 0 else np.nan

    metrics = {
        "MAE":  round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "MAPE": round(float(mape), 4) if not np.isnan(mape) else None,
        "R2":   round(float(r2), 4),
    }
    return metrics


def train_gru_for_zone(
    zone: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scalers: dict,
) -> Dict:
    """
    Train the GRU model for a single zone and evaluate on the test set.
    Saves the model and metrics.

    Returns a dict with training history and evaluation metrics.
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        raise ImportError("TensorFlow is required. pip install tensorflow")

    logger.info(f"Training GRU for zone: {zone}")

    # Filter by zone
    train_z = train_df[train_df["zone"] == zone].copy().reset_index(drop=True)
    val_z = val_df[val_df["zone"] == zone].copy().reset_index(drop=True)
    test_z = test_df[test_df["zone"] == zone].copy().reset_index(drop=True)

    # Available features for this run
    available_features = [f for f in GRU_INPUT_FEATURES if f in train_z.columns]
    logger.info(f"Using {len(available_features)} features for zone {zone}")

    # Build sequences
    X_train, y_train = build_sequences(train_z, available_features)
    X_val, y_val = build_sequences(val_z, available_features)
    X_test, y_test = build_sequences(test_z, available_features)

    logger.info(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")

    # Build model
    model = build_gru_model(
        n_features=len(available_features),
        seq_len=GRU_SEQUENCE_LENGTH,
        horizon=GRU_FORECAST_HORIZON,
    )

    # Callbacks
    model_save_path = MODELS_DIR / f"gru_{zone.lower()}.keras"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_save_path),
            save_best_only=True,
            monitor="val_loss",
        ),
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=GRU_EPOCHS,
        batch_size=GRU_BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on test set
    y_pred_scaled = model.predict(X_test, verbose=0)
    demand_scaler = scalers.get("demand_mw")
    metrics = evaluate_predictions(y_test, y_pred_scaled, scaler=demand_scaler)
    logger.info(f"Zone {zone} GRU metrics: {metrics}")

    # Save metrics
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = METRICS_DIR / f"gru_{zone.lower()}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"zone": zone, "model": "GRU", "metrics": metrics}, f, indent=2)

    # Save feature list
    feature_path = MODELS_DIR / f"gru_{zone.lower()}_features.pkl"
    with open(feature_path, "wb") as f:
        pickle.dump(available_features, f)

    return {
        "zone": zone,
        "model_path": str(model_save_path),
        "metrics": metrics,
        "history": {
            "train_loss": history.history["loss"],
            "val_loss": history.history["val_loss"],
        },
    }


def train_all_zones(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scalers: dict,
) -> Dict[str, Dict]:
    """Train GRU models for all three zones."""
    results = {}
    for zone in ZONES:
        try:
            result = train_gru_for_zone(zone, train_df, val_df, test_df, scalers)
            results[zone] = result
        except Exception as e:
            logger.error(f"GRU training failed for zone {zone}: {e}")
            results[zone] = {"zone": zone, "error": str(e)}
    return results


def load_gru_model(zone: str):
    """Load a saved GRU model for inference."""
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow required. pip install tensorflow")

    model_path = MODELS_DIR / f"gru_{zone.lower()}.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"GRU model not found for zone '{zone}': {model_path}")
    model = tf.keras.models.load_model(str(model_path))
    logger.info(f"Loaded GRU model for zone: {zone}")
    return model


def predict_24h(
    zone: str,
    recent_df: pd.DataFrame,
    scalers: dict,
) -> Dict:
    """
    Generate 24-hour demand forecast for a zone using the saved GRU model.

    Args:
        zone: One of Residential / Commercial / Industrial
        recent_df: At least GRU_SEQUENCE_LENGTH rows of processed zone data
        scalers: Dict of fitted scalers

    Returns:
        Dict with predicted_demand_mw, peak_mw, peak_hour,
              confidence_lower, confidence_upper
    """
    from config.settings import CONFIDENCE_BAND_FACTOR

    model = load_gru_model(zone)

    feature_path = MODELS_DIR / f"gru_{zone.lower()}_features.pkl"
    with open(feature_path, "rb") as f:
        features = pickle.load(f)

    zone_df = recent_df[recent_df["zone"] == zone].copy()
    zone_df = zone_df.sort_values("timestamp").tail(GRU_SEQUENCE_LENGTH)

    if len(zone_df) < GRU_SEQUENCE_LENGTH:
        raise ValueError(
            f"Need at least {GRU_SEQUENCE_LENGTH} rows for zone '{zone}', "
            f"got {len(zone_df)}"
        )

    available = [f for f in features if f in zone_df.columns]
    X = zone_df[available].values[-GRU_SEQUENCE_LENGTH:].reshape(
        1, GRU_SEQUENCE_LENGTH, len(available)
    ).astype(np.float32)

    y_scaled = model.predict(X, verbose=0)[0]  # shape: (24,)

    # Inverse transform
    demand_scaler = scalers.get("demand_mw")
    if demand_scaler is not None:
        predicted_mw = demand_scaler.inverse_transform(
            y_scaled.reshape(-1, 1)
        ).flatten()
    else:
        predicted_mw = y_scaled

    predicted_mw = np.maximum(predicted_mw, 0.0)

    peak_idx = int(np.argmax(predicted_mw))
    peak_mw = float(predicted_mw[peak_idx])
    peak_hour = peak_idx  # Offset from current hour

    # Estimated range — labelled as approximate in UI
    band = CONFIDENCE_BAND_FACTOR * predicted_mw
    lower = predicted_mw - band
    upper = predicted_mw + band

    return {
        "zone": zone,
        "predicted_demand_mw": predicted_mw.tolist(),
        "peak_mw": round(peak_mw, 2),
        "peak_hour_offset": peak_hour,
        "confidence_lower_mw": lower.tolist(),
        "confidence_upper_mw": upper.tolist(),
        "confidence_note": "Estimated range (±7.5% of prediction). "
                           "Not a statistically derived confidence interval.",
    }
