"""
Anomaly Detection — Isolation Forest
======================================
Detects unusual consumption patterns by zone using Isolation Forest.

IMPORTANT:
- Anomalies are labelled as "Possible abnormal consumption / possible non-technical loss"
- They are NOT automatically labelled as electricity theft
- Ground-truth theft data would be required to make that determination
"""

import numpy as np
import pandas as pd
import logging
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    ISOLATION_FOREST_CONTAMINATION,
    ISOLATION_FOREST_N_ESTIMATORS,
    ANOMALY_DEVIATION_THRESHOLD,
    MODELS_DIR, ZONES
)

logger = logging.getLogger(__name__)

# Features used by Isolation Forest
ANOMALY_FEATURES = [
    "demand_mw",
    "temperature_c",
    "humidity_pct",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "rolling_mean_24h",
    "lag_24h",
]


def train_isolation_forest(
    train_df: pd.DataFrame,
) -> Dict[str, object]:
    """
    Train one Isolation Forest model per zone.
    Saves each model for inference.

    Returns dict of {zone: model}
    """
    from sklearn.ensemble import IsolationForest

    models = {}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for zone in ZONES:
        logger.info(f"Training Isolation Forest for zone: {zone}")
        zone_df = train_df[train_df["zone"] == zone].copy()

        available = [f for f in ANOMALY_FEATURES if f in zone_df.columns]
        X = zone_df[available].fillna(zone_df[available].median())

        model = IsolationForest(
            n_estimators=ISOLATION_FOREST_N_ESTIMATORS,
            contamination=ISOLATION_FOREST_CONTAMINATION,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X)

        # Save model
        model_path = MODELS_DIR / f"isolation_forest_{zone.lower()}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "features": available}, f)

        models[zone] = model
        logger.info(f"  Saved: {model_path}")

    return models


def load_anomaly_model(zone: str) -> Dict:
    """Load saved Isolation Forest model for a zone."""
    path = MODELS_DIR / f"isolation_forest_{zone.lower()}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Anomaly model not found for zone '{zone}': {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def detect_anomalies(
    df: pd.DataFrame,
    zone: str = None,
) -> pd.DataFrame:
    """
    Run Isolation Forest anomaly detection on a dataframe.

    If zone is provided, runs only for that zone.
    Returns dataframe with added columns:
        anomaly_score, is_anomaly, deviation_pct, status, status_note
    """
    results = []

    zones_to_check = [zone] if zone else ZONES

    for z in zones_to_check:
        if z not in df["zone"].unique():
            logger.warning(f"Zone '{z}' not found in data — skipping")
            continue

        try:
            model_data = load_anomaly_model(z)
        except FileNotFoundError:
            logger.warning(f"No anomaly model for zone '{z}' — skipping")
            continue

        model = model_data["model"]
        features = model_data["features"]

        zone_df = df[df["zone"] == z].copy().reset_index(drop=True)
        available = [f for f in features if f in zone_df.columns]
        X = zone_df[available].fillna(zone_df[available].median())

        # Isolation Forest: -1 = anomaly, 1 = normal
        predictions = model.predict(X)
        scores = model.score_samples(X)   # More negative = more anomalous

        zone_df["anomaly_score"] = np.round(-scores, 4)   # Flip sign for intuition
        zone_df["is_anomaly"] = (predictions == -1).astype(int)

        # Calculate deviation from rolling baseline
        if "rolling_mean_24h" in zone_df.columns and "demand_mw" in zone_df.columns:
            baseline = zone_df["rolling_mean_24h"].replace(0, np.nan)
            zone_df["deviation_pct"] = (
                (zone_df["demand_mw"] - baseline) / baseline * 100
            ).round(2)
        else:
            zone_df["deviation_pct"] = np.nan

        # Status classification
        def classify_status(row):
            if row["is_anomaly"] == 1:
                dev = row.get("deviation_pct", 0)
                if pd.isna(dev):
                    return "Anomaly", "Possible abnormal consumption / possible non-technical loss"
                direction = f"+{dev:.1f}%" if dev >= 0 else f"{dev:.1f}%"
                return ("Anomaly",
                        f"Possible abnormal consumption / possible non-technical loss "
                        f"({direction} deviation from expected)")
            return "Normal", "Within expected range"

        statuses = zone_df.apply(classify_status, axis=1)
        zone_df["status"] = [s[0] for s in statuses]
        zone_df["status_note"] = [s[1] for s in statuses]

        results.append(zone_df)

    if results:
        return pd.concat(results, ignore_index=True)
    return df.copy()


def get_anomaly_summary(anomaly_df: pd.DataFrame) -> List[Dict]:
    """
    Return a list of detected anomalies with officer-readable details.
    """
    anomalies = anomaly_df[anomaly_df["is_anomaly"] == 1].copy()
    anomalies = anomalies.sort_values("anomaly_score", ascending=False)

    summary = []
    for _, row in anomalies.iterrows():
        ts = row.get("timestamp", "Unknown")
        if hasattr(ts, "strftime"):
            ts_str = ts.strftime("%Y-%m-%d %I:%M %p")
        else:
            ts_str = str(ts)

        dev = row.get("deviation_pct", None)
        dev_str = f"{dev:+.1f}%" if (dev is not None and not pd.isna(dev)) else "N/A"

        summary.append({
            "zone": row.get("zone", "Unknown"),
            "timestamp": ts_str,
            "observed_load_mw": round(float(row.get("demand_mw", 0)), 2),
            "expected_load_mw": round(float(row.get("rolling_mean_24h", 0)), 2)
                                if not pd.isna(row.get("rolling_mean_24h", np.nan)) else None,
            "deviation": dev_str,
            "anomaly_score": round(float(row.get("anomaly_score", 0)), 4),
            "status": row.get("status", "Anomaly"),
            "status_note": row.get("status_note", "Possible abnormal consumption"),
        })

    return summary
