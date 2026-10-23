"""
Data Preprocessing Pipeline
============================
Implements a robust, reusable preprocessing pipeline for the
Solar-Aware Demand Forecasting System.

Design principles:
- No random train/test splits (chronological only)
- No data leakage (scalers fit on training data only)
- All preprocessing objects saved for reuse at inference
- Every engineered feature is documented inline
"""

import numpy as np
import pandas as pd
import logging
import pickle
import sys
from pathlib import Path
from typing import Tuple, Dict, Optional

from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from config.settings import (
    PROCESSED_DATA_DIR, MODELS_DIR, ZONES, ZONE_ENCODING,
    INDIAN_HOLIDAYS, TRAIN_RATIO, VALIDATION_RATIO,
    GRU_SEQUENCE_LENGTH
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─── Feature Documentation ────────────────────────────────────────────────────
FEATURE_DOCS = {
    "hour":                   "Hour of day (0–23) — captures diurnal demand cycles",
    "day_of_week":            "Day of week (0=Mon … 6=Sun) — captures weekly patterns",
    "month":                  "Calendar month (1–12) — captures seasonal variation",
    "is_weekend":             "1 if Saturday/Sunday, else 0 — weekend demand differs",
    "is_holiday":             "1 if Indian public holiday, else 0",
    "season":                 "Indian meteorological season (encoded)",
    "temperature_c":          "Ambient temperature (°C) — strong demand driver",
    "humidity_pct":           "Relative humidity (%) — affects AC/cooling load",
    "rainfall_mm":            "Hourly rainfall (mm) — proxy for cloud cover",
    "solar_generation_mw":    "Solar generation (MW) — used for net demand calc",
    "zone_encoded":           "Numerical encoding of distribution zone",
    "lag_1h":                 "Demand 1 hour ago — autoregressive feature",
    "lag_24h":                "Demand 24 hours ago (same hour, previous day)",
    "lag_48h":                "Demand 48 hours ago — two-day pattern",
    "lag_168h":               "Demand 168 hours ago (same hour, previous week)",
    "rolling_mean_6h":        "Rolling mean demand over past 6 hours",
    "rolling_std_6h":         "Rolling std dev demand over past 6 hours",
    "rolling_mean_24h":       "Rolling mean demand over past 24 hours",
    "rolling_max_24h":        "Rolling max demand over past 24 hours",
    "hour_sin":               "Sine encoding of hour — preserves cyclical nature",
    "hour_cos":               "Cosine encoding of hour — preserves cyclical nature",
    "month_sin":              "Sine encoding of month — preserves cyclical nature",
    "month_cos":              "Cosine encoding of month — preserves cyclical nature",
    "day_sin":                "Sine encoding of day-of-week",
    "day_cos":                "Cosine encoding of day-of-week",
    "solar_lag_1h":           "Solar generation 1 hour ago",
    "solar_rolling_mean_6h":  "Rolling mean solar generation over 6 hours",
}


def load_raw_data(filepath: Path) -> pd.DataFrame:
    """
    Load raw CSV data and perform initial validation.
    Raises ValueError with useful messages for common data issues.
    """
    logger.info(f"Loading raw data from: {filepath}")
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Validate required columns
    required_columns = [
        "timestamp", "zone", "demand_mw", "temperature_c",
        "humidity_pct", "solar_generation_mw"
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamp column and sort chronologically."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    invalid = df["timestamp"].isna().sum()
    if invalid > 0:
        logger.warning(f"Dropping {invalid} rows with invalid timestamps")
        df = df.dropna(subset=["timestamp"])

    df.sort_values(["timestamp", "zone"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate (timestamp, zone) combinations, keeping first occurrence."""
    before = len(df)
    df = df.drop_duplicates(subset=["timestamp", "zone"], keep="first")
    removed = before - len(df)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate rows")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values:
    - Numerical columns: forward-fill then backward-fill (time-series safe)
    - Categorical columns: mode fill
    - demand_mw < 0: set to NaN then fill
    """
    df = df.copy()

    # Validate physical constraints
    neg_demand = (df["demand_mw"] < 0).sum()
    if neg_demand > 0:
        logger.warning(f"Found {neg_demand} negative demand values — setting to NaN")
        df.loc[df["demand_mw"] < 0, "demand_mw"] = np.nan

    neg_solar = (df["solar_generation_mw"] < 0).sum()
    if neg_solar > 0:
        logger.warning(f"Found {neg_solar} negative solar values — setting to 0")
        df.loc[df["solar_generation_mw"] < 0, "solar_generation_mw"] = 0.0

    # Fill per zone (avoid cross-zone leakage)
    numerical_cols = ["demand_mw", "temperature_c", "humidity_pct",
                      "rainfall_mm", "solar_generation_mw"]
    for col in numerical_cols:
        if col in df.columns:
            df[col] = df.groupby("zone")[col].transform(
                lambda x: x.ffill().bfill()
            )

    # Remaining NaNs after zone-wise fill
    remaining_na = df[numerical_cols].isna().sum().sum()
    if remaining_na > 0:
        logger.warning(f"Still {remaining_na} NaN values after fill — using column median")
        for col in numerical_cols:
            df[col] = df[col].fillna(df[col].median())

    # Optional rainfall column
    if "rainfall_mm" not in df.columns:
        df["rainfall_mm"] = 0.0

    return df


def handle_outliers(df: pd.DataFrame, iqr_multiplier: float = 3.0) -> pd.DataFrame:
    """
    Cap extreme outliers in demand_mw using IQR method per zone.
    Uses a conservative multiplier (3.0) to preserve legitimate peaks.
    """
    df = df.copy()
    for zone in df["zone"].unique():
        mask = df["zone"] == zone
        q1 = df.loc[mask, "demand_mw"].quantile(0.25)
        q3 = df.loc[mask, "demand_mw"].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        outliers = ((df.loc[mask, "demand_mw"] < lower) |
                    (df.loc[mask, "demand_mw"] > upper)).sum()
        if outliers > 0:
            logger.info(f"Zone {zone}: capping {outliers} outliers "
                        f"(range: [{lower:.1f}, {upper:.1f}] MW)")
        df.loc[mask, "demand_mw"] = df.loc[mask, "demand_mw"].clip(lower, upper)
    return df


def validate_zones(df: pd.DataFrame) -> pd.DataFrame:
    """Validate zone names and drop unknown zones."""
    unknown = df[~df["zone"].isin(ZONES)]
    if len(unknown) > 0:
        logger.warning(f"Dropping {len(unknown)} rows with unknown zones: "
                       f"{unknown['zone'].unique().tolist()}")
        df = df[df["zone"].isin(ZONES)].copy()
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate all forecasting features.
    All lag features are computed per-zone to prevent cross-zone leakage.
    """
    df = df.copy()
    df.sort_values(["zone", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Time-based features ──────────────────────────────────────────────────
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_holiday"] = df.get("is_holiday", 0).astype(int)

    # Season encoding (Indian meteorological)
    season_map = {3: "Summer", 4: "Summer", 5: "Summer",
                  6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
                  10: "PostMonsoon", 11: "PostMonsoon",
                  1: "Winter", 2: "Winter", 12: "Winter"}
    df["season"] = df["month"].map(season_map)
    season_encoding = {"Summer": 0, "Monsoon": 1, "PostMonsoon": 2, "Winter": 3}
    df["season_encoded"] = df["season"].map(season_encoding)

    # ── Cyclical time encoding (preserves periodicity) ────────────────────────
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # ── Zone encoding ─────────────────────────────────────────────────────────
    df["zone_encoded"] = df["zone"].map(ZONE_ENCODING)

    # ── Lag features (computed per zone to avoid cross-zone leakage) ──────────
    for zone in ZONES:
        mask = df["zone"] == zone
        zone_demand = df.loc[mask, "demand_mw"]
        zone_solar = df.loc[mask, "solar_generation_mw"]

        # Demand lag features
        df.loc[mask, "lag_1h"] = zone_demand.shift(1)
        df.loc[mask, "lag_24h"] = zone_demand.shift(24)
        df.loc[mask, "lag_48h"] = zone_demand.shift(48)
        df.loc[mask, "lag_168h"] = zone_demand.shift(168)

        # Rolling statistics (past-only — no look-ahead)
        df.loc[mask, "rolling_mean_6h"] = zone_demand.shift(1).rolling(6).mean()
        df.loc[mask, "rolling_std_6h"] = zone_demand.shift(1).rolling(6).std()
        df.loc[mask, "rolling_mean_24h"] = zone_demand.shift(1).rolling(24).mean()
        df.loc[mask, "rolling_max_24h"] = zone_demand.shift(1).rolling(24).max()

        # Solar lag features
        df.loc[mask, "solar_lag_1h"] = zone_solar.shift(1)
        df.loc[mask, "solar_rolling_mean_6h"] = zone_solar.shift(1).rolling(6).mean()

    # ── Drop rows where lag features are NaN (beginning of series) ────────────
    lag_cols = ["lag_1h", "lag_24h", "lag_48h", "lag_168h",
                "rolling_mean_6h", "rolling_std_6h",
                "rolling_mean_24h", "rolling_max_24h",
                "solar_lag_1h", "solar_rolling_mean_6h"]
    before = len(df)
    df.dropna(subset=lag_cols, inplace=True)
    dropped = before - len(df)
    if dropped > 0:
        logger.info(f"Dropped {dropped} rows with NaN lag features (expected at series start)")

    df.reset_index(drop=True, inplace=True)
    return df


def chronological_split(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train/validation/test using strict chronological order.
    NO random splitting — prevents data leakage in time-series forecasting.
    """
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VALIDATION_RATIO))

    train = df_sorted.iloc[:train_end].copy()
    val = df_sorted.iloc[train_end:val_end].copy()
    test = df_sorted.iloc[val_end:].copy()

    logger.info(f"Chronological split -- Train: {len(train):,} | "
                f"Val: {len(val):,} | Test: {len(test):,}")
    logger.info(f"Train: {train['timestamp'].min()} to {train['timestamp'].max()}")
    logger.info(f"Val:   {val['timestamp'].min()} to {val['timestamp'].max()}")
    logger.info(f"Test:  {test['timestamp'].min()} to {test['timestamp'].max()}")
    return train, val, test


def fit_scalers(
    train_df: pd.DataFrame,
    feature_cols: list
) -> Dict[str, StandardScaler]:
    """
    Fit StandardScalers on training data only.
    Returns dict of {feature_name: scaler}.
    Scalers are fitted on TRAINING data only to prevent data leakage.
    """
    scalers: Dict[str, StandardScaler] = {}
    for col in feature_cols:
        scaler = StandardScaler()
        scaler.fit(train_df[[col]])
        scalers[col] = scaler
    logger.info(f"Fitted scalers for {len(scalers)} features")
    return scalers


def apply_scalers(
    df: pd.DataFrame,
    scalers: Dict[str, StandardScaler],
    suffix: str = "_scaled"
) -> pd.DataFrame:
    """Apply pre-fitted scalers to a dataframe (train/val/test)."""
    df = df.copy()
    for col, scaler in scalers.items():
        if col in df.columns:
            df[f"{col}{suffix}"] = scaler.transform(df[[col]])
    return df


def save_preprocessor(obj: object, name: str) -> Path:
    """Persist a preprocessing object (scaler, encoder) for reuse at inference."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Saved preprocessor: {path}")
    return path


def load_preprocessor(name: str) -> object:
    """Load a saved preprocessing object."""
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Preprocessor not found: {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def run_full_pipeline(raw_filepath: Path) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict
]:
    """
    Run the complete preprocessing pipeline end-to-end.

    Returns:
        train_df, val_df, test_df, pipeline_meta
    """
    logger.info("=== Starting Preprocessing Pipeline ===")

    # Load
    df = load_raw_data(raw_filepath)

    # Clean
    df = parse_timestamps(df)
    df = remove_duplicates(df)
    df = validate_zones(df)
    df = handle_missing_values(df)
    df = handle_outliers(df)

    # Feature engineering
    df = engineer_features(df)

    # Define feature columns for scaling
    numerical_features = [
        "demand_mw", "temperature_c", "humidity_pct", "rainfall_mm",
        "solar_generation_mw", "lag_1h", "lag_24h", "lag_48h", "lag_168h",
        "rolling_mean_6h", "rolling_std_6h", "rolling_mean_24h",
        "rolling_max_24h", "solar_lag_1h", "solar_rolling_mean_6h"
    ]

    # Chronological split (NO random splitting)
    train_df, val_df, test_df = chronological_split(df)

    # Fit scalers on training data only (prevent leakage)
    scalers = fit_scalers(train_df, numerical_features)

    # Apply scalers to all splits
    train_df = apply_scalers(train_df, scalers)
    val_df = apply_scalers(val_df, scalers)
    test_df = apply_scalers(test_df, scalers)

    # Save preprocessor objects
    save_preprocessor(scalers, "demand_scalers")
    save_preprocessor(numerical_features, "numerical_feature_list")

    # Save processed datasets
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DATA_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
    df.to_csv(PROCESSED_DATA_DIR / "full_processed.csv", index=False)

    pipeline_meta = {
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "numerical_features": numerical_features,
        "feature_docs": FEATURE_DOCS,
        "train_end": str(train_df["timestamp"].max()),
        "val_end": str(val_df["timestamp"].max()),
        "test_end": str(test_df["timestamp"].max()),
    }

    logger.info("=== Preprocessing Pipeline Complete ===")
    return train_df, val_df, test_df, pipeline_meta
