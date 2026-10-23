"""
Global configuration settings for the Solar-Aware Demand Forecasting System.
All thresholds, paths, and constants are stored here for easy modification.
"""

import os
from pathlib import Path

# ─── Base Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "backend" / "models" / "saved"
METRICS_DIR = BASE_DIR / "backend" / "evaluation" / "metrics"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"
DB_PATH = BASE_DIR / "data" / "database" / "solar_forecast.db"

# ─── Zones ────────────────────────────────────────────────────────────────────
ZONES = ["Residential", "Commercial", "Industrial"]
ZONE_ENCODING = {"Residential": 0, "Commercial": 1, "Industrial": 2}

# ─── Peak Severity Thresholds (MW — configurable) ────────────────────────────
# These thresholds apply to net grid demand (after solar offset)
SEVERITY_THRESHOLDS = {
    "Normal":   (0,    400),    # 0–400 MW
    "Medium":   (400,  600),    # 400–600 MW
    "High":     (600,  800),    # 600–800 MW
    "Critical": (800,  99999),  # > 800 MW
}

# ─── GRU Model Settings ───────────────────────────────────────────────────────
GRU_SEQUENCE_LENGTH = 24       # Hours of history used as input window
GRU_FORECAST_HORIZON = 24      # Hours ahead to forecast
GRU_HIDDEN_UNITS = 64
GRU_DROPOUT = 0.2
GRU_EPOCHS = 50
GRU_BATCH_SIZE = 32
GRU_LEARNING_RATE = 0.001

# ─── XGBoost Solar Model Settings ────────────────────────────────────────────
XGBOOST_N_ESTIMATORS = 200
XGBOOST_MAX_DEPTH = 6
XGBOOST_LEARNING_RATE = 0.05
XGBOOST_SUBSAMPLE = 0.8
XGBOOST_COLSAMPLE_BYTREE = 0.8

# ─── Isolation Forest Settings ────────────────────────────────────────────────
ISOLATION_FOREST_CONTAMINATION = 0.05   # Expected fraction of anomalies
ISOLATION_FOREST_N_ESTIMATORS = 100
ANOMALY_DEVIATION_THRESHOLD = 0.20     # Flag if load deviates > 20% from expected

# ─── Forecast Confidence ──────────────────────────────────────────────────────
# Approximate uncertainty band — labelled as "Estimated Range" in UI
CONFIDENCE_BAND_FACTOR = 0.075         # ±7.5% of predicted value

# ─── Solar-Load Matching ──────────────────────────────────────────────────────
# Daytime hours used for solar-load matching calculation (inclusive)
SOLAR_DAYTIME_START_HOUR = 6
SOLAR_DAYTIME_END_HOUR = 18

# ─── What-If Simulator Limits ────────────────────────────────────────────────
SIMULATOR_LIMITS = {
    "industrial_shift_pct":   {"min": 0, "max": 30, "default": 10},
    "commercial_shift_pct":   {"min": 0, "max": 20, "default": 5},
    "residential_reduction_pct": {"min": 0, "max": 15, "default": 3},
    "solar_increase_pct":     {"min": 0, "max": 20, "default": 5},
}

# ─── Data Split Ratios (chronological) ───────────────────────────────────────
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15   # Remaining

# ─── Indian Holidays (yyyy-mm-dd) ─────────────────────────────────────────────
# Representative set — extend as needed
INDIAN_HOLIDAYS = [
    "2023-01-26",  # Republic Day
    "2023-03-08",  # Holi
    "2023-04-07",  # Good Friday
    "2023-04-14",  # Ambedkar Jayanti
    "2023-04-22",  # Eid ul-Fitr (approx)
    "2023-06-29",  # Eid ul-Adha (approx)
    "2023-08-15",  # Independence Day
    "2023-09-19",  # Ganesh Chaturthi (approx)
    "2023-10-02",  # Gandhi Jayanti
    "2023-10-24",  # Dussehra (approx)
    "2023-11-13",  # Diwali (approx)
    "2023-12-25",  # Christmas
    "2024-01-26",
    "2024-03-25",
    "2024-04-14",
    "2024-08-15",
    "2024-10-02",
    "2024-11-01",
    "2024-12-25",
]

# ─── API Settings ─────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_DEBUG = os.getenv("API_DEBUG", "false").lower() == "true"

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
