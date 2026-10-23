"""
Database Manager
=================
SQLite-based storage for historical data, predictions,
anomalies, recommendations, and simulation history.

Designed to be extensible for Phase 2 without rewriting.
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from config.settings import DB_PATH

logger = logging.getLogger(__name__)


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_database() -> None:
    """Create all tables if they do not exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at  TEXT NOT NULL,
            zone        TEXT NOT NULL,
            model       TEXT NOT NULL,
            forecast_start TEXT,
            forecast_json  TEXT,
            peak_mw     REAL,
            peak_hour   INTEGER,
            data_type   TEXT DEFAULT 'predicted'
        );

        CREATE TABLE IF NOT EXISTS solar_forecasts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at  TEXT NOT NULL,
            forecast_start TEXT,
            forecast_json  TEXT,
            peak_solar_mw  REAL,
            data_type   TEXT DEFAULT 'predicted'
        );

        CREATE TABLE IF NOT EXISTS anomalies (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            detected_at  TEXT NOT NULL,
            zone         TEXT NOT NULL,
            timestamp    TEXT,
            observed_mw  REAL,
            expected_mw  REAL,
            deviation_pct REAL,
            anomaly_score REAL,
            status       TEXT,
            status_note  TEXT
        );

        CREATE TABLE IF NOT EXISTS recommendations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at  TEXT NOT NULL,
            severity    TEXT,
            zone        TEXT,
            action      TEXT,
            reason      TEXT,
            category    TEXT,
            priority    INTEGER
        );

        CREATE TABLE IF NOT EXISTS simulation_history (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT NOT NULL,
            baseline_peak_mw REAL,
            scenario_inputs  TEXT,
            simulated_peak_mw REAL,
            peak_reduction_mw REAL,
            peak_reduction_pct REAL,
            baseline_severity TEXT,
            simulated_severity TEXT,
            full_result      TEXT
        );

        CREATE TABLE IF NOT EXISTS model_metrics (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            recorded_at TEXT NOT NULL,
            zone       TEXT,
            model_name TEXT NOT NULL,
            mae        REAL,
            rmse       REAL,
            mape       REAL,
            r2         REAL
        );
    """)

    conn.commit()
    conn.close()
    logger.info(f"Database initialised: {DB_PATH}")


def save_prediction(
    zone: str,
    model: str,
    forecast_json: List[float],
    peak_mw: float,
    peak_hour: int,
    forecast_start: str = None,
) -> int:
    """Insert a demand prediction record."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (created_at, zone, model, forecast_start,
                                  forecast_json, peak_mw, peak_hour)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(), zone, model,
        forecast_start or datetime.utcnow().isoformat(),
        json.dumps(forecast_json), peak_mw, peak_hour,
    ))
    row_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return row_id


def save_anomalies(anomaly_list: List[Dict]) -> None:
    """Bulk-insert anomaly records."""
    conn = get_connection()
    cursor = conn.cursor()
    for a in anomaly_list:
        cursor.execute("""
            INSERT INTO anomalies (detected_at, zone, timestamp, observed_mw,
                                   expected_mw, deviation_pct, anomaly_score,
                                   status, status_note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            a.get("zone"), a.get("timestamp"),
            a.get("observed_load_mw"), a.get("expected_load_mw"),
            a.get("deviation", "").replace("%", "").replace("+", "") or None,
            a.get("anomaly_score"),
            a.get("status"), a.get("status_note"),
        ))
    conn.commit()
    conn.close()


def save_model_metrics(zone: str, model_name: str, metrics: Dict) -> None:
    """Persist evaluation metrics for a model."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO model_metrics (recorded_at, zone, model_name, mae, rmse, mape, r2)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(), zone, model_name,
        metrics.get("MAE"), metrics.get("RMSE"),
        metrics.get("MAPE"), metrics.get("R2"),
    ))
    conn.commit()
    conn.close()


def get_latest_predictions(zone: str = None, limit: int = 24) -> List[Dict]:
    """Retrieve the most recent predictions."""
    conn = get_connection()
    cursor = conn.cursor()
    if zone:
        cursor.execute(
            "SELECT * FROM predictions WHERE zone=? ORDER BY created_at DESC LIMIT ?",
            (zone, limit)
        )
    else:
        cursor.execute(
            "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_recent_anomalies(limit: int = 50) -> List[Dict]:
    """Retrieve recent anomaly records."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM anomalies ORDER BY detected_at DESC LIMIT ?", (limit,)
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_simulation_history(limit: int = 20) -> List[Dict]:
    """Retrieve recent simulation history for officer review."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM simulation_history ORDER BY timestamp DESC LIMIT ?", (limit,)
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_model_metrics_all() -> List[Dict]:
    """Retrieve all stored model metrics for comparison."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM model_metrics ORDER BY recorded_at DESC")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows
