"""
Synthetic Data Generator
========================
Generates structurally realistic synthetic electricity-distribution data
representative of Indian power distribution contexts.

IMPORTANT: This data is synthetic/demo data.
It is NOT real Tata Power operational data.
It is clearly labelled as synthetic in the output file metadata.

Usage:
    python data/generate_synthetic_data.py
"""

import numpy as np
import pandas as pd
import logging
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    RAW_DATA_DIR, ZONES, INDIAN_HOLIDAYS, SOLAR_DAYTIME_START_HOUR,
    SOLAR_DAYTIME_END_HOUR
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─── Random seed for reproducibility ─────────────────────────────────────────
RNG = np.random.default_rng(seed=42)


def _is_holiday(date: pd.Timestamp) -> int:
    """Return 1 if the date is a known Indian holiday, else 0."""
    return int(date.strftime("%Y-%m-%d") in INDIAN_HOLIDAYS)


def _season(month: int) -> str:
    """Map calendar month to Indian meteorological season."""
    if month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    elif month in [10, 11]:
        return "Post-Monsoon"
    else:
        return "Winter"


def _temperature(hour: int, month: int) -> float:
    """
    Simulate hourly temperature (°C) for an Indian distribution city.
    Base temperature varies by month; diurnal swing added on top.
    """
    month_base = {
        1: 18, 2: 21, 3: 27, 4: 33, 5: 38,
        6: 34, 7: 30, 8: 29, 9: 29, 10: 28,
        11: 24, 12: 19
    }
    base = month_base[month]
    # Diurnal pattern: min at 5 AM, max at 2 PM
    diurnal = -4 * np.cos(2 * np.pi * (hour - 14) / 24)
    noise = RNG.normal(0, 1.2)
    return round(base + diurnal + noise, 2)


def _humidity(hour: int, month: int, temperature: float) -> float:
    """Simulate relative humidity (%) — inverse correlation with temperature."""
    month_base_humidity = {
        1: 65, 2: 55, 3: 45, 4: 35, 5: 30,
        6: 75, 7: 85, 8: 88, 9: 82, 10: 65,
        11: 55, 12: 60
    }
    base = month_base_humidity[month]
    # Early morning slightly higher humidity
    diurnal = 5 * np.cos(2 * np.pi * (hour - 6) / 24)
    noise = RNG.normal(0, 3)
    value = base + diurnal + noise
    return round(float(np.clip(value, 10, 100)), 2)


def _rainfall(month: int) -> float:
    """Simulate hourly rainfall (mm) — mostly 0 outside monsoon."""
    monsoon_months = [6, 7, 8, 9]
    if month in monsoon_months:
        # Monsoon: occasional rainfall events
        rain_prob = 0.15
        if RNG.random() < rain_prob:
            return round(float(RNG.exponential(scale=3.0)), 2)
    elif month in [10, 11]:
        if RNG.random() < 0.04:
            return round(float(RNG.exponential(scale=1.0)), 2)
    return 0.0


def _solar_generation(hour: int, month: int, temperature: float,
                       humidity: float, rainfall: float) -> float:
    """
    Simulate solar generation (MW) for a distribution zone.
    Zero at night; bell curve during the day reduced by clouds/rain.
    """
    if hour < SOLAR_DAYTIME_START_HOUR or hour >= SOLAR_DAYTIME_END_HOUR:
        return 0.0

    # Peak solar generation varies by month
    month_peak = {
        1: 180, 2: 200, 3: 220, 4: 230, 5: 210,
        6: 160, 7: 140, 8: 145, 9: 160, 10: 190,
        11: 185, 12: 175
    }
    peak = month_peak[month]

    # Bell curve centred at noon
    hour_factor = np.exp(-0.5 * ((hour - 12) / 3.5) ** 2)

    # Cloud / rain reduction
    cloud_factor = 1.0 - 0.4 * (humidity / 100) - 0.1 * min(rainfall / 10, 1.0)
    cloud_factor = max(cloud_factor, 0.1)

    noise = RNG.normal(0, 5)
    value = peak * hour_factor * cloud_factor + noise
    return round(float(max(value, 0.0)), 2)


def _zone_demand(
    zone: str, hour: int, day_of_week: int, month: int,
    is_holiday: int, temperature: float, humidity: float
) -> float:
    """
    Simulate zone-level electricity demand (MW).

    Profiles:
    - Residential: evening peak (7–10 PM), lower daytime, seasonal summer boost
    - Commercial: business hours peak (9 AM–6 PM), flat evenings
    - Industrial: relatively flat with moderate morning ramp-up; reduced weekends
    """
    is_weekend = int(day_of_week >= 5)

    if zone == "Residential":
        # Base load varies by month (summer AC load)
        base = 180 + (temperature - 25) * 3.5
        # Evening peak pattern
        hour_factor = 0.6 + 0.4 * np.exp(-0.5 * ((hour - 20) / 2.5) ** 2)
        # Daytime slight trough
        if 10 <= hour <= 16:
            hour_factor *= 0.75
        holiday_factor = 1.05 if (is_weekend or is_holiday) else 1.0
        noise = RNG.normal(0, 8)
        demand = base * hour_factor * holiday_factor + noise

    elif zone == "Commercial":
        base = 150 + (temperature - 25) * 2.0
        # Business hours pattern
        if 9 <= hour <= 20:
            hour_factor = 0.8 + 0.2 * np.sin(np.pi * (hour - 9) / 11)
        elif hour < 7 or hour >= 22:
            hour_factor = 0.25
        else:
            hour_factor = 0.5
        holiday_factor = 0.4 if (is_weekend or is_holiday) else 1.0
        noise = RNG.normal(0, 6)
        demand = base * hour_factor * holiday_factor + noise

    else:  # Industrial
        base = 280 + (temperature - 25) * 1.5
        # Shift-based: two main shifts (morning + afternoon)
        if 6 <= hour <= 14:
            hour_factor = 0.90 + 0.1 * np.sin(np.pi * (hour - 6) / 8)
        elif 14 <= hour <= 22:
            hour_factor = 0.85
        else:
            hour_factor = 0.55   # Night shift lower
        # Industrial demand reduced on weekends/holidays
        holiday_factor = 0.55 if is_holiday else (0.65 if is_weekend else 1.0)
        noise = RNG.normal(0, 10)
        demand = base * hour_factor * holiday_factor + noise

    return round(float(max(demand, 5.0)), 2)


def generate_dataset(
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    freq: str = "H"
) -> pd.DataFrame:
    """
    Generate a multi-zone hourly electricity dataset for the given date range.

    Returns a DataFrame with columns:
        timestamp, zone, demand_mw, temperature_c, humidity_pct,
        rainfall_mm, solar_generation_mw, is_holiday, data_type
    """
    # Use lowercase 'h' for pandas >=2.2 compatibility (uppercase 'H' is deprecated)
    freq_normalized = freq.lower() if freq.upper() == "H" else freq
    timestamps = pd.date_range(start=start_date, end=end_date, freq=freq_normalized)
    logger.info(f"Generating synthetic data: {len(timestamps)} hours × {len(ZONES)} zones")

    records = []
    for ts in timestamps:
        hour = ts.hour
        dow = ts.dayofweek      # 0=Monday … 6=Sunday
        month = ts.month
        holiday = _is_holiday(ts)
        temp = _temperature(hour, month)
        humid = _humidity(hour, month, temp)
        rain = _rainfall(month)
        solar = _solar_generation(hour, month, temp, humid, rain)

        for zone in ZONES:
            demand = _zone_demand(zone, hour, dow, month, holiday, temp, humid)
            records.append({
                "timestamp": ts,
                "zone": zone,
                "demand_mw": demand,
                "temperature_c": temp,
                "humidity_pct": humid,
                "rainfall_mm": rain,
                "solar_generation_mw": solar,
                "is_holiday": holiday,
                "data_type": "synthetic",   # CLEARLY LABELLED as synthetic
            })

    df = pd.DataFrame(records)
    df.sort_values(["timestamp", "zone"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info(f"Dataset shape: {df.shape}")
    return df


def inject_anomalies(df: pd.DataFrame, anomaly_rate: float = 0.02) -> pd.DataFrame:
    """
    Inject realistic anomalies into the dataset for training Isolation Forest.
    Anomalies represent unusual consumption spikes or drops.
    These are labelled for evaluation purposes only — not used during IF training.
    """
    df = df.copy()
    df["injected_anomaly"] = 0

    n_anomalies = int(len(df) * anomaly_rate)
    anomaly_indices = RNG.choice(df.index, size=n_anomalies, replace=False)

    for idx in anomaly_indices:
        # Spike: 35–80% above normal
        if RNG.random() < 0.65:
            multiplier = RNG.uniform(1.35, 1.80)
        else:
            # Drop: 30–60% below normal
            multiplier = RNG.uniform(0.40, 0.70)
        df.at[idx, "demand_mw"] = round(df.at[idx, "demand_mw"] * multiplier, 2)
        df.at[idx, "injected_anomaly"] = 1

    logger.info(f"Injected {n_anomalies} anomalies ({anomaly_rate*100:.1f}% of rows)")
    return df


def save_dataset(df: pd.DataFrame, filename: str = "synthetic_demand_data.csv") -> Path:
    """Save the generated dataset to the raw data directory."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DATA_DIR / filename
    df.to_csv(output_path, index=False)
    logger.info(f"Dataset saved: {output_path}")
    return output_path


if __name__ == "__main__":
    logger.info("=== Synthetic Data Generation ===")
    logger.info("NOTE: This generates SYNTHETIC/DEMO data.")
    logger.info("It is NOT real Tata Power operational data.")

    df = generate_dataset(start_date="2023-01-01", end_date="2024-12-31")
    df = inject_anomalies(df, anomaly_rate=0.02)
    path = save_dataset(df)

    print(f"\nDataset summary:")
    print(df.describe())
    print(f"\nSaved to: {path}")
    print(f"Zones: {df['zone'].unique().tolist()}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Total records: {len(df):,}")
    print(f"Anomalies injected: {df['injected_anomaly'].sum():,}")
