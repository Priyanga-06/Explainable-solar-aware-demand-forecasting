"""
Training Orchestrator
======================
Runs the complete training pipeline end-to-end.

Steps:
    1. Generate synthetic data (if real data not available)
    2. Run preprocessing pipeline
    3. Train GRU for all zones
    4. Train XGBoost solar model
    5. Train XGBoost demand baseline (for model comparison)
    6. Train Isolation Forest anomaly detectors
    7. Save all models, preprocessors, and evaluation metrics

Run once before starting the dashboard.
Do NOT retrain models on every dashboard load.

Usage:
    python backend/training/train_all.py
    python backend/training/train_all.py --use-existing-data
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, METRICS_DIR,
    ZONES, LOGS_DIR
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("train_all")


def ensure_log_file():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(LOGS_DIR / "training.log")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logging.getLogger().addHandler(fh)


def step1_generate_data(use_existing: bool) -> Path:
    """Generate synthetic data or use existing."""
    data_path = RAW_DATA_DIR / "synthetic_demand_data.csv"

    if use_existing and data_path.exists():
        logger.info(f"Using existing data: {data_path}")
        return data_path

    logger.info("=== STEP 1: Generating synthetic data ===")
    from data.generate_synthetic_data import generate_dataset, inject_anomalies, save_dataset
    df = generate_dataset(start_date="2023-01-01", end_date="2024-12-31")
    df = inject_anomalies(df, anomaly_rate=0.02)
    path = save_dataset(df)
    return path


def step2_preprocess(data_path: Path):
    """Run the full preprocessing pipeline."""
    logger.info("=== STEP 2: Preprocessing ===")
    from backend.data.preprocessing.pipeline import run_full_pipeline, load_preprocessor
    train_df, val_df, test_df, meta = run_full_pipeline(data_path)
    scalers = load_preprocessor("demand_scalers")
    logger.info(f"Preprocessing complete. Train: {meta['n_train']}, "
                f"Val: {meta['n_val']}, Test: {meta['n_test']}")
    return train_df, val_df, test_df, scalers, meta


def step3_train_gru(train_df, val_df, test_df, scalers):
    """Train GRU models for all zones."""
    logger.info("=== STEP 3: Training GRU demand models ===")
    from backend.models.gru.model import train_all_zones
    results = train_all_zones(train_df, val_df, test_df, scalers)
    for zone, result in results.items():
        if "error" not in result:
            logger.info(f"GRU {zone}: {result['metrics']}")
        else:
            logger.error(f"GRU {zone} FAILED: {result['error']}")
    return results


def step4_train_solar(train_df, val_df, test_df):
    """Train XGBoost solar model."""
    logger.info("=== STEP 4: Training XGBoost solar model ===")
    from backend.models.xgboost.solar_model import train_solar_model
    result = train_solar_model(train_df, val_df, test_df)
    logger.info(f"Solar model metrics: {result['metrics']}")
    return result


def step5_train_xgb_baseline(train_df, val_df, test_df):
    """Train XGBoost demand baseline for model comparison."""
    logger.info("=== STEP 5: Training XGBoost demand baselines ===")
    from backend.models.xgboost.solar_model import train_xgboost_baseline_for_demand
    results = {}
    for zone in ZONES:
        try:
            result = train_xgboost_baseline_for_demand(zone, train_df, val_df, test_df)
            results[zone] = result
            logger.info(f"XGBoost baseline {zone}: {result['metrics']}")
        except Exception as e:
            logger.error(f"XGBoost baseline {zone} FAILED: {e}")
    return results


def step6_train_anomaly(train_df):
    """Train Isolation Forest anomaly detectors."""
    logger.info("=== STEP 6: Training Isolation Forest anomaly detectors ===")
    from backend.anomaly.detector import train_isolation_forest
    models = train_isolation_forest(train_df)
    logger.info(f"Isolation Forest trained for {len(models)} zones")
    return models


def step7_init_database():
    """Initialise the SQLite database."""
    logger.info("=== STEP 7: Initialising database ===")
    from backend.data.database.db_manager import init_database
    # Ensure all required directories exist before any model/metric saves
    for d in [MODELS_DIR, METRICS_DIR, PROCESSED_DATA_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    init_database()


def main(use_existing_data: bool = False):
    ensure_log_file()
    logger.info("=" * 60)
    logger.info("Starting Phase 1 Training Pipeline")
    logger.info("=" * 60)
    logger.info("NOTE: Using SYNTHETIC/DEMO data — not real Tata Power data")

    try:
        data_path = step1_generate_data(use_existing_data)
        train_df, val_df, test_df, scalers, meta = step2_preprocess(data_path)
        step7_init_database()

        # GRU training (TensorFlow required; falls back to XGBoost if unavailable)
        gru_results = step3_train_gru(train_df, val_df, test_df, scalers)
        # Check if all zones failed (indicates TensorFlow is not installed)
        if all("error" in r for r in gru_results.values()):
            logger.warning("All GRU zones failed — TensorFlow not available. "
                           "Using XGBoost multi-step demand forecaster as fallback.")
            from backend.models.gru.fallback_model import train_all_zones_fallback
            gru_results = train_all_zones_fallback(train_df, val_df, test_df)
            for zone, result in gru_results.items():
                if "error" not in result:
                    logger.info(f"Fallback demand {zone}: {result['metrics']}")
                else:
                    logger.error(f"Fallback demand {zone} FAILED: {result['error']}")
        else:
            for zone, result in gru_results.items():
                if "error" not in result:
                    logger.info(f"GRU {zone}: {result['metrics']}")
                else:
                    logger.error(f"GRU {zone} FAILED: {result['error']}")

        # Solar model
        solar_result = step4_train_solar(train_df, val_df, test_df)

        # XGBoost demand baselines
        xgb_results = step5_train_xgb_baseline(train_df, val_df, test_df)

        # Anomaly detectors
        step6_train_anomaly(train_df)

        logger.info("=" * 60)
        logger.info("Training Pipeline COMPLETE")
        logger.info("=" * 60)
        logger.info("You can now start the API server and dashboard.")

    except Exception as e:
        logger.exception(f"Training pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all Phase 1 models")
    parser.add_argument(
        "--use-existing-data",
        action="store_true",
        help="Skip data generation and use existing synthetic data file",
    )
    args = parser.parse_args()
    main(use_existing_data=args.use_existing_data)
