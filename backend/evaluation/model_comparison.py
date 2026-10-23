"""
Model Comparison Module
========================
Compares GRU vs XGBoost baseline demand forecasting metrics.

All metrics come from actual model evaluation on the test set.
No manually entered values are used.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import METRICS_DIR, ZONES

logger = logging.getLogger(__name__)


def load_metrics(model_prefix: str, zone: str) -> Optional[Dict]:
    """Load saved evaluation metrics for a model-zone combination."""
    path = METRICS_DIR / f"{model_prefix}_{zone.lower()}_metrics.json"
    if not path.exists():
        logger.warning(f"Metrics not found: {path}")
        return None
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("metrics", data)


def build_comparison_table() -> List[Dict]:
    """
    Build a comparison table of GRU vs XGBoost baseline metrics per zone.

    Returns a list of rows:
        [{zone, model, MAE, RMSE, MAPE, R2}, ...]

    Only returns rows where metrics files exist (no fabricated values).
    """
    rows = []
    model_prefixes = {
        "GRU":              "gru",
        "XGBoost Baseline": "xgboost_baseline",
    }

    for zone in ZONES:
        for model_label, prefix in model_prefixes.items():
            metrics = load_metrics(prefix, zone)
            if metrics:
                rows.append({
                    "zone": zone,
                    "model": model_label,
                    "MAE":  metrics.get("MAE"),
                    "RMSE": metrics.get("RMSE"),
                    "MAPE": metrics.get("MAPE"),
                    "R2":   metrics.get("R2"),
                })
            else:
                rows.append({
                    "zone": zone,
                    "model": model_label,
                    "MAE":  None,
                    "RMSE": None,
                    "MAPE": None,
                    "R2":   None,
                    "note": "Model not yet trained — run training script first.",
                })

    return rows


def load_solar_metrics() -> Optional[Dict]:
    """Load XGBoost solar model metrics."""
    path = METRICS_DIR / "xgboost_solar_metrics.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("metrics", data)


def get_best_model_per_zone() -> Dict[str, str]:
    """
    Identify the best-performing model per zone based on MAE.
    Returns {zone: model_name}.
    """
    comparison = build_comparison_table()
    best = {}
    for zone in ZONES:
        zone_rows = [r for r in comparison if r["zone"] == zone and r.get("MAE") is not None]
        if zone_rows:
            zone_rows.sort(key=lambda x: x["MAE"])
            best[zone] = zone_rows[0]["model"]
        else:
            best[zone] = "Not available"
    return best


def get_full_comparison() -> Dict:
    """
    Return a complete model comparison report for the dashboard.
    """
    comparison_table = build_comparison_table()
    solar_metrics = load_solar_metrics()
    best_models = get_best_model_per_zone()

    return {
        "demand_model_comparison": comparison_table,
        "solar_model": {
            "name": "XGBoost Solar",
            "metrics": solar_metrics or {"note": "Solar model not yet trained."},
        },
        "best_model_per_zone": best_models,
        "metric_definitions": {
            "MAE":  "Mean Absolute Error (MW) — average magnitude of prediction error",
            "RMSE": "Root Mean Squared Error (MW) — penalises large errors more heavily",
            "MAPE": "Mean Absolute Percentage Error (%) — relative prediction accuracy",
            "R2":   "R² Coefficient of Determination — explained variance (1.0 = perfect)",
        },
        "data_source_note": (
            "All metrics are calculated from actual model evaluation on the held-out test set. "
            "No manually entered values are used."
        ),
    }
