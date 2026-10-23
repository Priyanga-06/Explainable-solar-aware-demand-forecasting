"""
Peak Severity Classification
==============================
Classifies net grid demand peaks into severity levels.

Severity levels and thresholds are loaded from config (not hard-coded).
This allows operators to adjust thresholds without changing code.

Levels:
    Normal   → 0–400 MW
    Medium   → 400–600 MW
    High     → 600–800 MW
    Critical → > 800 MW
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import SEVERITY_THRESHOLDS


def classify_severity(net_demand_mw: float) -> Dict:
    """
    Classify net grid demand into a severity level.

    Args:
        net_demand_mw: Net grid demand in MW (demand - solar)

    Returns:
        Dict with level, threshold_range, description, colour_hint
    """
    net_demand_mw = max(net_demand_mw, 0.0)

    for level, (low, high) in SEVERITY_THRESHOLDS.items():
        if low <= net_demand_mw < high:
            return _build_severity_result(level, net_demand_mw, low, high)

    # Fallback — should not occur with correct threshold config
    return _build_severity_result("Critical", net_demand_mw, 800, 99999)


def _build_severity_result(
    level: str, value: float, low: float, high: float
) -> Dict:
    descriptions = {
        "Normal":   "Grid demand is within normal operating range. "
                    "No immediate action required.",
        "Medium":   "Grid demand is moderately elevated. "
                    "Monitor closely and prepare contingency resources.",
        "High":     "Grid demand is high. "
                    "Consider activating demand-response measures and notifying operations team.",
        "Critical": "Grid demand is at critical levels. "
                    "Immediate action required — activate all available demand-response protocols.",
    }
    colour_hints = {
        "Normal": "#22c55e",    # Green
        "Medium": "#f59e0b",    # Amber
        "High":   "#f97316",    # Orange
        "Critical": "#ef4444",  # Red
    }
    high_display = f"{high} MW" if high < 99999 else "no upper limit"
    return {
        "level": level,
        "net_demand_mw": round(value, 2),
        "threshold_range": f"{low} - {high_display}",
        "description": descriptions[level],
        "colour_hint": colour_hints[level],
        "thresholds_source": "config/settings.py (SEVERITY_THRESHOLDS)",
    }


def classify_24h_severity(
    net_demand_24h: List[float],
    timestamps: List[str] = None,
) -> List[Dict]:
    """
    Classify severity for each hour of a 24-hour forecast.

    Returns list of dicts with hour_index, timestamp, severity details.
    """
    results = []
    for i, demand in enumerate(net_demand_24h):
        severity = classify_severity(demand)
        severity["hour_index"] = i
        severity["timestamp"] = timestamps[i] if timestamps and i < len(timestamps) else f"H+{i}"
        results.append(severity)
    return results


def get_peak_severity(
    predicted_demand_24h: List[float],
    predicted_solar_24h: List[float],
    timestamps: List[str] = None,
) -> Dict:
    """
    Calculate peak severity from 24-hour demand and solar forecasts.

    Returns:
        Dict with peak_net_demand_mw, peak_hour, peak_timestamp,
              severity, all_hours_severity
    """
    net_demands = [
        max(d - s, 0.0)
        for d, s in zip(predicted_demand_24h, predicted_solar_24h)
    ]

    peak_idx = int(max(range(len(net_demands)), key=lambda i: net_demands[i]))
    peak_net = net_demands[peak_idx]
    peak_ts = timestamps[peak_idx] if timestamps and peak_idx < len(timestamps) else f"H+{peak_idx}"

    severity = classify_severity(peak_net)

    # Hourly severity breakdown
    hourly = classify_24h_severity(net_demands, timestamps)

    return {
        "peak_net_demand_mw": round(peak_net, 2),
        "peak_hour_index": peak_idx,
        "peak_timestamp": peak_ts,
        "severity": severity,
        "net_demand_24h": [round(v, 2) for v in net_demands],
        "hourly_severity": hourly,
        "thresholds": SEVERITY_THRESHOLDS,
    }
