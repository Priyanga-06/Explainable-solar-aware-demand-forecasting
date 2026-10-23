"""
Net Grid Demand Calculator
===========================
Calculates net grid demand as: Demand - Solar Generation.

Ensures physically valid results (net demand >= 0).
Also calculates the solar-load matching score.
"""

import numpy as np
from typing import List, Dict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    SOLAR_DAYTIME_START_HOUR,
    SOLAR_DAYTIME_END_HOUR,
)


def calculate_net_demand(
    demand_mw: List[float],
    solar_mw: List[float],
) -> List[float]:
    """
    Calculate hourly net grid demand.

    Net Grid Demand = max(Forecast Demand - Forecast Solar, 0)

    The max(., 0) ensures physical validity — net demand cannot be negative
    (excess solar would be curtailed or fed back to grid, but is not
    subtracted from another zone's demand in this model).
    """
    if len(demand_mw) != len(solar_mw):
        raise ValueError(
            f"demand_mw and solar_mw must have equal length. "
            f"Got {len(demand_mw)} vs {len(solar_mw)}"
        )
    net = [max(d - s, 0.0) for d, s in zip(demand_mw, solar_mw)]
    return [round(v, 2) for v in net]


def calculate_solar_load_match(
    demand_24h: List[float],
    solar_24h: List[float],
    hour_offsets: List[int] = None,
    current_hour: int = 0,
) -> Dict:
    """
    Calculate solar-load matching score for daytime hours.

    Formula:
        Solar-Load Match % = (Total Daytime Solar / Total Daytime Demand) × 100

    Where "daytime" is defined by SOLAR_DAYTIME_START_HOUR and
    SOLAR_DAYTIME_END_HOUR in config/settings.py.

    This represents the percentage of daytime demand that solar generation
    is capable of meeting.

    Args:
        demand_24h: 24-hour demand forecast (MW)
        solar_24h:  24-hour solar forecast (MW)
        hour_offsets: Optional list of actual hours for each forecast step
        current_hour: Starting hour of the forecast window

    Returns:
        Dict with match_pct, label, description, daytime_demand_mw,
              daytime_solar_mw
    """
    if not demand_24h or not solar_24h:
        return {"match_pct": 0.0, "label": "N/A", "description": "Insufficient data"}

    # Determine actual hour for each forecast step
    if hour_offsets:
        hours = hour_offsets
    else:
        hours = [(current_hour + i) % 24 for i in range(len(demand_24h))]

    # Filter to daytime hours
    daytime_indices = [
        i for i, h in enumerate(hours)
        if SOLAR_DAYTIME_START_HOUR <= h < SOLAR_DAYTIME_END_HOUR
        and i < len(demand_24h)
        and i < len(solar_24h)
    ]

    if not daytime_indices:
        return {
            "match_pct": 0.0,
            "label": "No Daytime Hours",
            "description": "No daytime hours in the forecast window.",
            "daytime_demand_mw": 0.0,
            "daytime_solar_mw": 0.0,
        }

    total_demand = sum(demand_24h[i] for i in daytime_indices)
    total_solar = sum(solar_24h[i] for i in daytime_indices)

    if total_demand <= 0:
        match_pct = 0.0
    else:
        match_pct = min((total_solar / total_demand) * 100, 100.0)

    match_pct = round(match_pct, 1)

    # Label
    if match_pct >= 70:
        label = "Excellent Match"
    elif match_pct >= 50:
        label = "Good Match"
    elif match_pct >= 30:
        label = "Moderate Match"
    elif match_pct > 0:
        label = "Low Match"
    else:
        label = "No Solar"

    description = (
        f"Solar generation is meeting approximately {match_pct:.1f}% "
        f"of relevant daytime demand ({SOLAR_DAYTIME_START_HOUR}:00–"
        f"{SOLAR_DAYTIME_END_HOUR}:00)."
    )

    return {
        "match_pct": match_pct,
        "label": label,
        "description": description,
        "daytime_demand_mw": round(total_demand, 2),
        "daytime_solar_mw": round(total_solar, 2),
        "formula": "Solar-Load Match % = (Total Daytime Solar / Total Daytime Demand) × 100",
    }
