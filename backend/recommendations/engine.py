"""
Adaptive Recommendation Engine
================================
Generates rule-based operational recommendations for power distribution officers.

Each recommendation:
    - Explains WHY it was generated (not just what to do)
    - References the triggering condition (severity, time, zone, solar, anomaly)
    - Is ranked by priority
    - Is NOT an automated action — it supports human decision-making
"""

import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import SOLAR_DAYTIME_START_HOUR, SOLAR_DAYTIME_END_HOUR


def generate_recommendations(
    severity_level: str,
    peak_hour_index: int,
    peak_net_demand_mw: float,
    current_hour: int,
    solar_generation_mw: float,
    zone_demands: Dict[str, float],
    anomalies: List[Dict] = None,
    solar_load_match_pct: float = None,
) -> List[Dict]:
    """
    Generate ranked operational recommendations.

    Args:
        severity_level: Normal / Medium / High / Critical
        peak_hour_index: Hour index (0–23) when peak is expected
        peak_net_demand_mw: Expected peak net demand in MW
        current_hour: Current hour of day
        solar_generation_mw: Current or forecast solar generation
        zone_demands: Dict of {zone: demand_mw}
        anomalies: List of anomaly dicts from anomaly detector
        solar_load_match_pct: Solar-load matching score (0–100)

    Returns:
        List of recommendation dicts, sorted by priority (1 = highest)
    """
    recommendations = []
    priority = 1

    # ── Critical severity ─────────────────────────────────────────────────────
    if severity_level == "Critical":
        recommendations.append({
            "priority": priority,
            "action": "Activate all demand-response protocols immediately.",
            "reason": (f"Net grid demand is forecast to reach {peak_net_demand_mw:.0f} MW, "
                       f"which is in the Critical range (>800 MW). "
                       f"Immediate demand-side management is required."),
            "category": "Emergency",
        })
        priority += 1

        recommendations.append({
            "priority": priority,
            "action": "Alert operations team and notify backup supply sources.",
            "reason": "Critical demand levels require coordination across all supply resources.",
            "category": "Emergency",
        })
        priority += 1

    # ── High severity ─────────────────────────────────────────────────────────
    if severity_level in ("High", "Critical"):
        industrial = zone_demands.get("Industrial", 0)
        if industrial > 200:
            recommendations.append({
                "priority": priority,
                "action": "Request shift of non-critical industrial load to off-peak hours.",
                "reason": (f"Industrial zone demand is {industrial:.0f} MW. "
                           f"Shifting flexible industrial processes can reduce peak net demand "
                           f"significantly."),
                "category": "Demand Response",
            })
            priority += 1

    # ── Evening peak preparation ──────────────────────────────────────────────
    hours_to_peak = peak_hour_index - current_hour
    if 17 <= peak_hour_index <= 22 and hours_to_peak > 0:
        recommendations.append({
            "priority": priority,
            "action": f"Prepare additional supply for evening peak at hour H+{peak_hour_index}.",
            "reason": (f"Peak demand is expected at H+{peak_hour_index} (evening). "
                       f"Evening demand increases as solar generation declines and "
                       f"residential consumption rises."),
            "category": "Supply Preparation",
        })
        priority += 1

    # ── Morning peak ──────────────────────────────────────────────────────────
    if 8 <= peak_hour_index <= 11:
        recommendations.append({
            "priority": priority,
            "action": "Monitor morning demand ramp-up closely.",
            "reason": (f"Peak is expected in morning hours (H+{peak_hour_index}). "
                       f"Commercial and industrial loads typically ramp up simultaneously, "
                       f"creating a sharp demand increase."),
            "category": "Monitoring",
        })
        priority += 1

    # ── Solar utilisation ─────────────────────────────────────────────────────
    if (SOLAR_DAYTIME_START_HOUR <= current_hour <= SOLAR_DAYTIME_END_HOUR
            and solar_generation_mw > 50):
        if solar_load_match_pct is not None and solar_load_match_pct < 60:
            recommendations.append({
                "priority": priority,
                "action": "Maximise solar utilisation — shift deferrable loads to daytime.",
                "reason": (f"Solar generation is available ({solar_generation_mw:.0f} MW) "
                           f"but solar-load matching is only {solar_load_match_pct:.1f}%. "
                           f"Scheduling flexible loads now will reduce evening grid demand."),
                "category": "Solar Optimisation",
            })
            priority += 1
        elif solar_load_match_pct is not None and solar_load_match_pct >= 60:
            recommendations.append({
                "priority": priority,
                "action": "Continue current daytime solar utilisation strategy.",
                "reason": (f"Solar-load matching is {solar_load_match_pct:.1f}% — "
                           f"solar generation is meeting a good portion of daytime demand."),
                "category": "Solar Optimisation",
            })
            priority += 1

    # ── Anomaly response ──────────────────────────────────────────────────────
    if anomalies:
        anomaly_zones = list({a["zone"] for a in anomalies})
        recommendations.append({
            "priority": priority,
            "action": (f"Investigate abnormal consumption in: "
                       f"{', '.join(anomaly_zones)}."),
            "reason": (f"{len(anomalies)} anomalous consumption pattern(s) detected. "
                       f"These may indicate possible non-technical losses, "
                       f"equipment faults, or data quality issues. "
                       f"Manual verification is recommended."),
            "category": "Anomaly Response",
        })
        priority += 1

    # ── Medium severity ───────────────────────────────────────────────────────
    if severity_level == "Medium":
        recommendations.append({
            "priority": priority,
            "action": "Monitor grid demand — prepare contingency resources.",
            "reason": (f"Net grid demand is in the Medium range ({peak_net_demand_mw:.0f} MW). "
                       f"No immediate action required, but resources should be on standby."),
            "category": "Monitoring",
        })
        priority += 1

    # ── Normal severity ───────────────────────────────────────────────────────
    if severity_level == "Normal":
        recommendations.append({
            "priority": priority,
            "action": "Grid demand is normal — consider scheduling maintenance during this window.",
            "reason": (f"Net grid demand is {peak_net_demand_mw:.0f} MW (Normal range). "
                       f"Low-demand periods are optimal for planned maintenance activities."),
            "category": "Maintenance",
        })
        priority += 1

    # ── Commercial zone ───────────────────────────────────────────────────────
    commercial = zone_demands.get("Commercial", 0)
    if commercial > 180 and severity_level in ("High", "Critical"):
        recommendations.append({
            "priority": priority,
            "action": "Issue advisory to large commercial consumers for voluntary load reduction.",
            "reason": (f"Commercial zone demand is {commercial:.0f} MW. "
                       f"Voluntary reduction by large commercial consumers can help "
                       f"during peak conditions."),
            "category": "Demand Response",
        })
        priority += 1

    # Sort by priority
    recommendations.sort(key=lambda x: x["priority"])
    return recommendations
