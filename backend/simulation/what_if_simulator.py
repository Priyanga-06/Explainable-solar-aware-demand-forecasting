"""
What-If Peak Mitigation Simulator
====================================
Allows officers to test hypothetical peak-reduction scenarios
BEFORE taking operational action.

CRITICAL DESIGN RULES:
    1. This simulator does NOT modify or replace any ML prediction.
    2. It operates entirely on the BASELINE forecast values.
    3. All outputs are clearly labelled "Simulated What-If Scenario".
    4. Simulated values are NEVER mixed with actual or predicted values.
    5. The simulation is a deterministic calculation — not a new ML inference.

Simulation logic:
    Simulated Demand = Baseline Demand
                       × (1 - industrial_shift_pct/100)
                       × (1 - commercial_shift_pct/100)
                       × (1 - residential_reduction_pct/100)
    Simulated Solar  = Baseline Solar × (1 + solar_increase_pct/100)
    Simulated Net    = max(Simulated Demand - Simulated Solar, 0)
"""

import sys
import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import SIMULATOR_LIMITS, DB_PATH
from backend.severity.peak_severity import classify_severity

logger = logging.getLogger(__name__)


def validate_inputs(params: Dict) -> Dict:
    """
    Validate simulator input parameters against configured limits.
    Raises ValueError with a descriptive message for invalid inputs.
    """
    validated = {}
    for key, limits in SIMULATOR_LIMITS.items():
        value = params.get(key, limits["default"])
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Parameter '{key}' must be a number, got: {params.get(key)}")

        if not (limits["min"] <= value <= limits["max"]):
            raise ValueError(
                f"Parameter '{key}' = {value} is out of allowed range "
                f"[{limits['min']}, {limits['max']}]"
            )
        validated[key] = value
    return validated


def run_simulation(
    baseline_demand_24h: List[float],
    baseline_solar_24h: List[float],
    industrial_shift_pct: float = 0.0,
    commercial_shift_pct: float = 0.0,
    residential_reduction_pct: float = 0.0,
    solar_increase_pct: float = 0.0,
    zone_demands: Dict[str, List[float]] = None,
    timestamps: List[str] = None,
) -> Dict:
    """
    Calculate the simulated what-if scenario.

    This is a deterministic calculation on baseline forecast values.
    It does NOT re-run any ML model.

    The reductions are applied proportionally to each zone's contribution
    to total demand, based on the scenario inputs.

    Args:
        baseline_demand_24h: 24-hour total demand forecast (MW)
        baseline_solar_24h:  24-hour solar forecast (MW)
        industrial_shift_pct:    % of industrial load to shift off-peak
        commercial_shift_pct:    % of commercial load to shift off-peak
        residential_reduction_pct: % residential demand reduction
        solar_increase_pct:      % increase in solar utilisation
        zone_demands:            Optional per-zone 24h demand lists
        timestamps:              Optional list of timestamp strings

    Returns:
        Dict with full baseline and simulated scenario comparison
    """
    # Validate inputs
    params = {
        "industrial_shift_pct": industrial_shift_pct,
        "commercial_shift_pct": commercial_shift_pct,
        "residential_reduction_pct": residential_reduction_pct,
        "solar_increase_pct": solar_increase_pct,
    }
    validated = validate_inputs(params)

    ind_shift = validated["industrial_shift_pct"] / 100
    com_shift = validated["commercial_shift_pct"] / 100
    res_red = validated["residential_reduction_pct"] / 100
    sol_inc = validated["solar_increase_pct"] / 100

    # ── Zone-level weighting ────────────────────────────────────────────────
    # If per-zone demands are available, use them for weighted reduction.
    # Otherwise, apply a default zone split (representative of Indian distribution).
    if zone_demands:
        ind_avg = sum(zone_demands.get("Industrial", [0])) / max(len(zone_demands.get("Industrial", [1])), 1)
        com_avg = sum(zone_demands.get("Commercial", [0])) / max(len(zone_demands.get("Commercial", [1])), 1)
        res_avg = sum(zone_demands.get("Residential", [0])) / max(len(zone_demands.get("Residential", [1])), 1)
        total_zone = ind_avg + com_avg + res_avg or 1
        ind_weight = ind_avg / total_zone
        com_weight = com_avg / total_zone
        res_weight = res_avg / total_zone
    else:
        # Representative default split for Indian distribution
        ind_weight, com_weight, res_weight = 0.45, 0.30, 0.25

    # ── Simulate hourly values ──────────────────────────────────────────────
    simulated_demand = []
    simulated_solar = []
    simulated_net = []

    for i, (d, s) in enumerate(zip(baseline_demand_24h, baseline_solar_24h)):
        # Demand reduction = weighted sum of each zone's contribution × its shift
        demand_reduction_factor = (
            ind_weight * ind_shift
            + com_weight * com_shift
            + res_weight * res_red
        )
        sim_d = d * (1.0 - demand_reduction_factor)
        sim_s = s * (1.0 + sol_inc)
        sim_net = max(sim_d - sim_s, 0.0)

        simulated_demand.append(round(sim_d, 2))
        simulated_solar.append(round(sim_s, 2))
        simulated_net.append(round(sim_net, 2))

    # ── Baseline metrics ────────────────────────────────────────────────────
    baseline_net = [max(d - s, 0.0) for d, s in zip(baseline_demand_24h, baseline_solar_24h)]
    baseline_peak_net = max(baseline_net)
    baseline_peak_idx = baseline_net.index(baseline_peak_net)
    baseline_peak_demand = baseline_demand_24h[baseline_peak_idx]
    baseline_peak_solar = baseline_solar_24h[baseline_peak_idx]
    baseline_severity = classify_severity(baseline_peak_net)

    # ── Simulated metrics ───────────────────────────────────────────────────
    sim_peak_net = max(simulated_net)
    sim_peak_idx = simulated_net.index(sim_peak_net)
    sim_peak_demand = simulated_demand[sim_peak_idx]
    sim_peak_solar = simulated_solar[sim_peak_idx]
    sim_severity = classify_severity(sim_peak_net)

    # ── Reduction calculations ──────────────────────────────────────────────
    peak_reduction_mw = baseline_peak_net - sim_peak_net
    peak_reduction_pct = (peak_reduction_mw / baseline_peak_net * 100
                          if baseline_peak_net > 0 else 0.0)

    baseline_peak_ts = timestamps[baseline_peak_idx] if timestamps else f"H+{baseline_peak_idx}"
    sim_peak_ts = timestamps[sim_peak_idx] if timestamps else f"H+{sim_peak_idx}"

    result = {
        "data_type": "simulated",  # ALWAYS labelled — never mixed with real/predicted data
        "simulation_label": "Simulated What-If Scenario",
        "baseline_label": "Baseline Forecast",
        "scenario_inputs": {
            "industrial_shift_pct": validated["industrial_shift_pct"],
            "commercial_shift_pct": validated["commercial_shift_pct"],
            "residential_reduction_pct": validated["residential_reduction_pct"],
            "solar_increase_pct": validated["solar_increase_pct"],
        },
        "baseline": {
            "peak_demand_mw": round(baseline_peak_demand, 2),
            "peak_solar_mw": round(baseline_peak_solar, 2),
            "peak_net_demand_mw": round(baseline_peak_net, 2),
            "peak_timestamp": baseline_peak_ts,
            "severity": baseline_severity["level"],
            "demand_24h": [round(v, 2) for v in baseline_demand_24h],
            "solar_24h": [round(v, 2) for v in baseline_solar_24h],
            "net_24h": [round(v, 2) for v in baseline_net],
        },
        "simulated": {
            "peak_demand_mw": round(sim_peak_demand, 2),
            "peak_solar_mw": round(sim_peak_solar, 2),
            "peak_net_demand_mw": round(sim_peak_net, 2),
            "peak_timestamp": sim_peak_ts,
            "severity": sim_severity["level"],
            "demand_24h": simulated_demand,
            "solar_24h": simulated_solar,
            "net_24h": simulated_net,
        },
        "impact": {
            "peak_reduction_mw": round(peak_reduction_mw, 2),
            "peak_reduction_pct": round(peak_reduction_pct, 2),
            "severity_changed": baseline_severity["level"] != sim_severity["level"],
            "severity_improvement": (
                f"{baseline_severity['level']} -> {sim_severity['level']}"
                if baseline_severity["level"] != sim_severity["level"]
                else "No change in severity level"
            ),
            "recommended_scenario": _recommend_scenario(
                peak_reduction_pct, sim_severity["level"], validated
            ),
        },
        "disclaimer": (
            "This is a scenario simulation based on proportional adjustments to the baseline "
            "forecast. It is NOT a new ML model prediction. Simulated values should be used "
            "for decision support only and do not represent actual grid outcomes."
        ),
    }

    # Save to database for audit trail
    _save_simulation_to_db(result)

    return result


def _recommend_scenario(
    reduction_pct: float,
    sim_severity: str,
    inputs: Dict,
) -> str:
    """Generate a plain-language recommendation based on simulation outcomes."""
    if reduction_pct >= 10 and sim_severity in ("Normal", "Medium"):
        return (f"This scenario achieves a {reduction_pct:.1f}% peak reduction and brings "
                f"severity to '{sim_severity}'. Recommended for implementation.")
    elif reduction_pct >= 5:
        return (f"This scenario achieves a {reduction_pct:.1f}% peak reduction. "
                f"Consider implementing with targeted demand-response measures.")
    elif reduction_pct > 0:
        return (f"This scenario achieves a modest {reduction_pct:.1f}% reduction. "
                f"May need to increase load shift or demand reduction parameters.")
    else:
        return "No meaningful peak reduction achieved. Adjust parameters and re-run simulation."


def _save_simulation_to_db(result: Dict) -> None:
    """
    Persist simulation results to the database for audit and Phase 2 use.
    """
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS simulation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                baseline_peak_mw REAL,
                scenario_inputs TEXT,
                simulated_peak_mw REAL,
                peak_reduction_mw REAL,
                peak_reduction_pct REAL,
                baseline_severity TEXT,
                simulated_severity TEXT,
                full_result TEXT
            )
        """)

        cursor.execute("""
            INSERT INTO simulation_history (
                timestamp, baseline_peak_mw, scenario_inputs,
                simulated_peak_mw, peak_reduction_mw, peak_reduction_pct,
                baseline_severity, simulated_severity, full_result
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            result["baseline"]["peak_net_demand_mw"],
            json.dumps(result["scenario_inputs"]),
            result["simulated"]["peak_net_demand_mw"],
            result["impact"]["peak_reduction_mw"],
            result["impact"]["peak_reduction_pct"],
            result["baseline"]["severity"],
            result["simulated"]["severity"],
            json.dumps(result),
        ))

        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Could not save simulation to DB: {e}")
