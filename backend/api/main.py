"""
FastAPI Application — Main Entry Point
========================================
Provides all REST API endpoints for the officer dashboard.

API design:
    - All endpoints return structured JSON
    - Errors return {status, message, detail}
    - Simulated data is clearly separated from predicted data
    - No hard-coded secrets or API keys
    - Logging on all requests

Endpoints:
    GET  /health
    GET  /api/v1/forecast/demand
    GET  /api/v1/forecast/zones
    GET  /api/v1/forecast/solar
    GET  /api/v1/net-demand
    GET  /api/v1/anomalies
    POST /api/v1/explain/shap
    GET  /api/v1/severity
    GET  /api/v1/recommendations
    GET  /api/v1/models/metrics
    POST /api/v1/simulation/what-if
    GET  /api/v1/simulation/history
    POST /api/v1/reports/generate
    GET  /api/v1/dashboard
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd

from config.settings import API_HOST, API_PORT, API_DEBUG, LOG_LEVEL, LOG_FORMAT, SIMULATOR_LIMITS

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("api")

app = FastAPI(
    title="Solar-Aware Demand Forecasting API",
    description=(
        "Officer-facing decision support system for Indian power distribution networks. "
        "Designed for Indian power distribution contexts and demonstrated using "
        "available distribution-network-relevant data."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Simple in-memory cache ────────────────────────────────────────────────────
_dashboard_cache: Dict = {"payload": None, "timestamp": 0.0}
CACHE_TTL_SECONDS = 600  # 10 minutes


# ─── Pydantic Models ──────────────────────────────────────────────────────────

class WhatIfRequest(BaseModel):
    industrial_shift_pct: float = Field(
        default=10.0, ge=0, le=30,
        description="Percentage of industrial load to shift off-peak"
    )
    commercial_shift_pct: float = Field(
        default=5.0, ge=0, le=20,
        description="Percentage of commercial load to shift off-peak"
    )
    residential_reduction_pct: float = Field(
        default=3.0, ge=0, le=15,
        description="Percentage of residential demand reduction"
    )
    solar_increase_pct: float = Field(
        default=5.0, ge=0, le=20,
        description="Percentage increase in solar utilisation"
    )


class SHAPRequest(BaseModel):
    zone: str = Field(..., description="Zone: Residential / Commercial / Industrial")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_recent_data(n_rows_per_zone: int = 500) -> pd.DataFrame:
    """
    Load recent processed data for inference.
    Loads from full_processed.csv which already has all engineered features.
    Using 500 rows per zone ensures anomaly detection has enough context.
    """
    from config.settings import PROCESSED_DATA_DIR
    data_path = PROCESSED_DATA_DIR / "full_processed.csv"
    if not data_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Processed data not found. Run backend/training/train_all.py first."
        )
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    df.sort_values(["zone", "timestamp"], inplace=True)
    # Take last n_rows per zone — data already has lag/rolling features
    recent = df.groupby("zone").tail(n_rows_per_zone)
    return recent.reset_index(drop=True)


def _error_response(message: str, status_code: int = 500, detail: str = "") -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"status": "error", "message": message, "detail": detail}
    )


def _get_cached_payload(recent_df: pd.DataFrame) -> Dict:
    """Return cached dashboard payload or recompute if stale."""
    now = time.time()
    if (_dashboard_cache["payload"] is not None
            and (now - _dashboard_cache["timestamp"]) < CACHE_TTL_SECONDS):
        logger.debug("Returning cached dashboard payload")
        return _dashboard_cache["payload"]
    from backend.forecasting.inference import run_full_inference
    payload = run_full_inference(recent_df)
    _dashboard_cache["payload"] = payload
    _dashboard_cache["timestamp"] = now
    return payload


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    """Service health check."""
    return {"status": "ok", "service": "solar-aware-demand-forecasting", "version": "1.0.0"}


@app.get("/api/v1/dashboard", tags=["Dashboard"])
def get_dashboard():
    """
    Full dashboard payload — all KPIs, forecasts, anomalies,
    recommendations, and explanations in one call.
    """
    try:
        recent_df = _load_recent_data()
        payload = _get_cached_payload(recent_df)
        return {"status": "ok", "data": payload}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Dashboard inference failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/forecast/demand", tags=["Forecast"])
def get_demand_forecast(zone: Optional[str] = Query(None, description="Zone name (optional)")):
    """
    24-hour zone-wise demand forecast.
    Returns GRU predictions (or XGBoost fallback if GRU unavailable).
    """
    try:
        recent_df = _load_recent_data()
        payload = _get_cached_payload(recent_df)
        if zone:
            if zone not in payload["zone_forecasts"]:
                raise HTTPException(status_code=400, detail=f"Unknown zone: {zone}")
            return {"status": "ok", "zone": zone, "data": payload["zone_forecasts"][zone]}
        return {"status": "ok", "data": payload["zone_forecasts"]}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Demand forecast failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/forecast/zones", tags=["Forecast"])
def get_zone_summary():
    """Summary of zone-wise current demand and forecast peak."""
    try:
        recent_df = _load_recent_data()
        payload = _get_cached_payload(recent_df)
        summary = []
        for zone, forecast in payload["zone_forecasts"].items():
            summary.append({
                "zone": zone,
                "peak_mw": forecast.get("peak_mw"),
                "peak_hour_offset": forecast.get("peak_hour_offset"),
            })
        return {"status": "ok", "data": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/forecast/solar", tags=["Forecast"])
def get_solar_forecast():
    """24-hour solar generation forecast (XGBoost)."""
    try:
        recent_df = _load_recent_data()
        payload = _get_cached_payload(recent_df)
        return {
            "status": "ok",
            "data": {
                "solar_forecast": payload["solar_forecast"],
                "solar_24h": payload["solar_24h"],
                "timestamps": payload["forecast_timestamps"],
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/net-demand", tags=["Forecast"])
def get_net_demand():
    """Net grid demand = total demand forecast - solar forecast."""
    try:
        recent_df = _load_recent_data()
        payload = _get_cached_payload(recent_df)
        return {
            "status": "ok",
            "data": {
                "net_demand_24h": payload["net_demand_24h"],
                "total_demand_24h": payload["total_demand_24h"],
                "solar_24h": payload["solar_24h"],
                "timestamps": payload["forecast_timestamps"],
                "solar_load_match": payload["solar_load_match"],
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/anomalies", tags=["Anomaly"])
def get_anomalies(zone: Optional[str] = Query(None)):
    """Detected anomalies with zone, deviation, and status."""
    try:
        recent_df = _load_recent_data()
        payload = _get_cached_payload(recent_df)
        anomalies = payload["anomalies"]
        if zone:
            anomalies = [a for a in anomalies if a.get("zone") == zone]
        return {"status": "ok", "data": anomalies, "count": len(anomalies)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/explain/shap", tags=["Explainability"])
def get_shap_explanation(request: SHAPRequest):
    """SHAP-based explanation for demand forecast in a given zone."""
    try:
        from backend.explainability.shap_explainer import explain_xgboost_demand
        recent_df = _load_recent_data()

        # Data from _load_recent_data already has engineered features.
        # Apply scalers only (no re-engineering).
        from config.settings import MODELS_DIR
        import pickle
        with open(MODELS_DIR / "demand_scalers.pkl", "rb") as f:
            scalers = pickle.load(f)
        from backend.data.preprocessing.pipeline import apply_scalers
        processed = apply_scalers(recent_df, scalers)

        explanation = explain_xgboost_demand(request.zone, processed)
        return {"status": "ok", "zone": request.zone, "data": explanation}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("SHAP explanation failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/severity", tags=["Severity"])
def get_peak_severity():
    """Peak severity classification with threshold ranges."""
    try:
        recent_df = _load_recent_data()
        payload = _get_cached_payload(recent_df)
        return {"status": "ok", "data": payload["peak_severity"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/recommendations", tags=["Recommendations"])
def get_recommendations():
    """Adaptive rule-based recommendations with explanations."""
    try:
        recent_df = _load_recent_data()
        payload = _get_cached_payload(recent_df)
        return {
            "status": "ok",
            "data": payload["recommendations"],
            "count": len(payload["recommendations"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/models/metrics", tags=["Model Comparison"])
def get_model_metrics():
    """Actual evaluation metrics for GRU and XGBoost models."""
    try:
        from backend.evaluation.model_comparison import get_full_comparison
        comparison = get_full_comparison()
        return {"status": "ok", "data": comparison}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/simulation/what-if", tags=["What-If Simulator"])
def run_what_if_simulation(request: WhatIfRequest):
    """
    Run a what-if peak mitigation scenario simulation.

    IMPORTANT: This is a deterministic scenario calculation, NOT a new ML prediction.
    All outputs are clearly labelled 'Simulated What-If Scenario'.
    Simulated values are NOT mixed with actual or predicted values.
    """
    try:
        from backend.forecasting.inference import run_full_inference
        from backend.simulation.what_if_simulator import run_simulation

        recent_df = _load_recent_data()
        payload = run_full_inference(recent_df)

        # Extract per-zone 24h demand for weighted simulation
        zone_demands = {
            zone: payload["zone_forecasts"][zone]["predicted_demand_mw"]
            for zone in payload["zone_forecasts"]
            if "predicted_demand_mw" in payload["zone_forecasts"][zone]
        }

        simulation_result = run_simulation(
            baseline_demand_24h=payload["total_demand_24h"],
            baseline_solar_24h=payload["solar_24h"],
            industrial_shift_pct=request.industrial_shift_pct,
            commercial_shift_pct=request.commercial_shift_pct,
            residential_reduction_pct=request.residential_reduction_pct,
            solar_increase_pct=request.solar_increase_pct,
            zone_demands=zone_demands,
            timestamps=payload["forecast_timestamps"],
        )

        return {"status": "ok", "data": simulation_result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("What-if simulation failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/simulation/history", tags=["What-If Simulator"])
def get_simulation_history(limit: int = Query(20, ge=1, le=100)):
    """Retrieve past simulation scenarios (officer audit trail)."""
    try:
        from backend.data.database.db_manager import get_simulation_history
        history = get_simulation_history(limit=limit)
        return {"status": "ok", "data": history, "count": len(history)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/simulation/limits", tags=["What-If Simulator"])
def get_simulation_limits():
    """Return configurable limits for the what-if simulator inputs."""
    return {"status": "ok", "data": SIMULATOR_LIMITS}


@app.post("/api/v1/reports/generate", tags=["Reports"])
def generate_report(include_simulation: bool = Query(False)):
    """
    Generate a decision-support report.
    Returns a structured JSON report containing all dashboard sections.
    Frontend renders this as PDF/printable.

    Simulated values are clearly labelled 'Scenario Simulation'.
    """
    try:
        recent_df = _load_recent_data()
        payload = _get_cached_payload(recent_df)

        from datetime import datetime
        report = {
            "report_type": "Decision Support Report",
            "generated_at": datetime.utcnow().isoformat(),
            "data_note": (
                "This report is based on synthetic/demo data for demonstration purposes. "
                "It is NOT based on real Tata Power operational data. "
                "Designed for Indian power distribution contexts."
            ),
            "sections": {
                "kpi_summary": payload["kpi"],
                "forecast_24h": {
                    "total_demand": payload["total_demand_24h"],
                    "solar": payload["solar_24h"],
                    "net_demand": payload["net_demand_24h"],
                    "timestamps": payload["forecast_timestamps"],
                    "confidence": payload["forecast_confidence"],
                },
                "zone_forecasts": {
                    zone: {
                        "peak_mw": data.get("peak_mw"),
                        "peak_hour": data.get("peak_hour_offset"),
                    }
                    for zone, data in payload["zone_forecasts"].items()
                },
                "solar_load_match": payload["solar_load_match"],
                "peak_severity": {
                    "level": payload["peak_severity"]["severity"]["level"],
                    "peak_net_mw": payload["peak_severity"]["peak_net_demand_mw"],
                    "peak_time": payload["peak_severity"]["peak_timestamp"],
                    "description": payload["peak_severity"]["severity"]["description"],
                },
                "anomalies": payload["anomalies"],
                "shap_explanation": payload["peak_cause_explanation"],
                "recommendations": payload["recommendations"],
                "model_comparison": payload["model_comparison"]["demand_model_comparison"],
            },
        }

        if include_simulation:
            try:
                from backend.data.database.db_manager import get_simulation_history
                import json
                last_sim = get_simulation_history(limit=1)
                if last_sim:
                    sim_data = last_sim[0]
                    full_result = json.loads(sim_data.get("full_result", "{}"))
                    report["sections"]["simulation"] = {
                        "label": "Scenario Simulation — Most Recent What-If Run",
                        "note": "Simulated values are NOT actual or predicted values.",
                        "baseline_peak_mw": sim_data.get("baseline_peak_mw"),
                        "simulated_peak_mw": sim_data.get("simulated_peak_mw"),
                        "peak_reduction_mw": sim_data.get("peak_reduction_mw"),
                        "peak_reduction_pct": sim_data.get("peak_reduction_pct"),
                        "baseline_severity": sim_data.get("baseline_severity"),
                        "simulated_severity": sim_data.get("simulated_severity"),
                        "scenario_inputs": full_result.get("scenario_inputs", {}),
                        "recommended_scenario": full_result.get("impact", {}).get("recommended_scenario", ""),
                        "timestamp": sim_data.get("timestamp"),
                    }
                else:
                    report["sections"]["simulation"] = {
                        "note": (
                            "No simulation scenario run yet. "
                            "Use the What-If Simulator first."
                        )
                    }
            except Exception as sim_err:
                logger.warning(f"Could not load simulation for report: {sim_err}")
                report["sections"]["simulation"] = {"note": "Simulation data unavailable."}

        return {"status": "ok", "data": report}
    except Exception as e:
        logger.exception("Report generation failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/cache/invalidate", tags=["System"])
def invalidate_cache():
    """Force refresh of the dashboard cache on next request."""
    _dashboard_cache["payload"] = None
    _dashboard_cache["timestamp"] = 0.0
    return {"status": "ok", "message": "Cache invalidated"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_DEBUG,
        log_level=LOG_LEVEL.lower(),
    )
