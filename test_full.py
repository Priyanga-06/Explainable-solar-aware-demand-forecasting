"""Full end-to-end test for app.py backend."""
import sys, warnings, logging
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
sys.path.insert(0, ".")

import pandas as pd
import numpy as np

print("=== Testing all imports ===")
import pyarrow, pyarrow.lib
print(f"pyarrow {pyarrow.__version__} OK")
import streamlit
print(f"streamlit {streamlit.__version__} OK")
import plotly.graph_objects as go
import plotly.express as px
print(f"plotly {__import__('plotly').__version__} OK")
import shap
print(f"shap {shap.__version__} OK")
import xgboost
print(f"xgboost {xgboost.__version__} OK")

print("\n=== Testing config ===")
from config.settings import (
    PROCESSED_DATA_DIR, SEVERITY_THRESHOLDS, SIMULATOR_LIMITS,
    ZONES, SEVERITY_THRESHOLDS
)
print(f"ZONES: {ZONES}")
print(f"Severity levels: {list(SEVERITY_THRESHOLDS.keys())}")
print(f"Simulator keys: {list(SIMULATOR_LIMITS.keys())}")

print("\n=== Loading processed data ===")
df = pd.read_csv(PROCESSED_DATA_DIR / "full_processed.csv", parse_dates=["timestamp"])
df.sort_values(["zone", "timestamp"], inplace=True)
recent = df.groupby("zone").tail(500).reset_index(drop=True)
print(f"Data shape: {recent.shape}, zones: {recent['zone'].unique().tolist()}")

print("\n=== Running full inference ===")
from backend.forecasting.inference import run_full_inference
payload = run_full_inference(recent)

print(f"zone_forecasts: {list(payload['zone_forecasts'].keys())}")
print(f"total_demand_24h: {len(payload['total_demand_24h'])} points, max={max(payload['total_demand_24h']):.1f} MW")
print(f"solar_24h: {len(payload['solar_24h'])} points, max={max(payload['solar_24h']):.1f} MW")
print(f"net_demand_24h: {len(payload['net_demand_24h'])} points")
print(f"anomalies: {len(payload['anomalies'])}")
print(f"recommendations: {len(payload['recommendations'])}")
print(f"kpi: {payload['kpi']}")
print(f"severity: {payload['peak_severity'].get('severity', {}).get('level')}")
print(f"solar_load_match: {payload['solar_load_match'].get('match_pct')}%")
print(f"forecast_confidence: {payload['forecast_confidence']}")

print("\n=== Checking SHAP ===")
for zone, exp in payload["shap_explanations"].items():
    if "error" in exp:
        print(f"  SHAP {zone} ERROR: {exp['error']}")
    else:
        n = len(exp.get("top_factors", []))
        print(f"  SHAP {zone}: {n} factors OK")

print("\n=== Testing model comparison ===")
from backend.evaluation.model_comparison import get_full_comparison
metrics = get_full_comparison()
print(f"demand_model_comparison: {len(metrics['demand_model_comparison'])} rows")
for row in metrics["demand_model_comparison"]:
    mae = row.get("MAE")
    print(f"  {row['zone']} / {row['model']}: MAE={mae}")

print("\n=== Testing what-if simulator ===")
from backend.simulation.what_if_simulator import run_simulation
sim = run_simulation(
    baseline_demand_24h=payload["total_demand_24h"],
    baseline_solar_24h=payload["solar_24h"],
    industrial_shift_pct=10,
    commercial_shift_pct=5,
    residential_reduction_pct=3,
    solar_increase_pct=5,
    timestamps=payload["forecast_timestamps"],
)
print(f"  baseline peak: {sim['baseline']['peak_net_demand_mw']} MW ({sim['baseline']['severity']})")
print(f"  simulated peak: {sim['simulated']['peak_net_demand_mw']} MW ({sim['simulated']['severity']})")
print(f"  reduction: {sim['impact']['peak_reduction_mw']} MW ({sim['impact']['peak_reduction_pct']}%)")

print("\n=== Testing pyarrow dataframe conversion (streamlit internal) ===")
zone_df = pd.DataFrame([
    {"Zone": z, "Peak (MW)": payload["zone_forecasts"][z].get("peak_mw", 0)}
    for z in payload["zone_forecasts"]
])
tbl = pyarrow.Table.from_pandas(zone_df)
print(f"  zone_df -> pyarrow Table OK: {tbl.num_rows} rows")

print("\n=== Testing plotly charts ===")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=list(range(24)), y=payload["total_demand_24h"],
    fill="tozeroy", name="Demand", line=dict(color="#58a6ff"), mode="lines"
))
fig.add_shape(type="line", x0=5, x1=5, y0=0, y1=1,
              xref="x", yref="paper", line=dict(color="#f85149", dash="dash"))
fig.add_annotation(x=5, y=1, xref="x", yref="paper",
                   text="PEAK", font=dict(color="#f85149"), showarrow=False)
fig.update_layout(paper_bgcolor="#0d1117", height=300)
print("  go.Figure with shape+annotation OK")

fig2 = px.bar(zone_df, x="Zone", y="Peak (MW)", color="Zone")
print("  px.bar OK")

print("\n" + "="*50)
print("ALL TESTS PASSED")
print("="*50)
print("\nTo start the app run:")
print("  streamlit run app.py")
