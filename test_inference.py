"""Quick inference test script."""
import sys
sys.path.insert(0, '.')
import pandas as pd
from backend.forecasting.inference import run_full_inference

df = pd.read_csv('data/processed/full_processed.csv', parse_dates=['timestamp'])
df.sort_values(['zone', 'timestamp'], inplace=True)
recent = df.groupby('zone').tail(500).reset_index(drop=True)
print(f'Data rows: {len(recent)}')

payload = run_full_inference(recent)

print('\n=== KPI ===')
for k, v in payload['kpi'].items():
    print(f'  {k}: {v}')

print('\n=== Zone Forecasts ===')
for zone, fc in payload['zone_forecasts'].items():
    model = fc.get('model_used', 'GRU')
    print(f'  {zone}: peak={fc.get("peak_mw")} MW at H+{fc.get("peak_hour_offset")} [{model}]')

print('\n=== Solar ===')
print(f'  Peak solar: {payload["solar_forecast"].get("peak_solar_mw")} MW')
slm = payload['solar_load_match']
print(f'  Solar-load match: {slm.get("match_pct")}%  {slm.get("label")}')

print('\n=== Peak Severity ===')
sev = payload['peak_severity']
print(f'  Level: {sev["severity"]["level"]}')
print(f'  Peak net demand: {sev["peak_net_demand_mw"]} MW')
print(f'  Peak time: {sev["peak_timestamp"]}')

print('\n=== Anomalies ===')
print(f'  Count: {len(payload["anomalies"])}')
if payload['anomalies']:
    a = payload['anomalies'][0]
    print(f'  Sample: {a["zone"]} | {a["timestamp"]} | {a["deviation"]} | {a["status"]}')

print('\n=== Recommendations ===')
for r in payload['recommendations']:
    print(f'  P{r["priority"]} [{r["category"]}]: {r["action"][:70]}')

print('\n=== SHAP (Residential) ===')
shap = payload['shap_explanations'].get('Residential', {})
if 'top_factors' in shap:
    for f in shap['top_factors'][:3]:
        print(f'  {f["feature"]}: {f["shap_value"]:+.3f} ({f["impact"]})')
    print(f'  Peak cause: {shap.get("peak_cause_explanation", "")}')

print('\n=== Confidence ===')
c = payload['forecast_confidence']
print(f'  Expected peak: {c["expected_peak_mw"]} MW')
print(f'  Range: {c["estimated_lower_mw"]} - {c["estimated_upper_mw"]} MW')

print('\n=== Model Comparison (first 3 rows) ===')
mc = payload['model_comparison']['demand_model_comparison']
for row in mc[:3]:
    print(f'  {row["zone"]} | {row["model"]} | MAE={row["MAE"]} | R2={row["R2"]}')

print('\n=== Simulation test ===')
from backend.simulation.what_if_simulator import run_simulation
sim = run_simulation(
    baseline_demand_24h=payload['total_demand_24h'],
    baseline_solar_24h=payload['solar_24h'],
    industrial_shift_pct=10.0,
    commercial_shift_pct=5.0,
    residential_reduction_pct=3.0,
    solar_increase_pct=5.0,
    timestamps=payload['forecast_timestamps'],
)
print(f'  Baseline peak: {sim["baseline"]["peak_net_demand_mw"]} MW  ({sim["baseline"]["severity"]})')
print(f'  Simulated peak: {sim["simulated"]["peak_net_demand_mw"]} MW  ({sim["simulated"]["severity"]})')
print(f'  Reduction: {sim["impact"]["peak_reduction_mw"]} MW ({sim["impact"]["peak_reduction_pct"]}%)')
print(f'  Severity change: {sim["impact"]["severity_improvement"]}')

print('\n=== ALL TESTS PASSED ===')
