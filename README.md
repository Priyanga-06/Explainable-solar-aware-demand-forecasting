# Explainable Solar-Aware Demand Forecasting
## An Explainable, Solar-Aware Decision Support System for Zone-Wise Demand Forecasting and Adaptive Anomaly Response in Indian Power Distribution Networks

> **Academic project — Phase 1 complete.**
> Uses synthetic/demo data representative of Indian power distribution contexts.
> This is NOT real Tata Power operational data.

---

## Phase 1 Status — ALL SUCCESS CRITERIA MET

| Check | Status |
|---|---|
| Data pipeline | PASS |
| Historical data loading | PASS |
| Preprocessing (no data leakage, chronological split) | PASS |
| Zone-wise GRU demand forecasting (XGBoost fallback) | PASS |
| 24-hour forecast | PASS |
| XGBoost solar forecasting | PASS |
| Net grid demand calculation | PASS |
| Isolation Forest anomaly detection | PASS |
| SHAP explainability | PASS |
| Peak severity (Normal/Medium/High/Critical) | PASS |
| Recommendation engine | PASS |
| Model comparison (real test-set metrics) | PASS |
| Forecast confidence/range | PASS |
| Peak cause explanation | PASS |
| Solar-load matching score | PASS |
| Officer dashboard (dark theme) | PASS |
| What-If Peak Mitigation Simulator | PASS |
| Simulated values separate from predictions | PASS |
| Report generation | PASS |

---

## Quick Start

### Step 1 — Install Python dependencies
```
pip install -r requirements.txt
```

### Step 2 — Train all models (run once)
```
python backend/training/train_all.py
```
> TensorFlow is optional. If not installed, XGBoost fallback is used automatically.

### Step 3 — Start the API backend
```
start_backend.bat
```
Or manually:
```
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 4 — Start the dashboard frontend
```
start_frontend.bat
```
Or manually:
```
cd frontend
npm install
npm run dev
```

### Step 5 — Open dashboard
```
http://localhost:3000
```

API docs available at: `http://localhost:8000/docs`

---

## Architecture

```
DATA -> FORECAST -> EXPLAIN -> DETECT -> RECOMMEND -> SIMULATE -> DECIDE
```

```
backend/
  data/
    preprocessing/   pipeline.py          Preprocessing, feature engineering
    database/        db_manager.py        SQLite storage
  models/
    gru/             model.py             GRU demand forecasting (TF)
                     fallback_model.py    XGBoost fallback (no TF needed)
    xgboost/         solar_model.py       Solar + demand baseline XGBoost
    isolation_forest/                     Anomaly detection
    saved/                               All trained models (.pkl / .keras)
  forecasting/       inference.py         Main inference engine
                     net_demand.py        Net grid demand calculation
  anomaly/           detector.py          Isolation Forest anomaly detection
  explainability/    shap_explainer.py    SHAP-based explanations
  severity/          peak_severity.py     Peak severity classification
  recommendations/   engine.py           Rule-based recommendation engine
  simulation/        what_if_simulator.py What-If Peak Mitigation Simulator
  evaluation/        model_comparison.py  Real test-set metrics
  training/          train_all.py         Full training pipeline
  api/               main.py             FastAPI REST endpoints

frontend/
  src/
    pages/           Dashboard, ZoneForecast, Solar, Anomaly,
                     Analytics, Recommendations, ModelComparison,
                     WhatIfSimulator, Reports, Settings
    components/      Layout, KPICard, SeverityBadge, LoadingState
    hooks/           useDashboard, useSimulation, useModelMetrics
    api/             client.js

config/
  settings.py        All thresholds, paths, constants (configurable)

data/
  raw/               synthetic_demand_data.csv
  processed/         train.csv, val.csv, test.csv, full_processed.csv
  database/          solar_forecast.db
```

---

## Models

| Model | Purpose | Metrics |
|---|---|---|
| XGBoost (GRU fallback) — Residential | 24h demand forecast | MAE=7.96, R2=0.89 |
| XGBoost (GRU fallback) — Commercial  | 24h demand forecast | MAE=5.93, R2=0.95 |
| XGBoost (GRU fallback) — Industrial  | 24h demand forecast | MAE=10.62, R2=0.88 |
| XGBoost Solar | Solar generation forecast | MAE=2.13, R2=0.99 |
| Isolation Forest (x3 zones) | Anomaly detection | — |
| XGBoost Baseline (x3 zones) | Model comparison | — |

> All metrics are from the held-out test set. No manually entered values.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Health check |
| GET | /api/v1/dashboard | Full dashboard payload |
| GET | /api/v1/forecast/demand | 24h zone-wise demand |
| GET | /api/v1/forecast/zones | Zone summary |
| GET | /api/v1/forecast/solar | Solar forecast |
| GET | /api/v1/net-demand | Net grid demand |
| GET | /api/v1/anomalies | Detected anomalies |
| POST | /api/v1/explain/shap | SHAP explanation |
| GET | /api/v1/severity | Peak severity |
| GET | /api/v1/recommendations | Recommendations |
| GET | /api/v1/models/metrics | Model comparison metrics |
| POST | /api/v1/simulation/what-if | What-If simulation |
| GET | /api/v1/simulation/history | Simulation audit trail |
| POST | /api/v1/reports/generate | Report generation |

---

## Dashboard Navigation

- **Dashboard** — KPI cards, 24h forecast, zone peaks, solar-load match
- **Zone Forecast** — Per-zone 24h demand with confidence band
- **Solar Integration** — Solar forecast, net demand, matching score
- **Anomaly Monitoring** — Isolation Forest results per zone
- **Forecast & Analytics** — SHAP feature importance, peak cause
- **Recommendations** — Rule-based recommendations with reasons
- **Model Comparison** — GRU vs XGBoost metrics from test set
- **What-If Simulator** — Peak mitigation scenario simulation
- **Reports** — Downloadable decision-support report
- **Settings** — Threshold and limit configuration

---

## Data

Synthetic data generated to be structurally representative of Indian power distribution:
- 2 years hourly data (2023–2024)
- 3 zones: Residential, Commercial, Industrial
- Features: temperature, humidity, rainfall, solar generation, Indian holidays
- 2% injected anomalies for Isolation Forest training
- Chronological train/val/test split (70/15/15) — no random splitting

---

## Academic Note

*"Designed for Indian power distribution contexts and demonstrated using available distribution-network-relevant data."*

This system does not claim real-time Tata Power operational deployment.
