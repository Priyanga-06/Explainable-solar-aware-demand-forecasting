@echo off
echo ============================================================
echo  Solar-Aware Demand Forecasting — API Server (Port 8000)
echo ============================================================
cd /d "%~dp0"
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
pause
