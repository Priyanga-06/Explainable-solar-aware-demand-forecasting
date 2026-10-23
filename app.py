"""
Streamlit Dashboard — Solar-Aware Demand Forecasting
=====================================================
Single-file Streamlit app for Streamlit Cloud deployment.
Imports backend modules directly — no API server needed.

Run locally:
    streamlit run app.py
"""

import sys
import os
from pathlib import Path

# Ensure project root is on the path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── Dark theme config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Solar-Aware Demand Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .stApp { background-color: #0d1117; }
    .main .block-container { padding-top: 1rem; max-width: 1400px; }
    h1, h2, h3 { color: #e6edf3 !important; }
    .stMetric { background-color: #1c2433; border-radius: 8px; padding: 12px;
                border: 1px solid #30363d; border-left: 3px solid #58a6ff; }
    .stMetric label { color: #6e7681; font-size: 0.75rem; text-transform: uppercase;
                     letter-spacing: 0.05em; }
    .stMetric value { color: #e6edf3; font-weight: 700; }
    [data-testid="stSidebar"] { background-color: #161b22; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1c2433; border-radius: 6px 6px 0 0;
        padding: 8px 16px; color: #8b949e;
    }
    .stTabs [aria-selected="true"] {
        background-color: #58a6ff22 !important; color: #58a6ff !important;
    }
    .sim-warning {
        background: rgba(188,140,255,0.08); border: 1px solid rgba(188,140,255,0.3);
        border-radius: 8px; padding: 10px 14px; color: #bc8cff;
    }
    .data-note {
        background: rgba(88,166,255,0.06); border: 1px solid rgba(88,166,255,0.2);
        border-radius: 8px; padding: 8px 12px; font-size: 0.8rem; color: #8b949e;
    }
</style>
""", unsafe_allow_html=True)

# ─── Cached inference ──────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner="Running inference pipeline...")
def get_dashboard_data():
    """Load processed data and run full inference."""
    from config.settings import PROCESSED_DATA_DIR
    from backend.forecasting.inference import run_full_inference

    data_path = PROCESSED_DATA_DIR / "full_processed.csv"
    if not data_path.exists():
        st.error("Processed data not found. Training is needed first.")
        st.stop()

    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    df.sort_values(["zone", "timestamp"], inplace=True)
    recent = df.groupby("zone").tail(500).reset_index(drop=True)
    payload = run_full_inference(recent)
    return payload


@st.cache_data(ttl=600, show_spinner="Loading model metrics...")
def get_metrics():
    from backend.evaluation.model_comparison import get_full_comparison
    return get_full_comparison()


# ─── Color scheme ───────────────────────────────────────────────────────────
COLORS = {
    "demand": "#58a6ff",
    "solar":  "#d29922",
    "net":    "#db6d28",
    "bg":     "#0d1117",
    "card":   "#1c2433",
    "border": "#30363d",
    "text":   "#e6edf3",
    "muted":  "#6e7681",
    "Normal":   "#3fb950",
    "Medium":   "#d29922",
    "High":     "#db6d28",
    "Critical": "#f85149",
    "Residential": "#58a6ff",
    "Commercial":  "#39c5cf",
    "Industrial":  "#db6d28",
    "simulated": "#bc8cff",
}


# ─── Helper: metric card ────────────────────────────────────────────────────
def metric_card(title, value, unit="", color="#58a6ff", subtitle=""):
    """Render a metric card using st.metric."""
    if unit:
        val_str = f"{value} {unit}"
    else:
        val_str = str(value)
    col_color = color
    st.markdown(
        f"""<div style='background:{COLORS["card"]};border:1px solid {COLORS["border"]};
        border-left:3px solid {col_color};border-radius:8px;padding:12px;margin-bottom:8px'>
        <div style='font-size:0.7rem;color:{COLORS["muted"]};text-transform:uppercase;
        letter-spacing:0.05em'>{title}</div>
        <div style='font-size:1.5rem;font-weight:700;color:{col_color}'>{val_str}</div>
        <div style='font-size:0.7rem;color:{COLORS["muted"]};margin-top:2px'>{subtitle}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def chart_config():
    """Common Plotly layout config for dark theme."""
    return dict(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"], size=11),
        margin=dict(l=40, r=20, t=30, b=20),
        xaxis=dict(gridcolor=COLORS["border"], showgrid=True),
        yaxis=dict(gridcolor=COLORS["border"], showgrid=True),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ PowerAI Dashboard")
    st.caption("Solar-Aware Decision Support System")
    st.divider()

    # Data source notice
    st.markdown("**Data Source:**")
    st.markdown(
        "<span style='font-size:0.75rem;color:#8b949e'>Synthetic/Demo data — "
        "NOT real Tata Power data. Designed for Indian power distribution contexts.</span>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Severity thresholds
    st.markdown("**Severity Thresholds (config/settings.py):**")
    from config.settings import SEVERITY_THRESHOLDS
    for level, (low, high) in SEVERITY_THRESHOLDS.items():
        c = COLORS.get(level, "#8b949e")
        hi = f"{high} MW" if high < 99999 else "∞"
        st.markdown(
            f"<span style='color:{c};font-weight:600'>● {level}</span> "
            f"<span style='font-size:0.75rem;color:{COLORS["muted"]}'>{low}–{hi}</span>",
            unsafe_allow_html=True,
        )

    st.divider()

    # Refresh
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption("Auto-refresh: 10 min cache")


# ─── Load data ─────────────────────────────────────────────────────────────
data = get_dashboard_data()
metrics_data = get_metrics()

kpi = data["kpi"]
ts_list = data.get("forecast_timestamps", [])
total_demand = data.get("total_demand_24h", [])
solar_24h = data.get("solar_24h", [])
net_demand = data.get("net_demand_24h", [])
zone_fc = data.get("zone_forecasts", {})
peak_sev = data.get("peak_severity", {})
fc_conf = data.get("forecast_confidence", {})
solar_match = data.get("solar_load_match", {})
anomalies = data.get("anomalies", [])
recommendations = data.get("recommendations", [])
shap_exp = data.get("shap_explanations", {})
peak_cause = data.get("peak_cause_explanation", {})
severity_level = peak_sev.get("severity", {}).get("level", "Normal")
sev_color = COLORS.get(severity_level, "#8b949e")


# ─── Navigation tabs ────────────────────────────────────────────────────────
tab_names = [
    "📊 Dashboard", "🗺️ Zone Forecast", "☀️ Solar",
    "⚠️ Anomalies", "📈 Analytics", "💡 Recommendations",
    "⚖️ Model Comparison", "🔬 What-If Simulator", "📄 Reports",
]
tabs = st.tabs(tab_names)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown(f"## Officer Dashboard")
    st.markdown(
        f"<span style='font-size:0.8rem;color:{COLORS["muted"]}'>"
        f"Zone-wise demand forecasting & grid health monitoring — "
        f"Severity: </span>"
        f"<span style='color:{sev_color};font-weight:700'>{severity_level}</span>",
        unsafe_allow_html=True,
    )

    # KPI Row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        metric_card("Current Load", kpi.get("current_load_mw", 0), "MW", "#58a6ff", "Total across all zones")
    with k2:
        metric_card("Next-Hour Demand", kpi.get("next_hour_demand_mw", 0), "MW", "#39c5cf", "Forecast — next hour")
    with k3:
        metric_card("24-Hour Peak", kpi.get("peak_24h_mw", 0), "MW", sev_color,
                     f"Expected at {peak_sev.get('peak_timestamp', '—')}")
    with k4:
        metric_card("Net Grid Demand", kpi.get("net_grid_demand_mw", 0), "MW", "#db6d28", "After solar offset")

    k5, k6, k7, k8 = st.columns(4)
    with k5:
        metric_card("Solar Generation", kpi.get("solar_generation_mw", 0), "MW", "#d29922", "Current forecast")
    with k6:
        metric_card("Demand Status", severity_level, "", sev_color,
                     peak_sev.get("severity", {}).get("description", "")[:45])
    with k7:
        match_pct = solar_match.get("match_pct", 0)
        match_color = "#3fb950" if match_pct >= 50 else "#d29922"
        metric_card("Solar-Load Match", f"{match_pct}%", "", match_color, solar_match.get("label", ""))
    with k8:
        metric_card("Anomalies", len(anomalies), "", "#f85149" if anomalies else "#3fb950",
                     "Detected in recent window")

    st.divider()

    # 24-hour forecast chart
    st.markdown("### 24-Hour Demand & Solar Forecast")

    hours = [t.split(" ")[1] if " " in t else t for t in ts_list]
    chart_df = pd.DataFrame({
        "Hour": hours,
        "Total Demand (MW)": total_demand,
        "Solar Generation (MW)": solar_24h,
        "Net Grid Demand (MW)": net_demand,
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_df["Hour"], y=chart_df["Total Demand (MW)"],
                             fill="tozeroy", name="Total Demand",
                             line=dict(color=COLORS["demand"]), mode="lines"))
    fig.add_trace(go.Scatter(x=chart_df["Hour"], y=chart_df["Solar Generation (MW)"],
                             fill="tozeroy", name="Solar",
                             line=dict(color=COLORS["solar"]), mode="lines"))
    fig.add_trace(go.Scatter(x=chart_df["Hour"], y=chart_df["Net Grid Demand (MW)"],
                             fill="tozeroy", name="Net Grid Demand",
                             line=dict(color=COLORS["net"]), mode="lines"))

    # Peak marker
    peak_idx = peak_sev.get("peak_hour_index")
    if peak_idx is not None and peak_idx < len(hours):
        fig.add_vline(x=hours[peak_idx], line_dash="dash",
                       line_color=COLORS["Critical"],
                       annotation_text="PEAK", annotation_font_color=COLORS["Critical"])

    fig.update_layout(**chart_config(), height=300)
    st.plotly_chart(fig, use_container_width=True)

    # Zone peaks + forecast confidence side by side
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Zone-wise Peak Demand")
        zone_data = []
        for zone, fc in zone_fc.items():
            zone_data.append({"Zone": zone, "Peak (MW)": fc.get("peak_mw", 0),
                              "Peak Hour": f"H+{fc.get('peak_hour_offset', '—')}"})
        zone_df = pd.DataFrame(zone_data)
        fig2 = px.bar(zone_df, x="Zone", y="Peak (MW)",
                      color="Zone",
                      color_discrete_map=COLORS)
        fig2.update_layout(**chart_config(), height=250, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(zone_df, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("### Forecast Confidence")
        exp_peak = fc_conf.get("expected_peak_mw", 0)
        lo = fc_conf.get("estimated_lower_mw", 0)
        hi = fc_conf.get("estimated_upper_mw", 0)
        st.markdown(
            f"<div style='background:{COLORS["card"]};border:1px solid {COLORS["border"]};"
            f"border-radius:8px;padding:16px'>"
            f"<div style='font-size:0.7rem;color:{COLORS["muted"]}'>EXPECTED PEAK</div>"
            f"<div style='font-size:2rem;font-weight:700;color:{COLORS["demand"]}'>{exp_peak} MW</div>"
            f"<div style='font-size:0.7rem;color:{COLORS["muted"]};margin-top:8px'>ESTIMATED RANGE</div>"
            f"<div style='font-size:1.2rem;font-weight:600'>{lo} – {hi} MW</div>"
            f"<div style='font-size:0.7rem;color:{COLORS["muted"]};margin-top:6px'>"
            f"ⓘ {fc_conf.get('note', 'Estimated range ±7.5%')}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown(f"### Peak Severity")
        st.markdown(
            f"<div style='background:{COLORS["card"]};border:1px solid {COLORS["border"]};"
            f"border-left:3px solid {sev_color};border-radius:8px;padding:16px'>"
            f"<div style='font-size:2rem;font-weight:700;color:{sev_color}'>{severity_level}</div>"
            f"<div style='font-size:0.8rem;color:{COLORS["muted"]}'>"
            f"{peak_sev.get('peak_net_demand_mw', 0)} MW at {peak_sev.get('peak_timestamp', '—')}</div>"
            f"<div style='font-size:0.75rem;color:#8b949e;margin-top:6px'>"
            f"{peak_sev.get('severity', {}).get('description', '')}</div>"
            f"<div style='font-size:0.7rem;color:{COLORS["muted"]};margin-top:4px'>"
            f"Threshold: {peak_sev.get('severity', {}).get('threshold_range', '')}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: ZONE FORECAST
# ═══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("## Zone Forecast")
    st.caption("24-hour zone-wise electricity demand forecast")

    selected_zone = st.selectbox("Select Zone", ["Residential", "Commercial", "Industrial"])

    if selected_zone in zone_fc:
        fc = zone_fc[selected_zone]
        preds = fc.get("predicted_demand_mw", [])
        lower = fc.get("confidence_lower_mw", [])
        upper = fc.get("confidence_upper_mw", [])

        col1, col2, col3 = st.columns(3)
        with col1:
            metric_card("Peak Demand", fc.get("peak_mw", 0), "MW", COLORS[selected_zone], f"At H+{fc.get('peak_hour_offset', '—')}")
        with col2:
            metric_card("Model Used", fc.get("model_used", "XGBoost_Fallback"), "", "#8b949e", "Demand forecaster")
        with col3:
            metric_card("Forecast Points", len(preds), "hours", "#8b949e", "24-hour horizon")

        # Chart with confidence band
        zone_chart_df = pd.DataFrame({
            "Hour": hours[:len(preds)],
            "Demand (MW)": preds,
            "Lower Estimate": lower[:len(preds)] if lower else preds,
            "Upper Estimate": upper[:len(preds)] if upper else preds,
        })

        fig3 = go.Figure()
        zc = COLORS[selected_zone]
        fig3.add_trace(go.Scatter(
            x=zone_chart_df["Hour"], y=zone_chart_df["Upper Estimate"],
            fill=None, mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip"
        ))
        fig3.add_trace(go.Scatter(
            x=zone_chart_df["Hour"], y=zone_chart_df["Lower Estimate"],
            fill="tonexty", mode="lines", line=dict(width=0),
            fillcolor=f"rgba({88 if zc=='#58a6ff' else 57},{166 if zc=='#58a6ff' else 197},{255 if zc=='#58a6ff' else 207},0.1)",
            name="Estimated Range", hoverinfo="skip"
        ))
        fig3.add_trace(go.Scatter(
            x=zone_chart_df["Hour"], y=zone_chart_df["Demand (MW)"],
            mode="lines", name=f"{selected_zone} Demand",
            line=dict(color=zc, width=2)
        ))
        fig3.update_layout(**chart_config(), height=350)
        st.plotly_chart(fig3, use_container_width=True)

        st.info("ⓘ Shaded band is the Estimated Range (±7.5%). Not a statistically derived confidence interval.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: SOLAR INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("## Solar Integration")
    st.caption("XGBoost solar forecast, net grid demand, and solar-load matching")

    s1, s2, s3, s4 = st.columns(4)
    peak_solar = max(solar_24h) if solar_24h else 0
    with s1:
        metric_card("Peak Solar", f"{peak_solar:.0f}", "MW", "#d29922", "24h forecast peak")
    with s2:
        metric_card("Solar-Load Match", f"{solar_match.get('match_pct', 0)}%", "",
                     "#3fb950" if solar_match.get("match_pct", 0) >= 50 else "#d29922",
                     solar_match.get("label", ""))
    with s3:
        metric_card("Daytime Solar Total", f"{solar_match.get('daytime_solar_mw', 0):.0f}", "MW·h", "#39c5cf", "6AM–6PM")
    with s4:
        metric_card("Daytime Demand", f"{solar_match.get('daytime_demand_mw', 0):.0f}", "MW·h", "#58a6ff", "6AM–6PM")

    # Solar + demand + net chart
    solar_chart = pd.DataFrame({
        "Hour": hours,
        "Demand (MW)": total_demand,
        "Solar (MW)": solar_24h,
        "Net Demand (MW)": net_demand,
    })
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=solar_chart["Hour"], y=solar_chart["Demand (MW)"],
                              fill="tozeroy", name="Demand",
                              line=dict(color=COLORS["demand"])))
    fig4.add_trace(go.Scatter(x=solar_chart["Hour"], y=solar_chart["Solar (MW)"],
                              fill="tozeroy", name="Solar",
                              line=dict(color=COLORS["solar"])))
    fig4.add_trace(go.Scatter(x=solar_chart["Hour"], y=solar_chart["Net Demand (MW)"],
                              fill="tozeroy", name="Net Demand",
                              line=dict(color=COLORS["net"])))
    fig4.update_layout(**chart_config(), height=350)
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown(f"### Solar-Load Matching Formula")
    st.info(f"{solar_match.get('description', '')}\n\n"
            f"**Formula:** {solar_match.get('formula', 'Solar-Load Match % = (Total Daytime Solar / Total Daytime Demand) × 100')}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: ANOMALIES
# ═══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("## Anomaly Monitoring")
    st.caption("Unusual consumption patterns detected by Isolation Forest")

    zone_filter = st.selectbox("Filter by Zone", ["All", "Residential", "Commercial", "Industrial"],
                                key="anomaly_filter")
    filtered = anomalies if zone_filter == "All" else [a for a in anomalies if a.get("zone") == zone_filter]

    if filtered:
        anom_df = pd.DataFrame(filtered)
        display_cols = ["zone", "timestamp", "observed_load_mw", "expected_load_mw",
                        "deviation", "anomaly_score", "status", "status_note"]
        avail_cols = [c for c in display_cols if c in anom_df.columns]
        st.dataframe(anom_df[avail_cols], use_container_width=True, hide_index=True)
    else:
        st.success("No anomalies detected in the current window.")

    st.warning(
        "Anomalies are labelled as 'Possible abnormal consumption / possible non-technical loss'. "
        "They are NOT automatically classified as electricity theft. "
        "Ground-truth theft data would be required to make that determination."
    )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5: ANALYTICS (SHAP)
# ═══════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("## Forecast & Analytics")
    st.caption("SHAP explainability, peak cause analysis, feature importance")

    shap_zone = st.selectbox("Select Zone for SHAP", ["Residential", "Commercial", "Industrial"],
                              key="shap_zone")
    zone_exp = shap_exp.get(shap_zone, {})
    top_factors = zone_exp.get("top_factors", [])

    # Peak cause explanation
    if peak_cause.get("peak_cause_explanation"):
        st.markdown("### Peak Cause Explanation")
        st.info(peak_cause["peak_cause_explanation"])

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### Top Influencing Factors")
        if top_factors:
            for f in top_factors:
                direction = f.get("direction", "neutral")
                impact = f.get("impact", "Low")
                color = "#f85149" if direction == "positive" else "#3fb950"
                bar_pct = min(abs(f.get("shap_value", 0)) / 20 * 100, 100)
                st.markdown(
                    f"**{f['feature']}**  "
                    f"<span style='color:{color};font-size:0.75rem'>"
                    f"{'▲' if direction == 'positive' else '▼'} {impact} impact</span>",
                    unsafe_allow_html=True,
                )
                st.progress(int(bar_pct) / 100)
                st.caption(f.get("plain_explanation", ""))
        else:
            st.write("No SHAP data available.")

    with col_r:
        st.markdown("### Feature Importance (|SHAP|)")
        if top_factors:
            imp_df = pd.DataFrame([
                {"Feature": f["feature"], "|SHAP|": abs(f["shap_value"]),
                 "Direction": f["direction"]}
            for f in top_factors])
            fig5 = px.bar(imp_df, x="|SHAP|", y="Feature", orientation="h",
                          color="Direction",
                          color_discrete_map={"positive": "#f85149", "negative": "#3fb950"})
            fig5.update_layout(**chart_config(), height=300, showlegend=False)
            st.plotly_chart(fig5, use_container_width=True)

    if zone_exp.get("technical_note"):
        st.caption(f"ⓘ {zone_exp['technical_note']}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 6: RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("## Recommendations")
    st.caption(f"Rule-based adaptive recommendations — {len(recommendations)} active")

    for rec in recommendations:
        cat = rec.get("category", "General")
        color_map = {
            "Emergency": "#f85149", "Demand Response": "#db6d28",
            "Solar Optimisation": "#d29922", "Supply Preparation": "#58a6ff",
            "Monitoring": "#39c5cf", "Maintenance": "#3fb950",
            "Anomaly Response": "#f85149",
        }
        c = color_map.get(cat, "#58a6ff")
        st.markdown(
            f"<div style='background:{COLORS["card"]};border:1px solid {COLORS["border"]};"
            f"border-left:3px solid {c};border-radius:8px;padding:14px;margin-bottom:10px'>"
            f"<span style='font-size:0.65rem;color:{COLORS["muted"]};text-transform:uppercase;font-weight:700'>"
            f"Priority {rec.get('priority', '?')}</span> "
            f"<span style='background:{c}22;color:{c};padding:2px 8px;border-radius:10px;"
            f"font-size:0.7rem;font-weight:600'>{cat}</span><br>"
            f"<span style='font-size:0.95rem;font-weight:600;color:{COLORS["text"]}'>"
            f"{rec.get('action', '')}</span><br>"
            f"<span style='font-size:0.8rem;color:#8b949e'><strong>Why:</strong> {rec.get('reason', '')}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 7: MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("## Model Comparison")
    st.caption("GRU (XGBoost fallback) vs XGBoost Baseline — actual test-set metrics")

    comp_rows = metrics_data.get("demand_model_comparison", [])
    comp_df = pd.DataFrame(comp_rows)
    display_cols = [c for c in ["zone", "model", "MAE", "RMSE", "MAPE", "R2"] if c in comp_df.columns]
    comp_df = comp_df[display_cols].rename(columns={
        "zone": "Zone", "model": "Model", "MAE": "MAE (MW)",
        "RMSE": "RMSE (MW)", "MAPE": "MAPE (%)", "R2": "R²"
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.info(metrics_data.get("data_source_note", "All metrics from actual model evaluation on the test set."))

    # MAE chart
    if not comp_df.empty and "MAE (MW)" in comp_df.columns:
        mae_df = comp_df.dropna(subset=["MAE (MW)"])
        if not mae_df.empty:
            fig6 = px.bar(mae_df, x="Zone", y="MAE (MW)", color="Model",
                          barmode="group",
                          color_discrete_map={"GRU": "#58a6ff", "XGBoost Baseline": "#db6d28"})
            fig6.update_layout(**chart_config(), height=250)
            st.plotly_chart(fig6, use_container_width=True)

    # Solar model metrics
    solar_m = metrics_data.get("solar_model", {}).get("metrics", {})
    if solar_m and "MAE" in solar_m:
        st.markdown("### Solar Model — XGBoost")
        sm1, sm2, sm3, sm4 = st.columns(4)
        with sm1: metric_card("MAE", solar_m.get("MAE"), "MW", "#d29922")
        with sm2: metric_card("RMSE", solar_m.get("RMSE"), "MW", "#d29922")
        with sm3: metric_card("MAPE", solar_m.get("MAPE"), "%", "#d29922")
        with sm4: metric_card("R²", solar_m.get("R2"), "", "#d29922")

    # Metric definitions
    st.markdown("### Metric Definitions")
    for k, v in metrics_data.get("metric_definitions", {}).items():
        st.markdown(f"**{k}**: {v}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 8: WHAT-IF SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown("## 🔬 What-If Peak Mitigation Simulator")
    st.markdown(
        "<div class='sim-warning'>⚠ This is a scenario simulation, NOT a new ML prediction. "
        "It applies proportional adjustments to the Baseline Forecast. "
        "All results are labelled 'Simulated What-If Scenario'.</div>",
        unsafe_allow_html=True,
    )

    from config.settings import SIMULATOR_LIMITS

    col_ctrl, col_res = st.columns([1, 2])

    with col_ctrl:
        st.markdown("### Scenario Parameters")
        params = {}
        labels = {
            "industrial_shift_pct": "Industrial Load Shift",
            "commercial_shift_pct": "Commercial Load Shift",
            "residential_reduction_pct": "Residential Demand Reduction",
            "solar_increase_pct": "Solar Utilisation Increase",
        }
        for key, limits in SIMULATOR_LIMITS.items():
            params[key] = st.slider(
                labels.get(key, key),
                min_value=float(limits["min"]),
                max_value=float(limits["max"]),
                value=float(limits["default"]),
                step=1.0,
                key=f"slider_{key}",
            )

        run_btn = st.button("▶ RUN SIMULATION", type="primary", use_container_width=True)

    # Run simulation
    if run_btn or "sim_result" in st.session_state:
        if run_btn:
            from backend.simulation.what_if_simulator import run_simulation
            zone_demands = {
                z: zone_fc[z]["predicted_demand_mw"]
                for z in zone_fc if "predicted_demand_mw" in zone_fc[z]
            }
            sim_result = run_simulation(
                baseline_demand_24h=total_demand,
                baseline_solar_24h=solar_24h,
                industrial_shift_pct=params["industrial_shift_pct"],
                commercial_shift_pct=params["commercial_shift_pct"],
                residential_reduction_pct=params["residential_reduction_pct"],
                solar_increase_pct=params["solar_increase_pct"],
                zone_demands=zone_demands,
                timestamps=ts_list,
            )
            st.session_state["sim_result"] = sim_result
        else:
            sim_result = st.session_state["sim_result"]

        with col_res:
            st.markdown("### Simulation Results")
            st.markdown(
                f"<span style='background:#58a6ff22;color:#58a6ff;padding:2px 8px;"
                f"border-radius:10px;font-size:0.7rem;font-weight:600'>"
                f"BASELINE FORECAST</span> "
                f"<span style='background:#bc8cff22;color:#bc8cff;padding:2px 8px;"
                f"border-radius:10px;font-size:0.7rem;font-weight:600'>"
                f"SIMULATED WHAT-IF SCENARIO</span>",
                unsafe_allow_html=True,
            )

            # Comparison table
            b = sim_result["baseline"]
            s = sim_result["simulated"]
            imp = sim_result["impact"]

            comp_data = {
                "Metric": ["Peak Demand (MW)", "Solar (MW)", "Net Demand (MW)",
                          "Peak Time", "Severity"],
                "Baseline": [b["peak_demand_mw"], b["peak_solar_mw"],
                            b["peak_net_demand_mw"], b["peak_timestamp"], b["severity"]],
                "Simulated": [s["peak_demand_mw"], s["peak_solar_mw"],
                             s["peak_net_demand_mw"], s["peak_timestamp"], s["severity"]],
            }
            st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

            # Impact
            i1, i2, i3 = st.columns(3)
            with i1:
                metric_card("Peak Reduction", imp["peak_reduction_mw"], "MW", "#3fb950")
            with i2:
                metric_card("Peak Reduction %", imp["peak_reduction_pct"], "%", "#3fb950")
            with i3:
                metric_card("Severity Change", imp["severity_improvement"], "", "#bc8cff")

            st.info(f"**Recommended:** {imp['recommended_scenario']}")

            # Before/after chart
            sim_hours = [t.split(" ")[1] if " " in t else t for t in ts_list]
            sim_chart = pd.DataFrame({
                "Hour": sim_hours[:len(b["net_24h"])] if b.get("net_24h") else sim_hours,
                "Baseline Net": b.get("net_24h", []),
                "Simulated Net": s.get("net_24h", []),
            })
            fig7 = go.Figure()
            fig7.add_trace(go.Scatter(
                x=sim_chart["Hour"], y=sim_chart["Baseline Net"],
                name="Baseline Forecast", line=dict(color=COLORS["demand"], width=2)
            ))
            fig7.add_trace(go.Scatter(
                x=sim_chart["Hour"], y=sim_chart["Simulated Net"],
                name="Simulated Scenario", line=dict(color=COLORS["simulated"], width=2, dash="dash")
            ))
            fig7.update_layout(**chart_config(), height=300)
            st.plotly_chart(fig7, use_container_width=True)

            st.caption(f"⚠ {sim_result.get('disclaimer', '')}")
    else:
        with col_res:
            st.info("Adjust parameters and click **RUN SIMULATION** to see results.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 9: REPORTS
# ═══════════════════════════════════════════════════════════════════════════
with tabs[8]:
    st.markdown("## Decision Support Report")
    st.caption("Generate and download a report containing all dashboard sections")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        include_sim = st.checkbox("Include latest simulation results", value=True)
    with col_r2:
        if st.button("Generate Report", type="primary"):
            import json
            from datetime import datetime

            report = {
                "report_type": "Decision Support Report",
                "generated_at": datetime.utcnow().isoformat(),
                "data_note": (
                    "Synthetic/demo data. NOT real Tata Power data. "
                    "Designed for Indian power distribution contexts."
                ),
                "sections": {
                    "kpi_summary": kpi,
                    "forecast_24h": {
                        "total_demand": total_demand,
                        "solar": solar_24h,
                        "net_demand": net_demand,
                        "timestamps": ts_list,
                        "confidence": fc_conf,
                    },
                    "zone_forecasts": {
                        z: {"peak_mw": fc.get("peak_mw"), "peak_hour": fc.get("peak_hour_offset")}
                        for z, fc in zone_fc.items()
                    },
                    "solar_load_match": solar_match,
                    "peak_severity": {
                        "level": peak_sev.get("severity", {}).get("level"),
                        "peak_net_mw": peak_sev.get("peak_net_demand_mw"),
                        "peak_time": peak_sev.get("peak_timestamp"),
                    },
                    "anomalies": anomalies,
                    "shap_explanation": peak_cause,
                    "recommendations": recommendations,
                    "model_comparison": metrics_data.get("demand_model_comparison", []),
                },
            }

            if include_sim and "sim_result" in st.session_state:
                sim = st.session_state["sim_result"]
                report["sections"]["simulation"] = {
                    "label": "Scenario Simulation — What-If Run",
                    "note": "Simulated values are NOT actual or predicted values.",
                    "baseline_peak_mw": sim["baseline"]["peak_net_demand_mw"],
                    "simulated_peak_mw": sim["simulated"]["peak_net_demand_mw"],
                    "peak_reduction_mw": sim["impact"]["peak_reduction_mw"],
                    "peak_reduction_pct": sim["impact"]["peak_reduction_pct"],
                    "scenario_inputs": sim.get("scenario_inputs", {}),
                    "recommended_scenario": sim["impact"]["recommended_scenario"],
                }
            else:
                report["sections"]["simulation"] = {
                    "note": "No simulation run. Use the What-If Simulator first."
                }

            st.session_state["report"] = report
            st.success("Report generated!")

    if "report" in st.session_state:
        report = st.session_state["report"]
        st.json(report)

        # Download button
        report_json = json.dumps(report, indent=2, default=str)
        st.download_button(
            label="📄 Download Report (JSON)",
            data=report_json,
            file_name=f"power-grid-report-{datetime.now().strftime('%Y-%m-%d')}.json",
            mime="application/json",
        )
