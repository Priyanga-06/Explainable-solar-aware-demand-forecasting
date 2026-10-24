"""
PowerAI — Solar-Aware Decision Support System
==============================================
Streamlit single-file app. Run: streamlit run app.py
"""
import sys, warnings, json
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PowerAI — Solar-Aware Decision Support",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ═══════════════════════════════════════════════════════════════════════════
BG        = "#020D1A"
SIDEBAR   = "#071D32"
CARD      = "#03172A"
BORDER    = "#17608A"
BORDER_DIM= "#0D3A54"
TEXT      = "#F5F7FA"
TEXT2     = "#C6D5E5"
MUTED     = "#6A8BA8"
BLUE      = "#168BFF"
SOLAR     = "#FFC107"
GREEN     = "#35D058"
ORANGE    = "#FF982B"
RED       = "#FF4545"
PURPLE    = "#BC8CFF"

SEV_COLORS = {"Normal": GREEN, "Medium": ORANGE, "High": ORANGE, "Critical": RED}
ZONE_COLORS = {"Residential": BLUE, "Commercial": "#39C5CF", "Industrial": ORANGE}

# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; }}
.stApp {{ background-color: {BG} !important; color: {TEXT}; }}
.main .block-container {{
    padding: 0 1.5rem 2rem 1.5rem !important;
    max-width: 1500px !important;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background-color: {SIDEBAR} !important;
    border-right: 1px solid {BORDER_DIM} !important;
}}
[data-testid="stSidebar"] * {{ color: {TEXT2} !important; }}
[data-testid="stSidebar"] strong,
[data-testid="stSidebar"] b {{ color: {TEXT} !important; }}
[data-testid="stSidebarContent"] {{ padding: 1rem 1.2rem !important; }}

/* ── TABS — full fix for invisible text ── */
.stTabs [data-baseweb="tab-list"] {{
    background: {SIDEBAR};
    border: 1px solid {BORDER_DIM};
    border-radius: 10px;
    padding: 5px 6px;
    gap: 3px;
    flex-wrap: wrap;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    border: none !important;
    border-radius: 7px !important;
    padding: 8px 13px !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    color: {TEXT2} !important;
    white-space: nowrap;
    transition: all 0.18s ease;
}}
.stTabs [data-baseweb="tab"]:hover {{
    background: {BORDER_DIM} !important;
    color: {TEXT} !important;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, #0A3060, #0D4080) !important;
    color: {BLUE} !important;
    border: 1px solid {BORDER} !important;
    box-shadow: 0 0 12px rgba(22,139,255,0.2) !important;
}}
/* force text inside tab to inherit */
.stTabs [data-baseweb="tab"] p,
.stTabs [data-baseweb="tab"] span,
.stTabs [data-baseweb="tab"] div,
.stTabs [data-baseweb="tab"] button {{
    color: inherit !important;
    font-size: inherit !important;
    font-weight: inherit !important;
}}
[data-baseweb="tab-highlight"] {{ background: {BLUE} !important; height: 2px; }}
[data-baseweb="tab-border"] {{ display: none !important; }}

/* ── Metrics ── */
[data-testid="stMetric"] {{
    background: {CARD};
    border: 1px solid {BORDER_DIM};
    border-radius: 10px;
    padding: 14px 16px !important;
}}
[data-testid="stMetricLabel"] > div {{
    color: {MUTED} !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}
[data-testid="stMetricValue"] > div {{
    color: {TEXT} !important;
    font-size: 1.9rem !important;
    font-weight: 800 !important;
}}

/* ── Headings ── */
h1 {{ font-size: 30px !important; font-weight: 800 !important; color: {TEXT} !important; }}
h2 {{ font-size: 22px !important; font-weight: 700 !important; color: {TEXT} !important; }}
h3 {{ font-size: 17px !important; font-weight: 700 !important; color: {TEXT} !important; }}
h4 {{ font-size: 14px !important; font-weight: 600 !important; color: {TEXT2} !important; }}
p, li {{ color: {TEXT2}; font-size: 14px; }}

/* ── Buttons ── */
.stButton > button {{
    background: {CARD}; color: {TEXT2};
    border: 1px solid {BORDER}; border-radius: 7px;
    font-weight: 700; font-size: 13px; padding: 8px 18px;
    transition: all 0.18s;
}}
.stButton > button:hover {{ background: {BORDER_DIM}; color: {TEXT}; }}
[data-testid="baseButton-primary"] {{
    background: linear-gradient(135deg, #0A4DA8, {BLUE}) !important;
    color: #fff !important; border: none !important;
    font-weight: 700 !important;
}}

/* ── Inputs ── */
.stSelectbox label, .stSlider label, .stCheckbox label {{
    color: {TEXT2} !important; font-weight: 600 !important; font-size: 13px !important;
}}
[data-testid="stSelectbox"] div[data-baseweb="select"] {{
    background: {CARD}; border: 1px solid {BORDER_DIM}; border-radius: 7px;
}}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {{ border: 1px solid {BORDER_DIM}; border-radius: 8px; }}
.stDataFrame th {{ background: {SIDEBAR} !important; color: {TEXT} !important;
                  font-weight: 700 !important; font-size: 12px !important; }}
.stDataFrame td {{ color: {TEXT2} !important; font-size: 13px !important; }}

/* ── Alerts ── */
[data-testid="stAlert"] {{ border-radius: 8px; font-size: 13px; }}

/* ── Divider ── */
hr {{ border-color: {BORDER_DIM} !important; margin: 0.8rem 0 !important; }}

/* ── Progress bars ── */
[data-testid="stProgressBar"] > div {{ background: {BORDER_DIM}; }}
[data-testid="stProgressBar"] > div > div {{ background: {BLUE}; }}

/* ── Caption ── */
.stCaption, [data-testid="stCaptionContainer"] {{
    color: {MUTED} !important; font-size: 12px !important;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER_DIM}; border-radius: 3px; }}

/* ── Sim warning ── */
.sim-warn {{
    background: rgba(188,140,255,0.07);
    border: 1px solid rgba(188,140,255,0.3);
    border-left: 4px solid {PURPLE};
    border-radius: 8px; padding: 12px 16px;
    color: {PURPLE}; font-weight: 600; font-size: 13px;
    margin-bottom: 12px;
}}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def _f(v, d=2):
    try:
        return round(float(v), d)
    except Exception:
        return v

def kpi_card(label, value, unit="", accent=BLUE, sub=""):
    val_str = f"{value} {unit}".strip()
    st.markdown(
        f"<div style='background:{CARD};border:1px solid {BORDER_DIM};"
        f"border-top:3px solid {accent};border-radius:10px;"
        f"padding:16px 18px;height:100%;'>"
        f"<div style='font-size:11px;font-weight:700;color:{MUTED};"
        f"text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px'>{label}</div>"
        f"<div style='font-size:2rem;font-weight:800;color:{accent};line-height:1.1'>{val_str}</div>"
        f"<div style='font-size:11px;color:{MUTED};margin-top:5px'>{sub}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

def section_header(title, subtitle=""):
    st.markdown(
        f"<div style='margin:18px 0 10px 0'>"
        f"<div style='font-size:17px;font-weight:700;color:{TEXT};'>{title}</div>"
        + (f"<div style='font-size:12px;color:{MUTED};margin-top:2px'>{subtitle}</div>" if subtitle else "")
        + "</div>",
        unsafe_allow_html=True,
    )

def chart_cfg(h=320):
    return dict(
        paper_bgcolor=BG, plot_bgcolor=CARD,
        font=dict(color=TEXT2, size=11, family="Inter"),
        margin=dict(l=50, r=20, t=35, b=40),
        height=h,
        xaxis=dict(gridcolor=BORDER_DIM, showgrid=True, color=TEXT2,
                   linecolor=BORDER_DIM, tickfont=dict(size=11)),
        yaxis=dict(gridcolor=BORDER_DIM, showgrid=True, color=TEXT2,
                   linecolor=BORDER_DIM, tickfont=dict(size=11)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=TEXT2),
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=600, show_spinner="⚡ Running inference pipeline...")
def get_dashboard_data():
    from config.settings import PROCESSED_DATA_DIR
    from backend.forecasting.inference import run_full_inference
    data_path = PROCESSED_DATA_DIR / "full_processed.csv"
    if not data_path.exists():
        st.error("Processed data not found. Run: python backend/training/train_all.py")
        st.stop()
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    df.sort_values(["zone", "timestamp"], inplace=True)
    recent = df.groupby("zone").tail(500).reset_index(drop=True)
    payload = run_full_inference(recent)
    kpi = payload.get("kpi", {})
    payload["kpi"] = {k: _f(v) if hasattr(v, "__float__") else v for k, v in kpi.items()}
    return payload

@st.cache_data(ttl=600, show_spinner="Loading model metrics...")
def get_metrics():
    from backend.evaluation.model_comparison import get_full_comparison
    return get_full_comparison()


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        f"<div style='padding:10px 0 14px 0;border-bottom:1px solid {BORDER_DIM};margin-bottom:14px'>"
        f"<div style='font-size:22px;font-weight:800;color:{TEXT};letter-spacing:-0.02em'>"
        f"⚡ PowerAI</div>"
        f"<div style='font-size:11px;color:{MUTED};font-weight:600;text-transform:uppercase;"
        f"letter-spacing:0.07em;margin-top:3px'>Solar-Aware Decision Support</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div style='font-size:11px;font-weight:700;color:{MUTED};text-transform:uppercase;"
        f"letter-spacing:0.07em;margin-bottom:6px'>Data Source</div>"
        f"<div style='font-size:12px;color:{TEXT2};line-height:1.5;padding:10px 12px;"
        f"background:{CARD};border:1px solid {BORDER_DIM};border-radius:8px;margin-bottom:14px'>"
        f"Synthetic / demo data<br>NOT real operational data<br>"
        f"Indian power distribution context</div>",
        unsafe_allow_html=True,
    )

    from config.settings import SEVERITY_THRESHOLDS
    st.markdown(
        f"<div style='font-size:11px;font-weight:700;color:{MUTED};text-transform:uppercase;"
        f"letter-spacing:0.07em;margin-bottom:8px'>Severity Thresholds</div>",
        unsafe_allow_html=True,
    )
    sev_clr = {"Normal": GREEN, "Medium": ORANGE, "High": ORANGE, "Critical": RED}
    for lvl, (lo, hi) in SEVERITY_THRESHOLDS.items():
        c = sev_clr.get(lvl, BLUE)
        hi_s = f"{hi} MW" if hi < 99999 else "∞"
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"padding:5px 10px;background:{CARD};border:1px solid {BORDER_DIM};"
            f"border-left:3px solid {c};border-radius:6px;margin-bottom:4px'>"
            f"<span style='font-size:12px;font-weight:700;color:{c}'>{lvl}</span>"
            f"<span style='font-size:11px;color:{MUTED}'>{lo}–{hi_s}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)
    if st.button("🔄  Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.markdown(
        f"<div style='font-size:11px;color:{MUTED};text-align:center;margin-top:8px'>"
        f"Auto-refresh · 10 min cache</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
data        = get_dashboard_data()
metrics_data= get_metrics()

kpi          = data["kpi"]
ts_list      = data.get("forecast_timestamps", [])
total_demand = data.get("total_demand_24h", [])
solar_24h    = data.get("solar_24h", [])
net_demand   = data.get("net_demand_24h", [])
zone_fc      = data.get("zone_forecasts", {})
peak_sev     = data.get("peak_severity", {})
fc_conf      = data.get("forecast_confidence", {})
solar_match  = data.get("solar_load_match", {})
anomalies    = data.get("anomalies", [])
recommendations = data.get("recommendations", [])
shap_exp     = data.get("shap_explanations", {})
peak_cause   = data.get("peak_cause_explanation", {})
sev_level    = peak_sev.get("severity", {}).get("level", "Normal")
sev_col      = SEV_COLORS.get(sev_level, BLUE)
hours        = [t.split(" ")[1] if " " in t else t for t in ts_list]
now_str      = datetime.now().strftime("%d %b %Y  %H:%M")


# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL HEADER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    f"<div style='background:linear-gradient(135deg,{SIDEBAR} 0%,{BG} 100%);"
    f"border:1px solid {BORDER_DIM};border-bottom:2px solid {BORDER};"
    f"border-radius:12px;padding:18px 28px 14px 28px;margin-bottom:16px;"
    f"display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;'>"

    f"<div>"
    f"<div style='font-size:10px;font-weight:700;color:{MUTED};text-transform:uppercase;"
    f"letter-spacing:0.12em;margin-bottom:4px'>Academic Project · Synthetic Data</div>"
    f"<div style='font-size:26px;font-weight:800;color:{TEXT};letter-spacing:-0.02em;"
    f"line-height:1.15'>⚡ POWERAI</div>"
    f"<div style='font-size:13px;font-weight:600;color:{BLUE};margin-top:2px'>"
    f"SOLAR-AWARE DECISION SUPPORT SYSTEM</div>"
    f"<div style='font-size:12px;color:{MUTED};margin-top:2px'>"
    f"Zone-Wise Demand Forecasting &amp; Adaptive Anomaly Response</div>"
    f"</div>"

    f"<div style='display:flex;gap:12px;align-items:center;flex-wrap:wrap;'>"
    f"<div style='background:{CARD};border:1px solid {BORDER_DIM};border-radius:8px;"
    f"padding:10px 16px;text-align:center;min-width:120px'>"
    f"<div style='font-size:10px;font-weight:700;color:{MUTED};text-transform:uppercase;"
    f"letter-spacing:0.08em'>Grid Status</div>"
    f"<div style='font-size:1.5rem;font-weight:800;color:{sev_col};line-height:1.2'>{sev_level}</div>"
    f"</div>"
    f"<div style='background:{CARD};border:1px solid {BORDER_DIM};border-radius:8px;"
    f"padding:10px 16px;text-align:right;'>"
    f"<div style='font-size:10px;font-weight:700;color:{MUTED};text-transform:uppercase;"
    f"letter-spacing:0.08em'>Last Updated</div>"
    f"<div style='font-size:13px;font-weight:600;color:{TEXT2}'>{now_str}</div>"
    f"</div>"
    f"</div>"
    f"</div>",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════
# NAVIGATION TABS
# ═══════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📊  Dashboard",
    "🗺️  Zone Forecast",
    "☀️  Solar",
    "⚠️  Anomalies",
    "📈  Analytics",
    "💡  Recommendations",
    "⚖️  Model Comparison",
    "🔬  What-If Simulator",
    "📄  Reports",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    # KPI row 1
    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi_card("Current Load", kpi.get("current_load_mw",0), "MW", BLUE, "Total across all zones")
    with c2: kpi_card("Next-Hour Demand", kpi.get("next_hour_demand_mw",0), "MW", "#39C5CF", "Forecast — next hour")
    with c3: kpi_card("24-Hour Peak", kpi.get("peak_24h_mw",0), "MW", sev_col,
                      "At " + str(peak_sev.get("peak_timestamp","—")))
    with c4: kpi_card("Net Grid Demand", kpi.get("net_grid_demand_mw",0), "MW", ORANGE, "After solar offset")

    st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)

    c5,c6,c7,c8 = st.columns(4)
    with c5: kpi_card("Solar Generation", kpi.get("solar_generation_mw",0), "MW", SOLAR, "Current forecast")
    with c6: kpi_card("Demand Status", sev_level, "", sev_col,
                      peak_sev.get("severity",{}).get("description","")[:50])
    with c7:
        mp = solar_match.get("match_pct",0)
        kpi_card("Solar-Load Match", str(mp)+"%", "", GREEN if mp>=50 else ORANGE, solar_match.get("label",""))
    with c8: kpi_card("Anomalies Detected", len(anomalies), "", RED if anomalies else GREEN,
                      "In recent window")

    st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)
    st.divider()

    # Main forecast chart
    section_header("24-Hour Demand & Solar Forecast", "Zone-aggregated prediction vs solar generation")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours, y=total_demand, name="Total Demand",
        fill="tozeroy", mode="lines",
        line=dict(color=BLUE, width=2),
        fillcolor="rgba(22,139,255,0.08)"))
    fig.add_trace(go.Scatter(x=hours, y=solar_24h, name="Solar Generation",
        fill="tozeroy", mode="lines",
        line=dict(color=SOLAR, width=2),
        fillcolor="rgba(255,193,7,0.08)"))
    fig.add_trace(go.Scatter(x=hours, y=net_demand, name="Net Grid Demand",
        fill="tozeroy", mode="lines",
        line=dict(color=ORANGE, width=2),
        fillcolor="rgba(255,152,43,0.08)"))

    pk = peak_sev.get("peak_hour_index")
    if pk is not None and pk < len(hours):
        fig.add_shape(type="line", x0=hours[pk], x1=hours[pk], y0=0, y1=1,
                      xref="x", yref="paper",
                      line=dict(color=RED, dash="dash", width=1.5))
        fig.add_annotation(x=hours[pk], y=0.97, xref="x", yref="paper",
                           text="⚠ PEAK", font=dict(color=RED, size=11, family="Inter"),
                           showarrow=False, bgcolor=CARD,
                           bordercolor=RED, borderwidth=1, borderpad=4)

    fig.update_layout(**chart_cfg(340))
    st.plotly_chart(fig, use_container_width=True)

    # Zone peaks + Confidence
    col_a, col_b = st.columns(2)

    with col_a:
        section_header("Zone-wise Peak Demand")
        zdata = [{"Zone": z, "Peak (MW)": _f(fc.get("peak_mw",0)),
                  "Peak Hour": "H+"+str(fc.get("peak_hour_offset","—"))}
                 for z, fc in zone_fc.items()]
        zdf = pd.DataFrame(zdata)
        fig2 = px.bar(zdf, x="Zone", y="Peak (MW)", color="Zone",
                      color_discrete_map=ZONE_COLORS, text="Peak (MW)")
        fig2.update_traces(texttemplate="%{text:.0f}", textposition="outside",
                           textfont=dict(color=TEXT, size=12))
        fig2.update_layout(**chart_cfg(240), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(zdf, use_container_width=True, hide_index=True)

    with col_b:
        section_header("Forecast Confidence & Peak Severity")
        ep  = fc_conf.get("expected_peak_mw",0)
        lo  = fc_conf.get("estimated_lower_mw",0)
        hi2 = fc_conf.get("estimated_upper_mw",0)
        st.markdown(
            f"<div style='background:{CARD};border:1px solid {BORDER_DIM};"
            f"border-radius:10px;padding:18px;margin-bottom:12px'>"
            f"<div style='font-size:11px;font-weight:700;color:{MUTED};"
            f"text-transform:uppercase;letter-spacing:0.08em'>Expected Peak</div>"
            f"<div style='font-size:2.4rem;font-weight:800;color:{BLUE};margin:4px 0'>{ep} MW</div>"
            f"<div style='font-size:11px;color:{MUTED};text-transform:uppercase;"
            f"letter-spacing:0.06em;margin-top:8px'>Estimated Range</div>"
            f"<div style='font-size:1.2rem;font-weight:700;color:{TEXT2}'>{lo} – {hi2} MW</div>"
            f"<div style='font-size:11px;color:{MUTED};margin-top:6px'>"
            f"ⓘ {fc_conf.get('note','Estimated range ±7.5%')}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        pn  = peak_sev.get("peak_net_demand_mw",0)
        pts = peak_sev.get("peak_timestamp","—")
        sd  = peak_sev.get("severity",{}).get("description","")
        sr  = peak_sev.get("severity",{}).get("threshold_range","")
        st.markdown(
            f"<div style='background:{CARD};border:1px solid {BORDER_DIM};"
            f"border-left:4px solid {sev_col};border-radius:10px;padding:18px'>"
            f"<div style='font-size:11px;font-weight:700;color:{MUTED};"
            f"text-transform:uppercase;letter-spacing:0.08em'>Peak Severity</div>"
            f"<div style='font-size:2rem;font-weight:800;color:{sev_col};margin:4px 0'>{sev_level}</div>"
            f"<div style='font-size:13px;color:{TEXT2}'>{pn} MW · {pts}</div>"
            f"<div style='font-size:12px;color:{MUTED};margin-top:6px'>{sd}</div>"
            f"<div style='font-size:11px;color:{MUTED};margin-top:4px'>Threshold: {sr}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — ZONE FORECAST
# ═══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    section_header("Zone Forecast", "24-hour zone-wise electricity demand forecast")
    sel_z = st.selectbox("Select Zone", ["Residential", "Commercial", "Industrial"])

    if sel_z in zone_fc:
        fc    = zone_fc[sel_z]
        preds = fc.get("predicted_demand_mw",[])
        lower = fc.get("confidence_lower_mw",[])
        upper = fc.get("confidence_upper_mw",[])
        zc    = ZONE_COLORS.get(sel_z, BLUE)

        c1,c2,c3 = st.columns(3)
        with c1: kpi_card("Peak Demand", _f(fc.get("peak_mw",0)), "MW", zc,
                          "At H+"+str(fc.get("peak_hour_offset","—")))
        with c2: kpi_card("Model", fc.get("model_used","XGBoost_Fallback"), "", MUTED, "Demand forecaster")
        with c3: kpi_card("Horizon", len(preds), "hours", MUTED, "24-hour forecast")

        fig3 = go.Figure()
        hr_z = hours[:len(preds)]
        up   = upper[:len(preds)] if upper else preds
        dn   = lower[:len(preds)] if lower else preds
        fig3.add_trace(go.Scatter(x=hr_z, y=up, fill=None, mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig3.add_trace(go.Scatter(x=hr_z, y=dn, fill="tonexty", mode="lines",
            line=dict(width=0), fillcolor="rgba(22,139,255,0.10)",
            name="Estimated Range ±7.5%", hoverinfo="skip"))
        fig3.add_trace(go.Scatter(x=hr_z, y=preds, mode="lines+markers",
            name=sel_z+" Demand", line=dict(color=zc, width=2.5),
            marker=dict(size=3, color=zc)))
        fig3.update_layout(**chart_cfg(370))
        st.plotly_chart(fig3, use_container_width=True)
        st.info("ⓘ Shaded band = Estimated Range (±7.5%) — not a statistically derived confidence interval.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — SOLAR
# ═══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    section_header("Solar Integration", "XGBoost solar forecast, net grid demand & solar-load matching")

    sm_pct = solar_match.get("match_pct",0)
    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi_card("Peak Solar", f"{max(solar_24h):.0f}" if solar_24h else "0", "MW", SOLAR, "24h peak")
    with c2: kpi_card("Solar-Load Match", str(sm_pct)+"%", "",
                      GREEN if sm_pct>=50 else ORANGE, solar_match.get("label",""))
    with c3: kpi_card("Daytime Solar", f"{solar_match.get('daytime_solar_mw',0):.0f}", "MW·h", "#39C5CF", "6AM–6PM")
    with c4: kpi_card("Daytime Demand", f"{solar_match.get('daytime_demand_mw',0):.0f}", "MW·h", BLUE, "6AM–6PM")

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=hours, y=total_demand, name="Total Demand",
        fill="tozeroy", mode="lines", line=dict(color=BLUE, width=2),
        fillcolor="rgba(22,139,255,0.07)"))
    fig4.add_trace(go.Scatter(x=hours, y=solar_24h, name="Solar Generation",
        fill="tozeroy", mode="lines", line=dict(color=SOLAR, width=2),
        fillcolor="rgba(255,193,7,0.10)"))
    fig4.add_trace(go.Scatter(x=hours, y=net_demand, name="Net Grid Demand",
        fill="tozeroy", mode="lines", line=dict(color=ORANGE, width=2),
        fillcolor="rgba(255,152,43,0.07)"))
    fig4.update_layout(**chart_cfg(370))
    st.plotly_chart(fig4, use_container_width=True)

    f_txt = solar_match.get("formula","Solar-Load Match % = (Daytime Solar / Daytime Demand) × 100")
    st.info("**Formula:** " + f_txt + "\n\n" + solar_match.get("description",""))


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — ANOMALIES
# ═══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    section_header("Anomaly Monitoring", "Unusual consumption patterns — Isolation Forest detection")

    zf = st.selectbox("Filter by Zone", ["All","Residential","Commercial","Industrial"],
                      key="anom_filter")
    fil = anomalies if zf=="All" else [a for a in anomalies if a.get("zone")==zf]

    # summary badges
    n_anom = len(fil)
    badge_col = RED if n_anom>0 else GREEN
    st.markdown(
        f"<div style='display:inline-block;background:{badge_col}22;"
        f"border:1px solid {badge_col};border-radius:20px;padding:4px 14px;"
        f"font-size:13px;font-weight:700;color:{badge_col};margin-bottom:12px'>"
        f"{'⚠ '+str(n_anom)+' Anomalies Detected' if n_anom else '✓ No Anomalies Detected'}</div>",
        unsafe_allow_html=True,
    )

    if fil:
        adf = pd.DataFrame(fil)
        dc  = ["zone","timestamp","observed_load_mw","expected_load_mw",
               "deviation","anomaly_score","status","status_note"]
        ac  = [c for c in dc if c in adf.columns]
        st.dataframe(adf[ac], use_container_width=True, hide_index=True)
    else:
        st.success("No anomalies detected in the current window.")

    st.warning(
        "⚠ Anomalies are labelled as **Possible abnormal consumption / possible non-technical loss**. "
        "They are NOT automatically classified as electricity theft. "
        "Ground-truth theft data is required for that determination."
    )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════
with tabs[4]:
    section_header("Forecast & Analytics", "SHAP explainability · Feature importance · Peak cause")

    shap_z = st.selectbox("Select Zone for SHAP", ["Residential","Commercial","Industrial"],
                          key="shap_zone")
    zexp   = shap_exp.get(shap_z,{})
    tfacs  = zexp.get("top_factors",[])

    if peak_cause.get("peak_cause_explanation"):
        st.markdown(
            f"<div style='background:{CARD};border:1px solid {BORDER};border-left:4px solid {ORANGE};"
            f"border-radius:8px;padding:14px 18px;margin-bottom:14px'>"
            f"<div style='font-size:11px;font-weight:700;color:{ORANGE};text-transform:uppercase;"
            f"letter-spacing:0.08em;margin-bottom:6px'>⚡ Peak Cause Explanation</div>"
            f"<div style='font-size:13px;color:{TEXT2}'>{peak_cause['peak_cause_explanation']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    cl, cr = st.columns(2)
    with cl:
        section_header("Top Influencing Factors")
        if tfacs:
            for fac in tfacs:
                d   = fac.get("direction","neutral")
                imp = fac.get("impact","Low")
                fc2 = RED if d=="positive" else GREEN
                bp  = min(abs(fac.get("shap_value",0))/20*100, 100)
                arr = "▲" if d=="positive" else "▼"
                st.markdown(
                    f"<div style='margin-bottom:10px'>"
                    f"<div style='display:flex;justify-content:space-between;margin-bottom:3px'>"
                    f"<span style='font-size:13px;font-weight:600;color:{TEXT}'>{fac['feature']}</span>"
                    f"<span style='font-size:12px;color:{fc2};font-weight:700'>{arr} {imp}</span>"
                    f"</div>"
                    f"<div style='background:{BORDER_DIM};border-radius:3px;height:5px'>"
                    f"<div style='background:{fc2};width:{bp:.0f}%;height:5px;border-radius:3px'></div>"
                    f"</div>"
                    f"<div style='font-size:11px;color:{MUTED};margin-top:2px'>"
                    f"{fac.get('plain_explanation','')}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No SHAP data available.")

    with cr:
        section_header("Feature Importance (|SHAP|)")
        if tfacs:
            idf = pd.DataFrame([{"Feature":f["feature"],"|SHAP|":abs(f["shap_value"]),
                                  "Direction":f["direction"]} for f in tfacs])
            fig5 = px.bar(idf, x="|SHAP|", y="Feature", orientation="h",
                          color="Direction",
                          color_discrete_map={"positive":RED,"negative":GREEN})
            fig5.update_layout(**chart_cfg(300), showlegend=False)
            fig5.update_traces(texttemplate="%{x:.2f}", textposition="outside",
                               textfont=dict(color=TEXT2, size=11))
            st.plotly_chart(fig5, use_container_width=True)

    if zexp.get("technical_note"):
        st.caption("ⓘ " + zexp["technical_note"])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 6 — RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════
with tabs[5]:
    section_header("Recommendations",
                   "Rule-based adaptive recommendations — "+str(len(recommendations))+" active")

    if not recommendations:
        st.info("No recommendations generated for current conditions.")

    cat_colors = {
        "Emergency": RED, "Demand Response": ORANGE,
        "Solar Optimisation": SOLAR, "Supply Preparation": BLUE,
        "Monitoring": "#39C5CF", "Maintenance": GREEN,
        "Anomaly Response": RED,
    }
    for rec in recommendations:
        cat  = rec.get("category","General")
        rc   = cat_colors.get(cat, BLUE)
        pri  = rec.get("priority","?")
        act  = rec.get("action","")
        rsn  = rec.get("reason","")
        st.markdown(
            f"<div style='background:{CARD};border:1px solid {BORDER_DIM};"
            f"border-left:4px solid {rc};border-radius:10px;"
            f"padding:16px 18px;margin-bottom:10px'>"
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:8px'>"
            f"<span style='font-size:10px;font-weight:700;color:{MUTED};"
            f"text-transform:uppercase;letter-spacing:0.08em'>Priority {pri}</span>"
            f"<span style='background:{rc}22;color:{rc};padding:3px 10px;"
            f"border-radius:12px;font-size:11px;font-weight:700'>{cat}</span>"
            f"</div>"
            f"<div style='font-size:14px;font-weight:700;color:{TEXT};margin-bottom:6px'>{act}</div>"
            f"<div style='font-size:12px;color:{MUTED}'>"
            f"<strong style='color:{TEXT2}'>Why:</strong> {rsn}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 7 — MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
with tabs[6]:
    section_header("Model Comparison", "GRU (XGBoost fallback) vs XGBoost Baseline — held-out test set metrics")

    crow = metrics_data.get("demand_model_comparison",[])
    cdf  = pd.DataFrame(crow)
    dc2  = [c for c in ["zone","model","MAE","RMSE","MAPE","R2"] if c in cdf.columns]
    cdf  = cdf[dc2].rename(columns={"zone":"Zone","model":"Model",
                                     "MAE":"MAE (MW)","RMSE":"RMSE (MW)",
                                     "MAPE":"MAPE (%)","R2":"R²"})
    st.dataframe(cdf, use_container_width=True, hide_index=True)
    st.info(metrics_data.get("data_source_note","All metrics from actual model evaluation on the test set."))

    if not cdf.empty and "MAE (MW)" in cdf.columns:
        mdf = cdf.dropna(subset=["MAE (MW)"])
        if not mdf.empty:
            fig6 = px.bar(mdf, x="Zone", y="MAE (MW)", color="Model", barmode="group",
                          color_discrete_map={"GRU":BLUE,"XGBoost Baseline":ORANGE},
                          text="MAE (MW)")
            fig6.update_traces(texttemplate="%{text:.2f}", textposition="outside",
                               textfont=dict(color=TEXT2, size=11))
            fig6.update_layout(**chart_cfg(270))
            st.plotly_chart(fig6, use_container_width=True)

    sm = metrics_data.get("solar_model",{}).get("metrics",{})
    if sm and "MAE" in sm:
        section_header("Solar Model — XGBoost")
        s1,s2,s3,s4 = st.columns(4)
        with s1: kpi_card("MAE",  sm.get("MAE"), "MW", SOLAR)
        with s2: kpi_card("RMSE", sm.get("RMSE"),"MW", SOLAR)
        with s3: kpi_card("MAPE", sm.get("MAPE"),"%",  SOLAR)
        with s4: kpi_card("R²",   sm.get("R2"),  "",   SOLAR)

    section_header("Metric Definitions")
    for k,v in metrics_data.get("metric_definitions",{}).items():
        st.markdown(f"**{k}** — {v}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 8 — WHAT-IF SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════
with tabs[7]:
    section_header("What-If Peak Mitigation Simulator",
                   "Scenario simulation — NOT a new ML prediction")
    st.markdown(
        "<div class='sim-warn'>⚠ This simulator applies proportional adjustments to the "
        "Baseline Forecast. All outputs are clearly labelled 'Simulated What-If Scenario'. "
        "Results are for decision support only.</div>",
        unsafe_allow_html=True,
    )

    from config.settings import SIMULATOR_LIMITS
    ctrl, res = st.columns([1,2])

    with ctrl:
        section_header("Scenario Parameters")
        slabels = {
            "industrial_shift_pct":      "Industrial Load Shift (%)",
            "commercial_shift_pct":      "Commercial Load Shift (%)",
            "residential_reduction_pct": "Residential Reduction (%)",
            "solar_increase_pct":        "Solar Utilisation Increase (%)",
        }
        params = {}
        for key, lim in SIMULATOR_LIMITS.items():
            params[key] = st.slider(slabels.get(key,key),
                                    float(lim["min"]), float(lim["max"]),
                                    float(lim["default"]), 1.0,
                                    key="sl_"+key)
        run_btn = st.button("▶  RUN SIMULATION", type="primary", use_container_width=True)

    if run_btn or "sim_result" in st.session_state:
        if run_btn:
            from backend.simulation.what_if_simulator import run_simulation
            zd = {z: zone_fc[z]["predicted_demand_mw"]
                  for z in zone_fc if "predicted_demand_mw" in zone_fc[z]}
            sr = run_simulation(
                baseline_demand_24h=total_demand,
                baseline_solar_24h=solar_24h,
                industrial_shift_pct=params["industrial_shift_pct"],
                commercial_shift_pct=params["commercial_shift_pct"],
                residential_reduction_pct=params["residential_reduction_pct"],
                solar_increase_pct=params["solar_increase_pct"],
                zone_demands=zd, timestamps=ts_list,
            )
            st.session_state["sim_result"] = sr
        else:
            sr = st.session_state["sim_result"]

        with res:
            section_header("Simulation Results")
            st.markdown(
                f"<span style='background:{BLUE}22;color:{BLUE};padding:3px 10px;"
                f"border-radius:10px;font-size:11px;font-weight:700'>BASELINE FORECAST</span> "
                f"<span style='background:{PURPLE}22;color:{PURPLE};padding:3px 10px;"
                f"border-radius:10px;font-size:11px;font-weight:700;"
                f"margin-left:6px'>SIMULATED SCENARIO</span>",
                unsafe_allow_html=True,
            )
            st.markdown("<div style='margin:8px 0'></div>", unsafe_allow_html=True)

            b  = sr["baseline"];  s2 = sr["simulated"];  imp = sr["impact"]
            cd = {
                "Metric":    ["Peak Demand (MW)","Solar (MW)","Net Demand (MW)","Peak Time","Severity"],
                "Baseline":  [b["peak_demand_mw"],b["peak_solar_mw"],b["peak_net_demand_mw"],b["peak_timestamp"],b["severity"]],
                "Simulated": [s2["peak_demand_mw"],s2["peak_solar_mw"],s2["peak_net_demand_mw"],s2["peak_timestamp"],s2["severity"]],
            }
            st.dataframe(pd.DataFrame(cd), use_container_width=True, hide_index=True)

            i1,i2,i3 = st.columns(3)
            with i1: kpi_card("Peak Reduction", imp["peak_reduction_mw"], "MW", GREEN)
            with i2: kpi_card("Reduction %",    imp["peak_reduction_pct"], "%", GREEN)
            with i3: kpi_card("Severity Change",imp["severity_improvement"], "", PURPLE)

            st.info("**Recommended:** " + imp["recommended_scenario"])

            bnet = b.get("net_24h",[]); snet = s2.get("net_24h",[])
            shrs = [t.split(" ")[1] if " " in t else t for t in ts_list]
            fig7 = go.Figure()
            fig7.add_trace(go.Scatter(x=shrs[:len(bnet)], y=bnet,
                name="Baseline Forecast", line=dict(color=BLUE, width=2)))
            fig7.add_trace(go.Scatter(x=shrs[:len(snet)], y=snet,
                name="Simulated Scenario", line=dict(color=PURPLE, width=2, dash="dash")))
            fig7.update_layout(**chart_cfg(300))
            st.plotly_chart(fig7, use_container_width=True)
            st.caption("⚠ " + sr.get("disclaimer",""))
    else:
        with res:
            st.info("Set parameters and click **RUN SIMULATION** to see results.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 9 — REPORTS
# ═══════════════════════════════════════════════════════════════════════════
with tabs[8]:
    section_header("Decision Support Report",
                   "Generate and download a complete JSON report")

    cr1, cr2 = st.columns(2)
    with cr1:
        inc_sim = st.checkbox("Include latest simulation results", value=True)
    with cr2:
        if st.button("📄  Generate Report", type="primary"):
            report = {
                "report_type":  "PowerAI Decision Support Report",
                "generated_at": datetime.utcnow().isoformat(),
                "data_note": "Synthetic/demo data. Not real operational data.",
                "sections": {
                    "kpi_summary":    kpi,
                    "forecast_24h":   {"total_demand": total_demand,
                                       "solar": solar_24h, "net_demand": net_demand,
                                       "timestamps": ts_list, "confidence": fc_conf},
                    "zone_forecasts": {z:{"peak_mw":fc.get("peak_mw"),
                                          "peak_hour":fc.get("peak_hour_offset")}
                                       for z,fc in zone_fc.items()},
                    "solar_load_match": solar_match,
                    "peak_severity":  {"level": peak_sev.get("severity",{}).get("level"),
                                       "peak_net_mw": peak_sev.get("peak_net_demand_mw"),
                                       "peak_time": peak_sev.get("peak_timestamp")},
                    "anomalies":       anomalies,
                    "shap_explanation": peak_cause,
                    "recommendations": recommendations,
                    "model_comparison": metrics_data.get("demand_model_comparison",[]),
                },
            }
            if inc_sim and "sim_result" in st.session_state:
                sim = st.session_state["sim_result"]
                report["sections"]["simulation"] = {
                    "label": "Scenario Simulation — What-If Run",
                    "note":  "Simulated values are NOT actual or predicted values.",
                    "baseline_peak_mw":  sim["baseline"]["peak_net_demand_mw"],
                    "simulated_peak_mw": sim["simulated"]["peak_net_demand_mw"],
                    "peak_reduction_mw": sim["impact"]["peak_reduction_mw"],
                    "peak_reduction_pct":sim["impact"]["peak_reduction_pct"],
                    "scenario_inputs":   sim.get("scenario_inputs",{}),
                    "recommended":       sim["impact"]["recommended_scenario"],
                }
            else:
                report["sections"]["simulation"] = {
                    "note":"No simulation run. Use the What-If Simulator tab first."
                }
            st.session_state["report"] = report
            st.success("✓ Report generated!")

    if "report" in st.session_state:
        rpt  = st.session_state["report"]
        rjson= json.dumps(rpt, indent=2, default=str)
        st.download_button(
            label="⬇  Download Report (JSON)",
            data=rjson,
            file_name="powerai-report-"+datetime.now().strftime("%Y-%m-%d")+".json",
            mime="application/json",
        )
        st.json(rpt)
