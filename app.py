"""
5G Cell Congestion Predictor — Streamlit Dashboard  v3.0
=========================================================
File : app.py
Run  : streamlit run app.py

New in v3.0
-----------
• Executive Summary section (Section 1) at top of every run
• KPI summary cards with icons
• Pie charts: scenario distribution, congested vs healthy, recommendation breakdown
• Heatmaps: hour×day congestion intensity, cell congestion intensity
• Trend charts: PRB trend, RRC trend, congestion probability trend
• Cluster view: source cell + top-4 neighbours with congestion state
• Fixed scenario classification (70% threshold, LB priority)
• MW-hour exclusion retained
• Cleaner, less-cluttered layout
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ml_pipeline import (
    validate_kpi_data, engineer_features, define_target,
    train_model, get_feature_cols,
    generate_future_timestamps, build_prediction_input,
    predict_congestion, detect_recurring_patterns, MW_HOURS,
)
from neighbor_engine import (
    build_neighbor_map, build_ho_weight_map,
    classify_scenario, attach_recommendations,
    get_scenario_summary, auto_generate_tickets,
    get_neighbor_utilization, RECOMMENDATIONS,
)

# ══════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="5G Congestion Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════
# COLOUR CONSTANTS
# ══════════════════════════════════════════════════════
CLR_CLUSTER = "#e53935"
CLR_LB      = "#fb8c00"
CLR_NORMAL  = "#43a047"
CLR_NODATA  = "#8e24aa"
SCENARIO_COLORS = {
    "CLUSTER_CONGESTION":          CLR_CLUSTER,
    "LOAD_BALANCING_OPPORTUNITY":  CLR_LB,
    "NORMAL":                      CLR_NORMAL,
    "CONGESTED_NO_NEIGHBOR_DATA":  CLR_NODATA,
}

# ══════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
  .main-header {
      background: linear-gradient(135deg,#0a0f2c 0%,#1a237e 50%,#01579b 100%);
      padding:1.5rem 2rem; border-radius:12px; margin-bottom:1.2rem;
      text-align:center; color:white;
  }
  .section-header {
      background:#1a237e; color:white; padding:.5rem 1rem;
      border-radius:8px; margin:1.2rem 0 .6rem; font-size:1.1rem; font-weight:700;
  }
  .exec-card {
      border-radius:10px; padding:1rem 1.2rem; margin:.4rem 0;
      border-left:6px solid; font-size:.95rem;
  }
  .card-cluster { background:#ffebee; border-color:#e53935; }
  .card-lb      { background:#fff3e0; border-color:#fb8c00; }
  .card-green   { background:#e8f5e9; border-color:#43a047; }
  .card-blue    { background:#e3f2fd; border-color:#1565c0; }
  .kpi-metric   {
      text-align:center; background:#f5f5f5; border-radius:8px;
      padding:.7rem .5rem;
  }
  .kpi-val   { font-size:1.8rem; font-weight:800; color:#1a237e; }
  .kpi-label { font-size:.78rem; color:#555; margin-top:.2rem; }
  .ticket-card {
      background:#fff3e0; border-left:6px solid #e65100;
      padding:1rem; border-radius:8px; margin:.5rem 0;
  }
  .ticket-id { font-size:1rem; font-weight:bold; color:#b71c1c; }
  .insight-box {
      background:#e3f2fd; border-left:4px solid #1565c0;
      padding:.5rem .9rem; border-radius:6px; margin:.25rem 0; font-size:.88rem;
  }
  .warn-box { background:#fff8e1; border-left:4px solid #f57f17; padding:.5rem .9rem; border-radius:6px; margin:.25rem 0; }
  .err-box  { background:#ffebee; border-left:4px solid #b71c1c; padding:.5rem .9rem; border-radius:6px; margin:.25rem 0; }
  .healthy-badge  { color:#fff; background:#43a047; border-radius:4px; padding:2px 8px; font-size:.8rem; }
  .danger-badge   { color:#fff; background:#e53935; border-radius:4px; padding:2px 8px; font-size:.8rem; }
  .warning-badge  { color:#fff; background:#fb8c00; border-radius:4px; padding:2px 8px; font-size:.8rem; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════
def metric_card(value, label, bg="#f5f5f5", color="#1a237e"):
    st.markdown(
        f'<div class="kpi-metric" style="background:{bg};">'
        f'<div class="kpi-val" style="color:{color};">{value}</div>'
        f'<div class="kpi-label">{label}</div></div>',
        unsafe_allow_html=True,
    )


def section_header(title):
    st.markdown(f'<div class="section-header">🔹 {title}</div>', unsafe_allow_html=True)


def exec_card(text, card_class="card-blue"):
    st.markdown(f'<div class="exec-card {card_class}">{text}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
  <h1>📡 5G Cell Congestion Predictor <span style="font-size:.55em;opacity:.8">v3.0</span></h1>
  <p style="font-size:.95rem;opacity:.9;">
    XGBoost · Dual-threshold congestion · HO-weighted neighbour intelligence ·
    Cluster vs Load-Balancing classification · Executive Dashboard
  </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        """
        <div style="width:60px;height:60px;margin:.25rem 0 .75rem;">
          <svg viewBox="0 0 64 64" role="img" aria-label="Wireless network icon" style="width:60px;height:60px;display:block;">
            <circle cx="32" cy="46" r="5" fill="#1a237e"/>
            <path d="M20 36a17 17 0 0 1 24 0" fill="none" stroke="#1565c0" stroke-width="5" stroke-linecap="round"/>
            <path d="M10 26a31 31 0 0 1 44 0" fill="none" stroke="#43a047" stroke-width="5" stroke-linecap="round"/>
            <path d="M4 16a40 40 0 0 1 56 0" fill="none" stroke="#fb8c00" stroke-width="5" stroke-linecap="round"/>
          </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.title("⚙️ Configuration")

    st.subheader("📂 Data Upload")
    kpi_file = st.file_uploader("KPI Data (CSV)", type=["csv"], key="kpi")
    nbr_file = st.file_uploader("Neighbour Data (CSV)", type=["csv"], key="nbr")

    st.subheader("🎚️ Congestion Thresholds")
    prb_thr = st.slider("PRB Threshold (%)", 60, 95, 85, 1)
    rrc_thr = st.slider("RRC Users Threshold", 100, 500, 300, 10)

    st.subheader("🛠️ Options")
    excl_mw  = st.checkbox("Exclude MW Hours (01:00–05:00)", value=True)
    pred_hrs = st.slider("Prediction Window (hours)", 24, 168, 168, 24,
                         help="168h = 1 week (recommended for executive summary)")

    st.markdown("---")
    st.markdown("""
    **Scenario Logic (v3.0)**
    - 🔴 **Cluster Congestion**: Source congested + ≥70% neighbours congested
    - 🟠 **Load Balancing**: Source congested + ≥2 healthy neighbours
    - 🟢 **Normal**: Source not congested
    """)

    run_btn = st.button("🚀 Run Prediction", type="primary", use_container_width=True)

# ══════════════════════════════════════════════════════
# EARLY EXIT IF NO DATA
# ══════════════════════════════════════════════════════
if not run_btn or kpi_file is None:
    st.info("👈 Upload KPI CSV and click **Run Prediction** to begin.")
    with st.expander("📋 Required CSV format"):
        st.code("timestamp,cell_name,gnb_name,prb_utilization,avg_rrc_users,throughput_mbps\n"
                "2025-03-01 09:00:00,Cell_01,gNB_01,87.5,315,450.2")
    st.stop()

# ══════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════
with st.spinner("Loading data…"):
    kpi_raw = pd.read_csv(kpi_file)
    nbr_df  = pd.read_csv(nbr_file) if nbr_file else pd.DataFrame()

# ── Validate ─────────────────────────────────────────
val = validate_kpi_data(kpi_raw)
if not val["ok"]:
    for e in val["errors"]:
        st.markdown(f'<div class="err-box">❌ {e}</div>', unsafe_allow_html=True)
    st.stop()
for w in val["warnings"]:
    st.markdown(f'<div class="warn-box">⚠️ {w}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# FEATURE ENGINEERING + MODEL TRAINING
# ══════════════════════════════════════════════════════
with st.spinner("Engineering features & training model…"):
    df_fe, le_cell, le_gnb = engineer_features(kpi_raw, exclude_mw_hours=excl_mw)
    df_fe = define_target(df_fe, prb_thr, rrc_thr)
    model, metrics, feat_imp, feat_cols, model_name = train_model(df_fe)

# ══════════════════════════════════════════════════════
# GENERATE PREDICTIONS (1 WEEK)
# ══════════════════════════════════════════════════════
with st.spinner("Running 7-day predictions…"):
    cells    = kpi_raw["cell_name"].unique().tolist()
    fut_df   = generate_future_timestamps(pred_hrs)
    pred_in  = build_prediction_input(
        fut_df, cells, kpi_raw, le_cell, le_gnb,
        exclude_mw_hours=excl_mw, prb_threshold=prb_thr, rrc_threshold=rrc_thr,
    )
    pred_out = predict_congestion(model, pred_in, feat_cols)

# ── Build neighbour intelligence ─────────────────────
nbr_map, ho_map = {}, {}
if not nbr_df.empty:
    nbr_map = build_neighbor_map(nbr_df)
    ho_map  = build_ho_weight_map(nbr_df)

pred_classified = classify_scenario(
    pred_out, nbr_map, ho_map,
    kpi_df=kpi_raw, prb_threshold=prb_thr, rrc_threshold=rrc_thr,
)
pred_classified = attach_recommendations(pred_classified)
tickets         = auto_generate_tickets(pred_classified)
patterns        = detect_recurring_patterns(kpi_raw, prb_thr)

# ══════════════════════════════════════════════════════════════════════
# SECTION 1 — EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════
section_header("SECTION 1 — EXECUTIVE SUMMARY (Next 7 Days)")

cong_pred   = pred_classified[pred_classified["congestion_flag"] == 1]
cluster_df  = pred_classified[pred_classified["scenario_type"] == "CLUSTER_CONGESTION"]
lb_df       = pred_classified[pred_classified["scenario_type"] == "LOAD_BALANCING_OPPORTUNITY"]
normal_df   = pred_classified[pred_classified["scenario_type"] == "NORMAL"]

n_cong_cells    = cong_pred["cell_name"].nunique()
n_cluster_cells = cluster_df["cell_name"].nunique()
n_lb_cells      = lb_df["cell_name"].nunique()
n_healthy_cells = len(cells) - n_cong_cells

# Busiest day/hour
if not cong_pred.empty:
    cong_pred_c = cong_pred.copy()
    cong_pred_c["timestamp"] = pd.to_datetime(cong_pred_c["timestamp"])
    cong_pred_c["day_name"]  = cong_pred_c["timestamp"].dt.day_name()
    cong_pred_c["hour_str"]  = cong_pred_c["timestamp"].dt.strftime("%H:00")
    busiest_day  = cong_pred_c["day_name"].value_counts().idxmax()
    busiest_hour = cong_pred_c["hour_str"].value_counts().idxmax()
else:
    busiest_day  = "N/A"
    busiest_hour = "N/A"

# ── KPI Cards ───────────────────────────────────────
st.markdown("#### 📊 Prediction KPI Summary")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    metric_card(n_cong_cells, "Congested Cells", "#ffebee", "#e53935")
with c2:
    metric_card(n_cluster_cells, "Cluster Congestion", "#ffebee", "#b71c1c")
with c3:
    metric_card(n_lb_cells, "Load Balancing Opps", "#fff3e0", "#e65100")
with c4:
    metric_card(busiest_day, "Busiest Day", "#e3f2fd", "#1565c0")
with c5:
    metric_card(busiest_hour, "Busiest Hour", "#e3f2fd", "#1565c0")

st.markdown("---")

# ── Narrative Summary Cards ─────────────────────────
st.markdown("#### 📋 Next 1 Week Congestion Summary")
col_exec1, col_exec2 = st.columns(2)

with col_exec1:
    if n_lb_cells > 0:
        exec_card(
            f"🟠 <b>{n_lb_cells} cells</b> predicted congested — "
            f"<b>Load Balancing opportunity available</b>. "
            f"Activate MLB / CIO tuning to redistribute traffic.",
            "card-lb"
        )
    else:
        exec_card("✅ No load-balancing opportunities detected this week.", "card-green")

    if n_cluster_cells > 0:
        cluster_gnbs = cluster_df["gnb_name"].nunique() if "gnb_name" in cluster_df else "?"
        exec_card(
            f"🔴 <b>{n_cluster_cells} cells</b> in <b>{cluster_gnbs} cluster(s)</b> "
            f"predicted cluster-congested — <b>Capacity upgrade required urgently.</b>",
            "card-cluster"
        )
    else:
        exec_card("✅ No cluster-wide congestion predicted this week.", "card-green")

with col_exec2:
    exec_card(
        f"📈 <b>Busiest predicted window:</b> {busiest_day}s at {busiest_hour}. "
        f"Ensure proactive resource allocation before this window.",
        "card-blue"
    )
    exec_card(
        f"🔋 <b>{n_healthy_cells} of {len(cells)} cells</b> predicted healthy. "
        f"These cells can absorb overflow traffic from congested neighbours.",
        "card-green"
    )

# ── Ticket Preview ───────────────────────────────────
if tickets:
    with st.expander(f"🎫 {len(tickets)} Auto-Generated Capacity Tickets"):
        for t in tickets[:5]:
            st.markdown(
                f'<div class="ticket-card">'
                f'<span class="ticket-id">{t["ticket_id"]}</span> '
                f'— Severity: <b>{t["severity"]}</b> | Cluster: <b>{t["impacted_cluster"]}</b><br>'
                f'Cells: {", ".join(t["impacted_cells"][:5])}<br>'
                f'<small>{t["reason"]}</small>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════
section_header("SECTION 2 — VISUALIZATIONS")

tab1, tab2, tab3, tab4 = st.tabs(["🥧 Distribution", "🔥 Heatmaps", "📈 Trends", "🗺️ Cluster View"])

# ── TAB 1: PIE CHARTS ───────────────────────────────
with tab1:
    st.markdown("#### Scenario & Cell Health Distribution")
    pie1, pie2, pie3 = st.columns(3)

    # Pie 1: Scenario distribution
    scen_counts = pred_classified["scenario_type"].value_counts().reset_index()
    scen_counts.columns = ["Scenario","Count"]
    color_map = {
        "CLUSTER_CONGESTION":         CLR_CLUSTER,
        "LOAD_BALANCING_OPPORTUNITY": CLR_LB,
        "NORMAL":                     CLR_NORMAL,
        "CONGESTED_NO_NEIGHBOR_DATA": CLR_NODATA,
    }
    label_map = {
        "CLUSTER_CONGESTION":         "Cluster Congestion",
        "LOAD_BALANCING_OPPORTUNITY": "Load Balancing Opp",
        "NORMAL":                     "Normal",
        "CONGESTED_NO_NEIGHBOR_DATA": "No Neighbor Data",
    }
    scen_counts["Label"] = scen_counts["Scenario"].map(label_map)
    scen_counts["Color"] = scen_counts["Scenario"].map(color_map)

    with pie1:
        fig = px.pie(
            scen_counts, names="Label", values="Count",
            color="Scenario", color_discrete_map=color_map,
            title="Scenario Distribution",
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(showlegend=False, margin=dict(t=40,b=10,l=10,r=10), height=320)
        st.plotly_chart(fig, use_container_width=True)

    # Pie 2: Congested vs Healthy cells
    cell_health = pred_classified.groupby("cell_name")["congestion_flag"].max().reset_index()
    cong_count  = (cell_health["congestion_flag"] == 1).sum()
    hlth_count  = (cell_health["congestion_flag"] == 0).sum()
    with pie2:
        fig2 = go.Figure(go.Pie(
            labels=["Congested","Healthy"],
            values=[cong_count, hlth_count],
            marker_colors=[CLR_CLUSTER, CLR_NORMAL],
            textposition="inside", textinfo="percent+label",
        ))
        fig2.update_layout(title="Congested vs Healthy Cells", showlegend=False,
                           margin=dict(t=40,b=10,l=10,r=10), height=320)
        st.plotly_chart(fig2, use_container_width=True)

    # Pie 3: Recommendation breakdown
    cong_scen = pred_classified[pred_classified["congestion_flag"]==1]
    rec_counts = cong_scen["scenario_type"].value_counts().reset_index()
    rec_counts.columns = ["Scenario","Count"]
    rec_counts["Action"] = rec_counts["Scenario"].map({
        "CLUSTER_CONGESTION":         "Capacity Upgrade",
        "LOAD_BALANCING_OPPORTUNITY": "MLB / CIO Tuning",
        "CONGESTED_NO_NEIGHBOR_DATA": "Upload Nbr Data",
    })
    with pie3:
        if not rec_counts.empty:
            fig3 = px.pie(
                rec_counts, names="Action", values="Count",
                color="Scenario", color_discrete_map=color_map,
                title="Recommendation Breakdown",
            )
            fig3.update_traces(textposition="inside", textinfo="percent+label")
            fig3.update_layout(showlegend=False, margin=dict(t=40,b=10,l=10,r=10), height=320)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No congested cells to recommend on.")

# ── TAB 2: HEATMAPS ─────────────────────────────────
with tab2:
    st.markdown("#### Congestion Heatmaps")
    kpi_fe = kpi_raw.copy()
    kpi_fe["timestamp"]   = pd.to_datetime(kpi_fe["timestamp"])
    kpi_fe["hour"]        = kpi_fe["timestamp"].dt.hour
    kpi_fe["day_of_week"] = kpi_fe["timestamp"].dt.dayofweek
    kpi_fe["day_name"]    = kpi_fe["timestamp"].dt.day_name()
    kpi_fe["congested"]   = ((kpi_fe["prb_utilization"] > prb_thr) |
                              (kpi_fe["avg_rrc_users"]  > rrc_thr)).astype(int)

    h1, h2 = st.columns(2)

    # Heatmap 1: Hour × Day congestion rate
    with h1:
        pivot = (
            kpi_fe.groupby(["day_of_week","hour"])["congested"].mean() * 100
        ).reset_index().rename(columns={"congested":"congestion_rate"})
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        pivot["day_name"] = pivot["day_of_week"].map(
            {i:d for i,d in enumerate(dow_order)}
        )
        hm_pivot = pivot.pivot(index="day_name", columns="hour", values="congestion_rate")
        hm_pivot = hm_pivot.reindex(dow_order)
        fig_h1 = px.imshow(
            hm_pivot, color_continuous_scale="RdYlGn_r",
            labels={"color":"Congestion %"},
            title="Congestion Rate by Hour & Day (%)",
            aspect="auto",
        )
        fig_h1.update_layout(height=320, margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig_h1, use_container_width=True)

    # Heatmap 2: Cell congestion intensity
    with h2:
        cell_hour = (
            kpi_fe.groupby(["cell_name","hour"])["prb_utilization"].mean().reset_index()
        )
        cell_pivot = cell_hour.pivot(index="cell_name", columns="hour", values="prb_utilization")
        fig_h2 = px.imshow(
            cell_pivot, color_continuous_scale="RdYlGn_r",
            labels={"color":"Avg PRB %"},
            title="Cell PRB Utilization by Hour",
            aspect="auto",
        )
        fig_h2.update_layout(height=320, margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig_h2, use_container_width=True)

# ── TAB 3: TRENDS ───────────────────────────────────
with tab3:
    st.markdown("#### PRB & RRC Trends + Congestion Probability")
    cells_list = sorted(kpi_raw["cell_name"].unique())
    selected_cell = st.selectbox("Select Cell for Trend View", cells_list)

    cell_hist = kpi_raw[kpi_raw["cell_name"] == selected_cell].copy()
    cell_hist["timestamp"] = pd.to_datetime(cell_hist["timestamp"])
    cell_hist = cell_hist.sort_values("timestamp")

    if excl_mw:
        cell_hist = cell_hist[~cell_hist["timestamp"].dt.hour.isin(MW_HOURS)]

    # PRB trend
    fig_prb = go.Figure()
    fig_prb.add_trace(go.Scatter(
        x=cell_hist["timestamp"], y=cell_hist["prb_utilization"],
        mode="lines", name="PRB %", line=dict(color="#1565c0", width=1.5),
    ))
    fig_prb.add_hline(y=prb_thr, line_dash="dash", line_color="red",
                      annotation_text=f"Threshold ({prb_thr}%)")
    fig_prb.update_layout(title=f"PRB Utilization — {selected_cell}", height=260,
                          margin=dict(t=40,b=30,l=10,r=10),
                          yaxis_title="PRB %", xaxis_title="")
    st.plotly_chart(fig_prb, use_container_width=True)

    # RRC trend
    fig_rrc = go.Figure()
    fig_rrc.add_trace(go.Scatter(
        x=cell_hist["timestamp"], y=cell_hist["avg_rrc_users"],
        mode="lines", name="RRC Users", line=dict(color="#fb8c00", width=1.5),
    ))
    fig_rrc.add_hline(y=rrc_thr, line_dash="dash", line_color="red",
                      annotation_text=f"Threshold ({rrc_thr})")
    fig_rrc.update_layout(title=f"RRC Users — {selected_cell}", height=260,
                          margin=dict(t=40,b=30,l=10,r=10),
                          yaxis_title="Avg RRC Users", xaxis_title="")
    st.plotly_chart(fig_rrc, use_container_width=True)

    # Congestion probability trend (from predictions)
    cell_pred = pred_classified[pred_classified["cell_name"] == selected_cell].copy()
    cell_pred["timestamp"] = pd.to_datetime(cell_pred["timestamp"])
    cell_pred = cell_pred.sort_values("timestamp")

    if not cell_pred.empty:
        fig_prob = go.Figure()
        fig_prob.add_trace(go.Scatter(
            x=cell_pred["timestamp"], y=cell_pred["congestion_probability"] * 100,
            mode="lines+markers", name="Congestion Prob %",
            line=dict(color="#e53935", width=1.5),
            marker=dict(size=4),
        ))
        fig_prob.add_hline(y=50, line_dash="dash", line_color="gray",
                           annotation_text="Decision Threshold (50%)")
        fig_prob.update_layout(
            title=f"Predicted Congestion Probability — {selected_cell} (Next {pred_hrs}h)",
            height=260, margin=dict(t=40,b=30,l=10,r=10),
            yaxis_title="Probability (%)", yaxis_range=[0,100],
        )
        st.plotly_chart(fig_prob, use_container_width=True)

# ── TAB 4: CLUSTER VIEW ─────────────────────────────
with tab4:
    st.markdown("#### Cluster Topology: Source Cell + Top-4 Neighbours")

    cong_cells = sorted(
        pred_classified[pred_classified["congestion_flag"] == 1]["cell_name"].unique()
    )
    if not cong_cells:
        st.info("No congested cells predicted in the selected window.")
    else:
        sel_clust = st.selectbox("Select Congested Source Cell", cong_cells, key="clust_sel")

        nbrs_of_sel = nbr_map.get(sel_clust, [])
        all_clust_cells = [sel_clust] + nbrs_of_sel

        cell_state = {}
        for c in all_clust_cells:
            rows = pred_classified[pred_classified["cell_name"] == c]
            if rows.empty:
                cell_state[c] = "UNKNOWN"
            else:
                cong_rate = rows["congestion_flag"].mean()
                if cong_rate >= 0.5:
                    cell_state[c] = "CONGESTED"
                else:
                    cell_state[c] = "HEALTHY"

        # Source cell scenario
        src_scenario = pred_classified[
            pred_classified["cell_name"] == sel_clust
        ]["scenario_type"].value_counts().idxmax()

        scenario_label = {
            "CLUSTER_CONGESTION":         "🔴 Cluster Congestion",
            "LOAD_BALANCING_OPPORTUNITY": "🟠 Load Balancing Opportunity",
            "NORMAL":                     "🟢 Normal",
            "CONGESTED_NO_NEIGHBOR_DATA": "🟣 No Neighbor Data",
        }.get(src_scenario, src_scenario)

        st.markdown(f"**Scenario:** {scenario_label}")

        # Draw cluster as a simple visual table
        cols = st.columns(5)
        for i, cell in enumerate(all_clust_cells[:5]):
            state  = cell_state.get(cell, "UNKNOWN")
            is_src = cell == sel_clust
            bg     = "#ffcdd2" if state == "CONGESTED" else "#c8e6c9"
            border = "4px solid #e53935" if is_src else "2px solid #aaa"
            label  = "📡 SOURCE" if is_src else "📶 NEIGHBOUR"
            badge  = f'<span class="danger-badge">CONGESTED</span>' if state=="CONGESTED" \
                     else f'<span class="healthy-badge">HEALTHY</span>'

            # Historical avg PRB
            hist_prb = kpi_raw[kpi_raw["cell_name"]==cell]["prb_utilization"].mean()
            hist_rrc = kpi_raw[kpi_raw["cell_name"]==cell]["avg_rrc_users"].mean()

            with cols[i]:
                st.markdown(
                    f'<div style="background:{bg};border:{border};border-radius:10px;'
                    f'padding:.8rem;text-align:center;">'
                    f'<div style="font-size:.7rem;color:#555;">{label}</div>'
                    f'<div style="font-size:1.1rem;font-weight:800;">{cell}</div>'
                    f'{badge}<br>'
                    f'<div style="font-size:.78rem;margin-top:.4rem;">'
                    f'PRB: <b>{hist_prb:.1f}%</b><br>RRC: <b>{hist_rrc:.0f}</b>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

        # Neighbour utilisation bar chart
        if nbrs_of_sel:
            nbr_util = get_neighbor_utilization(sel_clust, kpi_raw, nbr_map)
            if not nbr_util.empty:
                nbr_util["Role"] = nbr_util["cell_name"].apply(
                    lambda x: "Source" if x == sel_clust else "Neighbour"
                )
                fig_nbr = px.bar(
                    nbr_util, x="cell_name", y=["avg_prb","max_prb"],
                    barmode="group",
                    color_discrete_sequence=["#1565c0","#e53935"],
                    title="PRB Utilization: Source vs Neighbours",
                    labels={"cell_name":"Cell","value":"PRB %","variable":"Metric"},
                )
                fig_nbr.add_hline(y=prb_thr, line_dash="dash", line_color="red",
                                  annotation_text=f"Threshold ({prb_thr}%)")
                fig_nbr.update_layout(height=320, margin=dict(t=40,b=10))
                st.plotly_chart(fig_nbr, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — DETAILED PREDICTIONS & RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════
section_header("SECTION 3 — DETAILED PREDICTIONS & RECOMMENDATIONS")

tab_a, tab_b, tab_c, tab_d = st.tabs([
    "📊 All Predictions", "🔴 Cluster Congestion", "🟠 Load Balancing", "🔁 Patterns"
])

# ── Tab A: All predictions table ─────────────────────
with tab_a:
    disp = pred_classified[["timestamp","cell_name","gnb_name",
                             "congestion_probability","congestion_flag",
                             "scenario_type","neighbor_congestion_ratio"]].copy()
    disp["congestion_probability"] = (disp["congestion_probability"]*100).round(1)
    disp = disp.rename(columns={
        "congestion_probability":"Congestion Prob (%)",
        "congestion_flag":"Congested",
        "scenario_type":"Scenario",
        "neighbor_congestion_ratio":"Nbr Cong Ratio",
    })
    st.dataframe(
        disp.sort_values("Congestion Prob (%)", ascending=False).head(200),
        use_container_width=True, height=400,
    )
    csv = pred_classified.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download All Predictions (CSV)", csv, "predictions.csv", "text/csv")

# ── Tab B: Cluster Congestion ─────────────────────────
with tab_b:
    if cluster_df.empty:
        st.success("✅ No cluster-wide congestion predicted.")
    else:
        st.markdown(f"**{n_cluster_cells} cells** in cluster congestion.")
        cluster_summary = (
            cluster_df.groupby(["cell_name","gnb_name"])
            .agg(
                max_prob=("congestion_probability","max"),
                avg_prob=("congestion_probability","mean"),
                congested_hours=("congestion_flag","sum"),
            ).reset_index().sort_values("max_prob", ascending=False).round(3)
        )
        st.dataframe(cluster_summary, use_container_width=True)

        # Show one detailed recommendation
        ex_cell = cluster_summary.iloc[0]["cell_name"]
        st.markdown(f"#### Recommendations for **{ex_cell}**")
        actions = RECOMMENDATIONS["CLUSTER_CONGESTION"]["actions"]
        for act in actions:
            badge_color = {"HIGH":"#e53935","MEDIUM":"#fb8c00","LOW":"#43a047"}.get(act["priority"],"#888")
            st.markdown(
                f'<div style="border-left:5px solid {badge_color};'
                f'background:#fafafa;padding:.6rem 1rem;border-radius:6px;margin:.3rem 0;">'
                f'<b>[{act["priority"]}] {act["action"]}</b><br>'
                f'<small>{act["detail"]}</small><br>'
                f'📊 <i>{act["kpi_impact"]}</i>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if tickets:
            st.markdown("#### 🎫 Capacity Tickets")
            for t in tickets:
                st.markdown(
                    f'<div class="ticket-card">'
                    f'<span class="ticket-id">{t["ticket_id"]}</span> — '
                    f'Severity: <b>{t["severity"]}</b> | Status: <b>{t["status"]}</b><br>'
                    f'Cluster: {t["impacted_cluster"]} | Cells: {", ".join(t["impacted_cells"][:4])}<br>'
                    f'<small>{t["reason"]}</small><br>'
                    f'<small><b>Actions:</b> {t["recommended_action"].replace(chr(10)," | ")}</small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

# ── Tab C: Load Balancing ─────────────────────────────
with tab_c:
    if lb_df.empty:
        st.info("No load balancing opportunities predicted in this window.")
    else:
        st.markdown(f"**{n_lb_cells} cells** with load balancing opportunity.")
        lb_summary = (
            lb_df.groupby(["cell_name","gnb_name"])
            .agg(
                max_prob=("congestion_probability","max"),
                congested_hours=("congestion_flag","sum"),
            ).reset_index().sort_values("max_prob", ascending=False).round(3)
        )

        # Show healthy neighbours
        def get_healthy_nbrs(cell):
            rows = lb_df[lb_df["cell_name"]==cell]
            if rows.empty or "healthy_neighbors" not in rows.columns:
                return "—"
            all_nbrs = [n for ns in rows["healthy_neighbors"] for n in (ns if isinstance(ns,list) else [])]
            unique_nbrs = list(dict.fromkeys(all_nbrs))
            return ", ".join(unique_nbrs[:3]) if unique_nbrs else "—"

        lb_summary["Healthy Neighbours"] = lb_summary["cell_name"].apply(get_healthy_nbrs)
        st.dataframe(lb_summary, use_container_width=True)

        st.markdown("#### MLB / CIO Recommendations")
        actions_lb = RECOMMENDATIONS["LOAD_BALANCING_OPPORTUNITY"]["actions"]
        for act in actions_lb:
            badge_color = {"HIGH":"#e53935","MEDIUM":"#fb8c00","LOW":"#43a047"}.get(act["priority"],"#888")
            st.markdown(
                f'<div style="border-left:5px solid {badge_color};'
                f'background:#fafafa;padding:.6rem 1rem;border-radius:6px;margin:.3rem 0;">'
                f'<b>[{act["priority"]}] {act["action"]}</b><br>'
                f'<small>{act["detail"]}</small><br>'
                f'📊 <i>{act["kpi_impact"]}</i>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ── Tab D: Recurring Patterns ─────────────────────────
with tab_d:
    if patterns.empty:
        st.info("No strong recurring congestion patterns detected (threshold: ≥55% of same slots).")
    else:
        st.markdown(f"**{len(patterns)} recurring congestion windows** detected in historical data.")
        for _, row in patterns.head(15).iterrows():
            rate_color = "#e53935" if row["congestion_rate"] >= 0.80 else "#fb8c00"
            st.markdown(
                f'<div class="insight-box">'
                f'<b style="color:{rate_color};">▶ {row["insight_text"]}</b>'
                f'</div>',
                unsafe_allow_html=True,
            )
        fig_pat = px.bar(
            patterns.head(20),
            x="cell_name", y="congestion_rate",
            color="day_name",
            title="Top Recurring Congestion Windows",
            labels={"congestion_rate":"Congestion Rate","cell_name":"Cell","day_name":"Day"},
        )
        fig_pat.update_layout(height=360, yaxis_tickformat=".0%")
        st.plotly_chart(fig_pat, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# SECTION 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════
section_header("SECTION 4 — MODEL PERFORMANCE")
with st.expander("View Model Metrics & Feature Importance"):
    m1, m2, m3, m4 = st.columns(4)
    with m1: metric_card(f"{metrics['accuracy']:.1%}", f"{model_name} Accuracy")
    with m2: metric_card(f"{metrics['precision_congested']:.1%}", "Precision (Congested)")
    with m3: metric_card(f"{metrics['recall_congested']:.1%}", "Recall (Congested)")
    with m4: metric_card(f"{metrics['auc_roc']:.3f}" if metrics['auc_roc'] else "N/A", "AUC-ROC")

    fig_fi = px.bar(
        feat_imp.head(15), x="importance", y="feature",
        orientation="h", title="Top 15 Feature Importances",
        color="importance", color_continuous_scale="Blues",
    )
    fig_fi.update_layout(height=420, yaxis={"categoryorder":"total ascending"},
                         margin=dict(t=40,b=10,r=10))
    st.plotly_chart(fig_fi, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#888;font-size:.82rem;">'
    '5G Congestion Predictor v3.0 · XGBoost ML · SON-ready · MW-hour aware'
    '</div>',
    unsafe_allow_html=True,
)
