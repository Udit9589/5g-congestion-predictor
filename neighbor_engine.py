"""
5G Cell Congestion — Neighbor Intelligence Engine  v3.0
========================================================
Module  : neighbor_engine.py

Changes in v3.0:
  • CLUSTER_THRESHOLD raised to 0.70 (70% of neighbors congested)
  • LOAD_BALANCING triggered if ≥2 neighbors are healthy
  • Healthy neighbor = PRB below threshold AND RRC below threshold
  • Ticket auto-generation for CLUSTER_CONGESTION
  • Richer recommendations for both scenarios
"""

import uuid, datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

CLUSTER_THRESHOLD = 0.70   # ≥70% of HO-weighted neighbours congested → CLUSTER
TOP_N_NEIGHBORS   = 4


# ═══════════════════════════════════════════════════════
# NEIGHBOR MAP
# ═══════════════════════════════════════════════════════

def build_neighbor_map(neighbors_df: pd.DataFrame) -> Dict[str, List[str]]:
    if not {"cell_name","neighbor_cell","ho_attempts"}.issubset(neighbors_df.columns):
        return (
            neighbors_df.groupby("cell_name")["neighbor_cell"]
            .apply(lambda x: list(x.head(TOP_N_NEIGHBORS))).to_dict()
        )
    top4 = (
        neighbors_df
        .sort_values(["cell_name","ho_attempts"], ascending=[True,False])
        .groupby("cell_name").head(TOP_N_NEIGHBORS)
    )
    return top4.groupby("cell_name")["neighbor_cell"].apply(list).to_dict()


def build_ho_weight_map(neighbors_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    if "ho_attempts" not in neighbors_df.columns:
        return {}
    top4 = (
        neighbors_df
        .sort_values(["cell_name","ho_attempts"], ascending=[True,False])
        .groupby("cell_name").head(TOP_N_NEIGHBORS)
    )
    ho_map = {}
    for cell, grp in top4.groupby("cell_name"):
        total = grp["ho_attempts"].sum()
        if total == 0:
            weights = {r["neighbor_cell"]: 1/len(grp) for _,r in grp.iterrows()}
        else:
            weights = {r["neighbor_cell"]: r["ho_attempts"]/total for _,r in grp.iterrows()}
        ho_map[cell] = weights
    return ho_map


# ═══════════════════════════════════════════════════════
# SCENARIO CLASSIFICATION  (CORE FIX)
# ═══════════════════════════════════════════════════════

def classify_scenario(
    predictions_df: pd.DataFrame,
    neighbor_map: Dict[str, List[str]],
    ho_weight_map: Optional[Dict[str, Dict[str, float]]] = None,
    kpi_df: Optional[pd.DataFrame] = None,
    prb_threshold: float = 85.0,
    rrc_threshold: int   = 300,
) -> pd.DataFrame:
    """
    Classify each (cell, timestamp) into:
      CLUSTER_CONGESTION          – source congested AND ≥70% neighbours congested
      LOAD_BALANCING_OPPORTUNITY  – source congested AND ≥2 neighbours healthy
                                    (healthy = PRB < threshold AND RRC < threshold)
      NORMAL                      – source not congested
      CONGESTED_NO_NEIGHBOR_DATA  – source congested, no neighbour info
    """
    df = predictions_df.copy()

    # Build lookup: (cell, timestamp) → congestion_flag
    flag_lookup = df.set_index(["cell_name","timestamp"])["congestion_flag"].to_dict()

    # Build KPI lookup for neighbor health checks
    kpi_lookup: Dict = {}
    if kpi_df is not None and not kpi_df.empty:
        kpi_lookup = (
            kpi_df.groupby("cell_name")
            .agg(avg_prb=("prb_utilization","mean"), avg_rrc=("avg_rrc_users","mean"))
            .to_dict("index")
        )

    scenarios, nbr_ratios, nbr_cong_lists, healthy_nbrs_list = [], [], [], []

    for _, row in df.iterrows():
        cell    = row["cell_name"]
        ts      = row["timestamp"]
        is_cong = row["congestion_flag"] == 1
        nbrs    = neighbor_map.get(cell, [])

        if not is_cong:
            scenarios.append("NORMAL")
            nbr_ratios.append(None)
            nbr_cong_lists.append([])
            healthy_nbrs_list.append([])
            continue

        if not nbrs:
            scenarios.append("CONGESTED_NO_NEIGHBOR_DATA")
            nbr_ratios.append(None)
            nbr_cong_lists.append([])
            healthy_nbrs_list.append([])
            continue

        weights   = (ho_weight_map or {}).get(cell, {})
        total_w   = 0.0
        cong_w    = 0.0
        cong_nbrs = []
        healthy_nbrs = []

        for n in nbrs:
            flag = flag_lookup.get((n, ts), 0)
            w    = weights.get(n, 1.0 / len(nbrs))
            total_w += w

            if flag:
                cong_w += w
                cong_nbrs.append(n)
            else:
                # Double-check against historical KPI: confirm neighbor is genuinely healthy
                nbr_kpi = kpi_lookup.get(n, {})
                nbr_prb = nbr_kpi.get("avg_prb", 0)
                nbr_rrc = nbr_kpi.get("avg_rrc", 0)
                if nbr_prb < prb_threshold and nbr_rrc < rrc_threshold:
                    healthy_nbrs.append(n)

        ratio = round(cong_w / total_w, 3) if total_w > 0 else 0.0
        nbr_ratios.append(ratio)
        nbr_cong_lists.append(cong_nbrs)
        healthy_nbrs_list.append(healthy_nbrs)

        # ── Decision logic ──────────────────────────────────────────────
        # PRIORITY: Load Balancing if ≥2 healthy neighbors exist
        if len(healthy_nbrs) >= 2:
            scenarios.append("LOAD_BALANCING_OPPORTUNITY")
        elif ratio >= CLUSTER_THRESHOLD:
            scenarios.append("CLUSTER_CONGESTION")
        elif len(cong_nbrs) == 0:
            # Source congested, no neighbor data matches → treat as LB
            scenarios.append("LOAD_BALANCING_OPPORTUNITY")
        else:
            # Mixed: some congested neighbors but not 70%
            scenarios.append("LOAD_BALANCING_OPPORTUNITY")

    df["scenario_type"]             = scenarios
    df["neighbor_congestion_ratio"] = nbr_ratios
    df["congested_neighbors"]       = nbr_cong_lists
    df["healthy_neighbors"]         = healthy_nbrs_list

    return df


# ═══════════════════════════════════════════════════════
# RECOMMENDATIONS
# ═══════════════════════════════════════════════════════

RECOMMENDATIONS = {
    "CLUSTER_CONGESTION": {
        "summary": "Cluster-wide congestion detected. Capacity action required urgently.",
        "actions": [
            {"priority":"HIGH",   "action":"Capacity Expansion",
             "detail":"Deploy additional RRU units or upgrade to 64T64R MIMO.",
             "kpi_impact":"PRB ↓ 20–30%"},
            {"priority":"HIGH",   "action":"Spectrum Addition",
             "detail":"Activate mid-band (n78 3.5GHz) or mmWave for hotspots.",
             "kpi_impact":"Throughput ↑ 2–5×"},
            {"priority":"MEDIUM", "action":"Carrier Aggregation (CA)",
             "detail":"Enable additional NR carrier on available bands.",
             "kpi_impact":"User throughput ↑ 30–50%"},
            {"priority":"MEDIUM", "action":"NSA → SA Migration",
             "detail":"Migrate to 5G SA to improve spectral efficiency.",
             "kpi_impact":"Latency ↓, efficiency ↑"},
        ],
    },
    "LOAD_BALANCING_OPPORTUNITY": {
        "summary": "Source cell congested but neighbours have available capacity. Redistribute traffic.",
        "actions": [
            {"priority":"HIGH",   "action":"CIO Tuning",
             "detail":"Reduce CIO on congested cell; increase on lightly-loaded neighbours.",
             "kpi_impact":"PRB ↓ 15–25%"},
            {"priority":"HIGH",   "action":"Mobility Load Balancing (MLB)",
             "detail":"Activate SON-MLB to auto-redistribute load within 15 mins.",
             "kpi_impact":"Automated relief within 15 mins"},
            {"priority":"MEDIUM", "action":"ANR Optimisation",
             "detail":"Add missing neighbour relations to improve HO coverage.",
             "kpi_impact":"HO success ↑, drop rate ↓"},
            {"priority":"LOW",    "action":"TTT Reduction",
             "detail":"Reduce Time-to-Trigger on congested cell to speed up HO.",
             "kpi_impact":"Faster load redistribution"},
        ],
    },
    "NORMAL": {
        "summary": "Cell operating normally. Continue monitoring.",
        "actions": [
            {"priority":"INFO", "action":"Preventive Monitoring",
             "detail":"Flag if PRB trend exceeds 70% sustained >2 hours.",
             "kpi_impact":"Proactive capacity management"},
        ],
    },
    "CONGESTED_NO_NEIGHBOR_DATA": {
        "summary": "Cell congested. Upload neighbour data for intelligent classification.",
        "actions": [
            {"priority":"MEDIUM", "action":"Upload Neighbour CSV",
             "detail":"Provide neighbors.csv to enable MLB / CIO recommendations.",
             "kpi_impact":"Enables targeted optimisation"},
            {"priority":"HIGH",   "action":"General Capacity Review",
             "detail":"Review PRB trend and consider expansion.",
             "kpi_impact":"Risk mitigation"},
        ],
    },
}


def generate_recommendations(row: pd.Series) -> List[Dict]:
    return RECOMMENDATIONS.get(row.get("scenario_type","NORMAL"), RECOMMENDATIONS["NORMAL"])["actions"]


def attach_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["recommendations"]       = df.apply(generate_recommendations, axis=1)
    df["recommendation_summary"] = df["scenario_type"].map(
        {k: v["summary"] for k, v in RECOMMENDATIONS.items()}
    )
    return df


def get_scenario_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("scenario_type")
        .agg(
            cell_count=("cell_name","nunique"),
            record_count=("cell_name","count"),
            avg_congestion_prob=("congestion_probability","mean"),
        )
        .reset_index()
        .sort_values("record_count", ascending=False)
    )


# ═══════════════════════════════════════════════════════
# TICKET GENERATION
# ═══════════════════════════════════════════════════════

def generate_capacity_ticket(cell, gnb, congested_neighbors, congestion_probability, peak_hour=None):
    ticket_id = "TKT-" + uuid.uuid4().hex[:8].upper()
    all_cells = [cell] + (congested_neighbors or [])
    reason = (
        f"Cluster congestion on {cell} (gNB: {gnb}). "
        f"Congestion prob: {congestion_probability:.0%}. "
        f"Impacted: {', '.join(all_cells)}."
    )
    if peak_hour is not None:
        reason += f" Recurring busy window at {peak_hour:02d}:00."
    return {
        "ticket_id":          ticket_id,
        "created_at":         datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "severity":           "CRITICAL" if congestion_probability >= 0.85 else "HIGH",
        "impacted_cluster":   gnb,
        "impacted_cells":     all_cells,
        "reason":             reason,
        "recommended_action": (
            "1. Capacity Expansion (TRX/RRU upgrade)\n"
            "2. Spectrum Addition (n78 / mmWave)\n"
            "3. Carrier Aggregation"
        ),
        "status": "OPEN",
    }


def auto_generate_tickets(pred_df: pd.DataFrame) -> List[Dict]:
    tickets = []
    cluster_rows = pred_df[pred_df["scenario_type"] == "CLUSTER_CONGESTION"].copy()
    if cluster_rows.empty:
        return tickets
    best = (
        cluster_rows
        .sort_values("congestion_probability", ascending=False)
        .drop_duplicates("cell_name")
    )
    for _, row in best.iterrows():
        tickets.append(generate_capacity_ticket(
            cell=row["cell_name"],
            gnb=row.get("gnb_name","Unknown"),
            congested_neighbors=row.get("congested_neighbors",[]),
            congestion_probability=row["congestion_probability"],
            peak_hour=int(row["hour"]) if "hour" in row else None,
        ))
    return tickets


# ═══════════════════════════════════════════════════════
# NEIGHBOUR UTILISATION COMPARISON
# ═══════════════════════════════════════════════════════

def get_neighbor_utilization(cell, kpi_df, neighbor_map):
    cells = [cell] + neighbor_map.get(cell, [])
    sub   = kpi_df[kpi_df["cell_name"].isin(cells)]
    if sub.empty:
        return pd.DataFrame()
    summary = (
        sub.groupby("cell_name")
        .agg(avg_prb=("prb_utilization","mean"),
             max_prb=("prb_utilization","max"),
             avg_rrc_users=("avg_rrc_users","mean"))
        .reset_index().round(2)
    )
    summary["is_source"] = summary["cell_name"] == cell
    return summary.sort_values("is_source", ascending=False).reset_index(drop=True)
