"""
5G Cell Congestion Prediction — ML Pipeline  v3.0
===================================================
Module  : ml_pipeline.py

Changes in v3.0:
  • Balanced synthetic data support
  • Dual-threshold congestion (PRB OR RRC)
  • MW-hour exclusion
  • XGBoost preferred, RandomForest fallback
  • Richer lag/rolling features
  • Recurring pattern detection
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    _USE_XGB = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    _USE_XGB = False

MW_HOURS   = {1, 2, 3, 4, 5}
PEAK_HOURS = {8, 9, 10, 11, 12, 17, 18, 19, 20, 21}
DOW_LABELS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]


# ═══════════════════════════════════════════════════════
# DATA VALIDATION
# ═══════════════════════════════════════════════════════

def validate_kpi_data(df: pd.DataFrame) -> dict:
    errors, warnings_list = [], []
    required = {"timestamp","cell_name","gnb_name","prb_utilization","avg_rrc_users","throughput_mbps"}
    missing  = required - set(df.columns)
    if missing:
        return {"ok": False, "errors": [f"Missing columns: {missing}"], "warnings": [], "stats": {}}

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        errors.append("Some timestamps could not be parsed.")

    days = (df["timestamp"].max() - df["timestamp"].min()).days
    if days < 55:
        warnings_list.append(f"Data spans {days} days. Recommend ≥60 days.")

    n_cells = df["cell_name"].nunique()
    if n_cells < 20:
        warnings_list.append(f"Only {n_cells} cells. Recommend ≥20.")

    null_counts = df[list(required)].isnull().sum()
    nulls = null_counts[null_counts > 0]
    if not nulls.empty:
        warnings_list.append("Null values: " + ", ".join(f"{c}={v}" for c, v in nulls.items()))

    stats = {
        "total_records":  len(df),
        "unique_cells":   n_cells,
        "unique_gnbs":    df["gnb_name"].nunique(),
        "date_range_days": days,
        "date_min":       str(df["timestamp"].min().date()),
        "date_max":       str(df["timestamp"].max().date()),
        "avg_prb":        round(df["prb_utilization"].mean(), 2),
        "max_prb":        round(df["prb_utilization"].max(), 2),
    }
    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings_list, "stats": stats}


# ═══════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame, exclude_mw_hours: bool = False):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if exclude_mw_hours:
        df = df[~df["timestamp"].dt.hour.isin(MW_HOURS)].copy()

    df["hour"]           = df["timestamp"].dt.hour
    df["day_of_week"]    = df["timestamp"].dt.dayofweek
    df["month"]          = df["timestamp"].dt.month
    df["is_weekend"]     = (df["day_of_week"] >= 5).astype(int)
    df["peak_hour_flag"] = df["hour"].isin(PEAK_HOURS).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df = df.sort_values(["cell_name","timestamp"]).reset_index(drop=True)

    df["prev_hour_prb"] = (
        df.groupby("cell_name")["prb_utilization"].shift(1).fillna(df["prb_utilization"])
    )
    df["prev_day_same_hour_prb"] = (
        df.groupby("cell_name")["prb_utilization"].shift(24).fillna(df["prb_utilization"])
    )

    for w in [3, 6, 12]:
        df[f"prb_roll_mean_{w}h"] = (
            df.groupby("cell_name")["prb_utilization"]
            .transform(lambda x: x.rolling(w, min_periods=1).mean())
        )
        df[f"prb_roll_max_{w}h"] = (
            df.groupby("cell_name")["prb_utilization"]
            .transform(lambda x: x.rolling(w, min_periods=1).max())
        )

    for w in [3, 6]:
        df[f"rrc_roll_mean_{w}h"] = (
            df.groupby("cell_name")["avg_rrc_users"]
            .transform(lambda x: x.rolling(w, min_periods=1).mean())
        )

    le_cell = LabelEncoder()
    le_gnb  = LabelEncoder()
    df["cell_name_enc"] = le_cell.fit_transform(df["cell_name"].astype(str))
    df["gnb_name_enc"]  = le_gnb.fit_transform(df["gnb_name"].astype(str))

    return df, le_cell, le_gnb


# ═══════════════════════════════════════════════════════
# TARGET DEFINITION
# ═══════════════════════════════════════════════════════

def define_target(df: pd.DataFrame, prb_threshold: float = 85.0, rrc_threshold: int = 300) -> pd.DataFrame:
    df = df.copy()
    df["congested"] = (
        (df["prb_utilization"] > prb_threshold) |
        (df["avg_rrc_users"]   > rrc_threshold)
    ).astype(int)
    return df


# ═══════════════════════════════════════════════════════
# MODEL TRAINING
# ═══════════════════════════════════════════════════════

FEATURE_COLS = [
    "hour","day_of_week","month","is_weekend","peak_hour_flag",
    "hour_sin","hour_cos","dow_sin","dow_cos",
    "avg_rrc_users","throughput_mbps",
    "prev_hour_prb","prev_day_same_hour_prb",
    "prb_roll_mean_3h","prb_roll_max_3h",
    "prb_roll_mean_6h","prb_roll_max_6h",
    "prb_roll_mean_12h","prb_roll_max_12h",
    "rrc_roll_mean_3h","rrc_roll_mean_6h",
    "cell_name_enc","gnb_name_enc",
]

def get_feature_cols():
    return FEATURE_COLS.copy()


def train_model(df: pd.DataFrame):
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available]
    y = df["congested"]

    if y.nunique() < 2:
        raise ValueError("Target has only one class. Check thresholds.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if _USE_XGB:
        scale_pos = (y == 0).sum() / max((y == 1).sum(), 1)
        model = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0,
        )
        model_name = "XGBoost"
    else:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        model_name = "RandomForest"

    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    report  = classification_report(y_test, y_pred, output_dict=True)
    auc     = roc_auc_score(y_test, y_proba) if y.nunique() > 1 else None

    metrics = {
        "model_name":          model_name,
        "accuracy":            report["accuracy"],
        "precision_congested": report.get("1", {}).get("precision", 0),
        "recall_congested":    report.get("1", {}).get("recall", 0),
        "f1_congested":        report.get("1", {}).get("f1-score", 0),
        "auc_roc":             auc,
        "train_samples":       len(X_train),
        "test_samples":        len(X_test),
    }

    feat_imp = pd.DataFrame({
        "feature":    available,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return model, metrics, feat_imp, available, model_name


# ═══════════════════════════════════════════════════════
# PREDICTION PIPELINE
# ═══════════════════════════════════════════════════════

def generate_future_timestamps(hours: int = 168) -> pd.DataFrame:
    """Default to 1 week (168h) for executive summary."""
    now   = pd.Timestamp.now().ceil("h")
    times = pd.date_range(start=now, periods=hours, freq="h")
    return pd.DataFrame({"timestamp": times})


def build_prediction_input(
    future_df, cells, kpi_df, le_cell, le_gnb,
    exclude_mw_hours=False, prb_threshold=85.0, rrc_threshold=300,
):
    future_df = future_df.copy()
    future_df["_key"] = 1
    cells_df = pd.DataFrame({"cell_name": cells, "_key": 1})
    pred_df  = future_df.merge(cells_df, on="_key").drop("_key", axis=1)

    if exclude_mw_hours:
        pred_df = pred_df[~pred_df["timestamp"].dt.hour.isin(MW_HOURS)].copy()

    cell_gnb_map = kpi_df.drop_duplicates("cell_name").set_index("cell_name")["gnb_name"]
    pred_df["gnb_name"] = pred_df["cell_name"].map(cell_gnb_map).fillna("Unknown")

    # Hour-of-day-aware KPI imputation using 75th percentile for PRB
    # (captures peak congestion patterns; mean is too smooth)
    kpi_copy = kpi_df.copy()
    kpi_copy["_hour"] = pd.to_datetime(kpi_copy["timestamp"]).dt.hour

    for col in ["avg_rrc_users","throughput_mbps","prb_utilization"]:
        agg_func = "quantile" if col in ("prb_utilization","avg_rrc_users") else "mean"
        if agg_func == "quantile":
            hour_stat = (
                kpi_copy.groupby(["cell_name","_hour"])[col]
                .quantile(0.75).reset_index()
                .rename(columns={"_hour":"_merge_hour", col: f"_hval_{col}"})
            )
        else:
            hour_stat = (
                kpi_copy.groupby(["cell_name","_hour"])[col]
                .mean().reset_index()
                .rename(columns={"_hour":"_merge_hour", col: f"_hval_{col}"})
            )
        pred_df["_merge_hour"] = pred_df["timestamp"].dt.hour
        pred_df = pred_df.merge(hour_stat, on=["cell_name","_merge_hour"], how="left")
        overall_avg = kpi_df.groupby("cell_name")[col].mean()
        mask = pred_df[f"_hval_{col}"].isna()
        pred_df.loc[mask, f"_hval_{col}"] = pred_df.loc[mask,"cell_name"].map(overall_avg)
        pred_df[col] = pred_df[f"_hval_{col}"].fillna(kpi_df[col].mean())
        pred_df.drop(columns=[f"_hval_{col}","_merge_hour"], inplace=True)

    pred_df["hour"]           = pred_df["timestamp"].dt.hour
    pred_df["day_of_week"]    = pred_df["timestamp"].dt.dayofweek
    pred_df["month"]          = pred_df["timestamp"].dt.month
    pred_df["is_weekend"]     = (pred_df["day_of_week"] >= 5).astype(int)
    pred_df["peak_hour_flag"] = pred_df["hour"].isin(PEAK_HOURS).astype(int)
    pred_df["hour_sin"] = np.sin(2 * np.pi * pred_df["hour"] / 24)
    pred_df["hour_cos"] = np.cos(2 * np.pi * pred_df["hour"] / 24)
    pred_df["dow_sin"]  = np.sin(2 * np.pi * pred_df["day_of_week"] / 7)
    pred_df["dow_cos"]  = np.cos(2 * np.pi * pred_df["day_of_week"] / 7)

    # Hour-aware lag feature imputation using 75th percentile per (cell, hour)
    kpi_copy2 = kpi_df.copy()
    kpi_copy2["_hour"] = pd.to_datetime(kpi_copy2["timestamp"]).dt.hour

    # Per-cell per-hour 75th percentile PRB (used for prev_hour and rolling features)
    hour_p75_prb = (
        kpi_copy2.groupby(["cell_name","_hour"])["prb_utilization"]
        .quantile(0.75).reset_index()
        .rename(columns={"_hour":"_h", "prb_utilization":"_p75_prb"})
    )
    hour_p75_rrc = (
        kpi_copy2.groupby(["cell_name","_hour"])["avg_rrc_users"]
        .quantile(0.75).reset_index()
        .rename(columns={"_hour":"_h", "avg_rrc_users":"_p75_rrc"})
    )

    # For prev_hour_prb: use p75 of (hour-1)
    pred_df["_prev_h"] = (pred_df["timestamp"].dt.hour - 1) % 24
    pred_df = pred_df.merge(
        hour_p75_prb.rename(columns={"_h":"_prev_h","_p75_prb":"_prev_prb"}),
        on=["cell_name","_prev_h"], how="left"
    )
    hist_p75_overall = kpi_df.groupby("cell_name")["prb_utilization"].quantile(0.75)
    pred_df["prev_hour_prb"] = pred_df["_prev_prb"].fillna(
        pred_df["cell_name"].map(hist_p75_overall)
    )
    pred_df.drop(columns=["_prev_h","_prev_prb"], inplace=True)

    # For rolling features: use per-hour p75 as best available estimate
    pred_df["_cur_h"] = pred_df["timestamp"].dt.hour
    pred_df = pred_df.merge(
        hour_p75_prb.rename(columns={"_h":"_cur_h","_p75_prb":"_cur_prb"}),
        on=["cell_name","_cur_h"], how="left"
    )
    pred_df = pred_df.merge(
        hour_p75_rrc.rename(columns={"_h":"_cur_h","_p75_rrc":"_cur_rrc"}),
        on=["cell_name","_cur_h"], how="left"
    )

    hist_p75_rrc = kpi_df.groupby("cell_name")["avg_rrc_users"].quantile(0.75)
    hist_max_prb = kpi_df.groupby("cell_name")["prb_utilization"].max()

    cur_prb = pred_df["_cur_prb"].fillna(pred_df["cell_name"].map(hist_p75_overall))
    cur_rrc = pred_df["_cur_rrc"].fillna(pred_df["cell_name"].map(hist_p75_rrc))

    pred_df["prev_day_same_hour_prb"] = cur_prb

    for w in [3, 6, 12]:
        pred_df[f"prb_roll_mean_{w}h"] = cur_prb
        pred_df[f"prb_roll_max_{w}h"]  = pred_df["cell_name"].map(hist_max_prb) * 0.9

    for w in [3, 6]:
        pred_df[f"rrc_roll_mean_{w}h"] = cur_rrc

    pred_df.drop(columns=["_cur_h","_cur_prb","_cur_rrc"], inplace=True, errors="ignore")

    known_cells = list(le_cell.classes_)
    pred_df["cell_name_enc"] = le_cell.transform(
        pred_df["cell_name"].apply(lambda x: x if x in known_cells else known_cells[0])
    )
    known_gnbs = list(le_gnb.classes_)
    pred_df["gnb_name_enc"] = le_gnb.transform(
        pred_df["gnb_name"].apply(lambda x: x if x in known_gnbs else known_gnbs[0])
    )
    return pred_df


def predict_congestion(model, pred_df, feature_cols):
    available = [c for c in feature_cols if c in pred_df.columns]
    proba = model.predict_proba(pred_df[available])[:, 1]
    pred_df = pred_df.copy()
    pred_df["congestion_probability"] = np.round(proba, 4)
    pred_df["congestion_flag"]        = (proba > 0.5).astype(int)
    return pred_df


# ═══════════════════════════════════════════════════════
# RECURRING PATTERN DETECTION
# ═══════════════════════════════════════════════════════

def detect_recurring_patterns(df: pd.DataFrame, prb_threshold: float = 85.0) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"]   = pd.to_datetime(df["timestamp"])
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["congested"]   = (df["prb_utilization"] > prb_threshold).astype(int)

    grp = (
        df.groupby(["cell_name","day_of_week","hour"])["congested"]
        .mean().reset_index()
        .rename(columns={"congested":"congestion_rate"})
    )
    grp = grp[grp["congestion_rate"] >= 0.55].copy()
    grp["day_name"] = grp["day_of_week"].map(lambda d: DOW_LABELS[d])
    grp["insight_text"] = grp.apply(
        lambda r: (
            f"{r['cell_name']} congested ≥{r['congestion_rate']:.0%} on "
            f"{r['day_name']}s at {r['hour']:02d}:00"
        ), axis=1,
    )
    return grp[["cell_name","day_name","hour","congestion_rate","insight_text"]]\
        .sort_values("congestion_rate", ascending=False)
