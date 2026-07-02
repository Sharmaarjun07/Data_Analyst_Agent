# =============================================================
#  services/eda_service.py
#  Phase 5 — EDA Engine
#
#  One public function:
#    run_eda(df, metadata) → full EDA result dict
#
#  Computes everything numerically first.
#  The LLM (Phase 8) will later read this dict and explain it.
#  Charts are built in upload.py using Plotly — not here.
# =============================================================

import pandas as pd
import numpy as np
from datetime import datetime


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def _safe_float(v):
    """Convert numpy scalar to plain Python float, return None if not finite."""
    try:
        f = float(v)
        return round(f, 6) if np.isfinite(f) else None
    except Exception:
        return None


def _descriptive_stats(series: pd.Series) -> dict:
    """Full descriptive stats for one numeric column."""
    clean = series.dropna()
    if clean.empty:
        return {}
    q1, q3  = clean.quantile(0.25), clean.quantile(0.75)
    iqr     = q3 - q1
    return {
        "count":    int(clean.count()),
        "mean":     _safe_float(clean.mean()),
        "median":   _safe_float(clean.median()),
        "mode":     _safe_float(clean.mode().iloc[0]) if not clean.mode().empty else None,
        "std":      _safe_float(clean.std()),
        "variance": _safe_float(clean.var()),
        "min":      _safe_float(clean.min()),
        "max":      _safe_float(clean.max()),
        "range":    _safe_float(clean.max() - clean.min()),
        "q1":       _safe_float(q1),
        "q3":       _safe_float(q3),
        "iqr":      _safe_float(iqr),
        "skewness": _safe_float(clean.skew()),
        "kurtosis": _safe_float(clean.kurt()),
        "cv":       _safe_float(clean.std() / clean.mean() * 100) if clean.mean() != 0 else None,
    }


def _skew_label(skew: float) -> str:
    if skew is None:
        return "unknown"
    if abs(skew) < 0.5:
        return "symmetric"
    if abs(skew) < 1.0:
        return "moderately skewed"
    return "highly skewed"


def _kurt_label(kurt: float) -> str:
    if kurt is None:
        return "unknown"
    if kurt > 1:
        return "heavy-tailed (leptokurtic)"
    if kurt < -1:
        return "light-tailed (platykurtic)"
    return "normal-tailed (mesokurtic)"


# ─────────────────────────────────────────────────────────────
#  MAIN PUBLIC FUNCTION
# ─────────────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame, metadata: dict) -> dict:
    """
    Run full exploratory data analysis on the (cleaned) DataFrame.

    Returns:
    {
      "computed_at": str,
      "numeric_stats":   { col: { count, mean, median, std, ... } },
      "correlation":     { "pearson":  [[...]], "spearman": [[...]], "columns": [...] },
      "distributions":   { col: { histogram_bins, histogram_counts, kde_x, kde_y } },
      "categorical_eda": { col: { value_counts: [{label,count,pct}], n_unique } },
      "time_series":     { col: { monthly, weekly, moving_avg_30 } }  — if date exists
      "summary_insights":[ plain-English strings the LLM can also enrich ]
    }
    """
    cols_meta   = {c["name"]: c for c in metadata.get("columns", [])}
    num_cols    = df.select_dtypes(include="number").columns.tolist()
    cat_cols    = [c for c in df.columns
                   if cols_meta.get(c, {}).get("dtype_category") == "categorical"]
    date_cols   = [c for c in df.columns
                   if cols_meta.get(c, {}).get("dtype_category") == "datetime"
                   or str(df[c].dtype).startswith("datetime")]

    result = {"computed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    # ── 1. Descriptive stats (numeric) ────────────────────────
    numeric_stats = {}
    for col in num_cols:
        stats = _descriptive_stats(df[col])
        if stats:
            stats["skew_label"] = _skew_label(stats.get("skewness"))
            stats["kurt_label"] = _kurt_label(stats.get("kurtosis"))
            numeric_stats[col]  = stats
    result["numeric_stats"] = numeric_stats

    # ── 2. Correlation matrices ────────────────────────────────
    corr_result = {"columns": [], "pearson": [], "spearman": []}
    if len(num_cols) >= 2:
        num_df   = df[num_cols].dropna()
        pearson  = num_df.corr(method="pearson")
        spearman = num_df.corr(method="spearman")
        corr_result["columns"] = num_cols
        corr_result["pearson"]  = [
            [_safe_float(v) for v in row] for row in pearson.values
        ]
        corr_result["spearman"] = [
            [_safe_float(v) for v in row] for row in spearman.values
        ]
        # Top correlated pairs (excluding self-correlations)
        pairs = []
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                v = pearson.iloc[i, j]
                if _safe_float(v) is not None:
                    pairs.append({
                        "col_a": num_cols[i],
                        "col_b": num_cols[j],
                        "pearson": round(float(v), 4),
                        "strength": (
                            "strong"   if abs(v) > 0.7 else
                            "moderate" if abs(v) > 0.4 else
                            "weak"
                        ),
                        "direction": "positive" if v > 0 else "negative",
                    })
        pairs.sort(key=lambda x: abs(x["pearson"]), reverse=True)
        corr_result["top_pairs"] = pairs[:10]
    result["correlation"] = corr_result

    # ── 3. Distribution data (for histograms) ─────────────────
    distributions = {}
    for col in num_cols:
        clean = df[col].dropna()
        if len(clean) < 2:
            continue
        n_bins  = min(50, max(10, int(np.sqrt(len(clean)))))
        counts, bin_edges = np.histogram(clean, bins=n_bins)
        bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(counts))]
        distributions[col] = {
            "bin_centers": [_safe_float(v) for v in bin_centers],
            "counts":      [int(v) for v in counts],
            "bin_edges":   [_safe_float(v) for v in bin_edges],
        }
    result["distributions"] = distributions

    # ── 4. Categorical EDA ────────────────────────────────────
    categorical_eda = {}
    for col in cat_cols:
        vc      = df[col].value_counts(dropna=False)
        n_total = len(df)
        top15   = vc.head(15)
        categorical_eda[col] = {
            "n_unique":    int(df[col].nunique()),
            "value_counts": [
                {
                    "label": str(k) if pd.notna(k) else "(missing)",
                    "count": int(v),
                    "pct":   round(v / n_total * 100, 2),
                }
                for k, v in top15.items()
            ],
        }
    result["categorical_eda"] = categorical_eda

    # ── 5. Time-series EDA (if date column exists) ────────────
    time_series = {}
    for col in date_cols:
        try:
            # Try to find a useful numeric column to aggregate
            ts_series = pd.to_datetime(df[col], errors="coerce")
            if ts_series.isna().mean() > 0.5:
                continue
            temp_df       = df.copy()
            temp_df["_ts"] = ts_series
            temp_df        = temp_df.dropna(subset=["_ts"])

            # Pick first numeric column as value
            val_col = num_cols[0] if num_cols else None
            if val_col is None:
                continue

            temp_df["_month"] = temp_df["_ts"].dt.to_period("M").astype(str)
            temp_df["_week"]  = temp_df["_ts"].dt.to_period("W").astype(str)

            monthly = temp_df.groupby("_month")[val_col].sum().reset_index()
            weekly  = temp_df.groupby("_week")[val_col].sum().reset_index()

            # 30-period moving average on monthly
            monthly["_ma"] = monthly[val_col].rolling(window=3, min_periods=1).mean()

            time_series[col] = {
                "value_col": val_col,
                "monthly": {
                    "periods": monthly["_month"].tolist(),
                    "values":  [_safe_float(v) for v in monthly[val_col]],
                    "moving_avg": [_safe_float(v) for v in monthly["_ma"]],
                },
                "weekly": {
                    "periods": weekly["_week"].tolist()[:52],   # cap at 52 weeks
                    "values":  [_safe_float(v) for v in weekly[val_col]][:52],
                },
            }
        except Exception:
            continue
    result["time_series"] = time_series

    # ── 6. Plain-English summary insights ─────────────────────
    insights = []

    # Missing summary
    miss_total = int(df.isnull().sum().sum())
    if miss_total == 0:
        insights.append("✅ No missing values — dataset is complete.")
    else:
        miss_pct = round(miss_total / df.size * 100, 1)
        insights.append(f"⚠️ {miss_total:,} missing cells ({miss_pct}% of data).")

    # Skewness insights
    for col, st_dict in numeric_stats.items():
        sk = st_dict.get("skewness")
        if sk and abs(sk) > 1:
            direction = "right" if sk > 0 else "left"
            insights.append(
                f"📐 '{col}' is highly skewed {direction} (skew={sk:.2f}) — "
                "consider log-transform before modeling."
            )

    # Strong correlations
    for pair in corr_result.get("top_pairs", [])[:3]:
        if abs(pair["pearson"]) > 0.7:
            insights.append(
                f"🔗 Strong {pair['direction']} correlation between "
                f"'{pair['col_a']}' and '{pair['col_b']}' "
                f"(r={pair['pearson']:.2f})."
            )

    # High-cardinality categories
    for col, ced in categorical_eda.items():
        if ced["n_unique"] > 50:
            insights.append(
                f"🏷️ '{col}' has {ced['n_unique']} unique categories — "
                "may need grouping before one-hot encoding."
            )

    # Dominant categories
    for col, ced in categorical_eda.items():
        vc = ced["value_counts"]
        if vc and vc[0]["pct"] > 70:
            insights.append(
                f"📊 '{col}' is dominated by '{vc[0]['label']}' "
                f"({vc[0]['pct']}% of rows) — check for class imbalance."
            )

    result["summary_insights"] = insights
    return result