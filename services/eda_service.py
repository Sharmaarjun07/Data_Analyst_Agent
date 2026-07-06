# =============================================================
#  services/eda_service.py
#  Phase 5 — EDA Engine
#
#  One public function:
#    run_eda(df, metadata) → full EDA result dict
#
#  Computes everything numerically first.
#  The LLM (Phase 8) will later read this dict and explain it.
#  Charts are built in the frontend (Plotly / Streamlit) — not here.
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


def _dtype_category(series: pd.Series, meta_entry: dict) -> str:
    """Best-effort dtype-category resolver, falling back to pandas dtype."""
    cat = meta_entry.get("dtype_category")
    if cat:
        return cat
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "categorical"


def _pick_value_column(df: pd.DataFrame, num_cols: list, metadata: dict):
    """
    Pick the column to aggregate for time-series analysis.
    Priority:
      1. metadata["target_column"] if it exists and is numeric
      2. a column explicitly flagged as target/measure in metadata["columns"]
      3. fall back to the first numeric column
    """
    target = metadata.get("target_column")
    if target and target in num_cols:
        return target

    for c in metadata.get("columns", []):
        if c.get("is_target") or c.get("role") in ("target", "measure"):
            name = c.get("name")
            if name in num_cols:
                return name

    return num_cols[0] if num_cols else None


# ─────────────────────────────────────────────────────────────
#  MAIN PUBLIC FUNCTION
# ─────────────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame, metadata: dict) -> dict:
    """
    Run full exploratory data analysis on the (cleaned) DataFrame.

    Returns:
    {
      "computed_at": str,
      "overview":        { shape, columns:[{name,dtype,dtype_category,missing,missing_pct}],
                            duplicate_rows, total_missing, total_missing_pct },
      "numeric_stats":   { col: { count, mean, median, std, ... } },
      "outliers":        { col: { method, count, pct, lower_bound, upper_bound,
                                   sample_values:[...], sample_indices:[...] } },
      "correlation":     { "pearson":  [[...]], "spearman": [[...]], "columns": [...],
                            "top_pairs":[...] },
      "distributions":   { col: { bin_centers, counts, bin_edges } },
      "categorical_eda": { col: { value_counts: [{label,count,pct}], n_unique } },
      "time_series":     { col: { value_col, monthly, weekly } }  — if date exists
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

    # Fallback: if metadata didn't classify anything as categorical, infer it
    if not cat_cols:
        cat_cols = [c for c in df.columns
                    if c not in num_cols and c not in date_cols]

    result = {"computed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    # ── 0. Data overview (shape, columns, dtypes, missing, duplicates) ──
    n_rows, n_cols = df.shape
    columns_overview = []
    for c in df.columns:
        miss = int(df[c].isnull().sum())
        columns_overview.append({
            "name":           c,
            "dtype":          str(df[c].dtype),
            "dtype_category": _dtype_category(df[c], cols_meta.get(c, {})),
            "missing":        miss,
            "missing_pct":    round(miss / n_rows * 100, 2) if n_rows else 0.0,
        })
    duplicate_rows   = int(df.duplicated().sum())
    total_missing    = int(df.isnull().sum().sum())
    result["overview"] = {
        "shape":              {"rows": n_rows, "columns": n_cols},
        "columns":            columns_overview,
        "duplicate_rows":     duplicate_rows,
        "duplicate_rows_pct": round(duplicate_rows / n_rows * 100, 2) if n_rows else 0.0,
        "total_missing":      total_missing,
        "total_missing_pct":  round(total_missing / df.size * 100, 2) if df.size else 0.0,
    }

    # ── 1. Descriptive stats (numeric) ────────────────────────
    numeric_stats = {}
    for col in num_cols:
        stats = _descriptive_stats(df[col])
        if stats:
            stats["skew_label"] = _skew_label(stats.get("skewness"))
            stats["kurt_label"] = _kurt_label(stats.get("kurtosis"))
            numeric_stats[col]  = stats
    result["numeric_stats"] = numeric_stats

    # ── 2. Outlier detection (IQR method) ─────────────────────
    outliers = {}
    for col in num_cols:
        clean = df[col].dropna()
        if len(clean) < 4:
            continue
        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        iqr    = q3 - q1
        if iqr == 0:
            lower, upper = q1, q3
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
        mask     = (clean < lower) | (clean > upper)
        out_vals = clean[mask]
        outliers[col] = {
            "method":         "IQR (1.5x)",
            "count":          int(out_vals.count()),
            "pct":            round(out_vals.count() / len(clean) * 100, 2),
            "lower_bound":    _safe_float(lower),
            "upper_bound":    _safe_float(upper),
            "sample_values":  [_safe_float(v) for v in out_vals.head(10).tolist()],
            "sample_indices": [int(i) for i in out_vals.head(10).index.tolist()],
        }
    result["outliers"] = outliers

    # ── 3. Correlation matrices (pairwise deletion, not listwise) ─
    corr_result = {"columns": [], "pearson": [], "spearman": []}
    if len(num_cols) >= 2:
        # Using df[num_cols].corr() directly lets pandas perform
        # pairwise deletion per pair of columns, instead of dropping
        # every row that has *any* missing value across *any* column.
        pearson  = df[num_cols].corr(method="pearson")
        spearman = df[num_cols].corr(method="spearman")
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

    # ── 4. Distribution data (for histograms / boxplots) ──────
    # Kept here (in addition to frontend Plotly histograms) so that
    # non-visual consumers (e.g. the Phase 8 LLM) have bin data too.
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

    # ── 5. Categorical EDA ────────────────────────────────────
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

    # ── 6. Time-series EDA (if date column exists) ────────────
    time_series = {}
    val_col = _pick_value_column(df, num_cols, metadata)
    for col in date_cols:
        try:
            ts_series = pd.to_datetime(df[col], errors="coerce")
            if ts_series.isna().mean() > 0.5:
                continue
            temp_df       = df.copy()
            temp_df["_ts"] = ts_series
            temp_df        = temp_df.dropna(subset=["_ts"])

            if val_col is None:
                continue

            temp_df["_month"] = temp_df["_ts"].dt.to_period("M").astype(str)
            temp_df["_week"]  = temp_df["_ts"].dt.to_period("W").astype(str)

            monthly = temp_df.groupby("_month")[val_col].sum().reset_index()
            weekly  = temp_df.groupby("_week")[val_col].sum().reset_index()

            monthly["_ma"] = monthly[val_col].rolling(window=3, min_periods=1).mean()

            time_series[col] = {
                "value_col": val_col,
                "monthly": {
                    "periods": monthly["_month"].tolist(),
                    "values":  [_safe_float(v) for v in monthly[val_col]],
                    "moving_avg": [_safe_float(v) for v in monthly["_ma"]],
                },
                "weekly": {
                    "periods": weekly["_week"].tolist()[:52],
                    "values":  [_safe_float(v) for v in weekly[val_col]][:52],
                },
            }
        except Exception:
            continue
    result["time_series"] = time_series

    # ── 7. Plain-English summary insights ─────────────────────
    insights = []

    # Duplicates
    if duplicate_rows == 0:
        insights.append("✅ No duplicate rows found.")
    else:
        insights.append(
            f"🧬 {duplicate_rows:,} duplicate rows found "
            f"({result['overview']['duplicate_rows_pct']}% of data)."
        )

    # Missing summary
    if total_missing == 0:
        insights.append("✅ No missing values — dataset is complete.")
    else:
        insights.append(f"⚠️ {total_missing:,} missing cells ({result['overview']['total_missing_pct']}% of data).")

    # Outlier insights
    for col, o in outliers.items():
        if o["count"] > 0:
            insights.append(
                f"🚨 '{col}' has {o['count']} outlier(s) ({o['pct']}%) "
                f"outside [{o['lower_bound']}, {o['upper_bound']}] via IQR method."
            )

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