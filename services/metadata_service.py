# =============================================================
#  services/metadata_service.py
#  Phase 3 — Metadata Engine
#  Extracts deep column-level metadata from a Pandas DataFrame.
#  The LLM and all later phases read from this — never raw data.
# =============================================================

import pandas as pd
import numpy as np
from datetime import datetime


# ─────────────────────────────────────────────────────────────
#  INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────

def _classify_dtype(series: pd.Series) -> str:
    """
    Map a Pandas dtype to one of four human-readable categories:
    numeric | categorical | datetime | text
    """
    dtype_str = str(series.dtype)
    if dtype_str.startswith(("int", "float", "uint")):
        return "numeric"
    if dtype_str == "bool":
        return "categorical"
    if dtype_str.startswith("datetime") or dtype_str == "date":
        return "datetime"
    # Object columns — try to detect dates hidden as strings
    if dtype_str == "object":
        sample = series.dropna().head(50)
        try:
            pd.to_datetime(sample, infer_datetime_format=True)
            return "datetime"
        except Exception:
            pass
        # Long strings → text; short strings → categorical
        avg_len = sample.astype(str).str.len().mean()
        n_unique = series.nunique()
        if avg_len > 40 or n_unique / max(len(series), 1) > 0.85:
            return "text"
        return "categorical"
    return "categorical"


def _safe_sample(series: pd.Series, n: int = 4) -> list:
    """Return up to n non-null sample values as plain Python objects."""
    values = series.dropna().unique()[:n]
    result = []
    for v in values:
        if isinstance(v, (np.integer,)):
            result.append(int(v))
        elif isinstance(v, (np.floating,)):
            result.append(round(float(v), 4))
        elif isinstance(v, (np.bool_,)):
            result.append(bool(v))
        else:
            result.append(str(v))
    return result


def _numeric_stats(series: pd.Series) -> dict:
    """Descriptive stats for a numeric column."""
    clean = series.dropna()
    if clean.empty:
        return {}
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = int(((clean < lower) | (clean > upper)).sum())
    return {
        "mean":     round(float(clean.mean()), 4),
        "median":   round(float(clean.median()), 4),
        "std":      round(float(clean.std()), 4),
        "min":      round(float(clean.min()), 4),
        "max":      round(float(clean.max()), 4),
        "q1":       round(float(q1), 4),
        "q3":       round(float(q3), 4),
        "skewness": round(float(clean.skew()), 4),
        "outliers": outliers,
    }


def _categorical_stats(series: pd.Series) -> dict:
    """Frequency breakdown for a categorical column."""
    counts = series.value_counts()
    top = counts.head(5)
    return {
        "top_values": [
            {"value": str(k), "count": int(v), "pct": round(v / len(series) * 100, 1)}
            for k, v in top.items()
        ],
        "n_unique": int(series.nunique()),
    }


def _column_recommendation(col_meta: dict) -> str:
    """
    Return a single plain-English recommendation for a column
    based on its metadata. This text is shown in the Dataset page
    and later passed to the LLM insight engine.
    """
    missing_pct = col_meta["missing_pct"]
    dtype       = col_meta["dtype_category"]
    n_unique    = col_meta.get("n_unique", 0)
    n_rows      = col_meta["total_rows"]

    if missing_pct > 50:
        return f"⚠️ High missingness ({missing_pct:.1f}%) — consider dropping this column."
    if missing_pct > 10 and dtype == "numeric":
        return f"Fill {missing_pct:.1f}% missing values with the median (skew-resistant)."
    if missing_pct > 10 and dtype == "categorical":
        return f"Fill {missing_pct:.1f}% missing values with the most frequent category."
    if missing_pct > 0 and dtype == "numeric":
        return f"Fill {missing_pct:.1f}% missing values — mean or median both reasonable here."

    if dtype == "numeric":
        outliers = col_meta.get("outliers", 0)
        skewness = abs(col_meta.get("skewness", 0))
        if outliers > 0:
            return f"🔍 {outliers} outlier(s) detected via IQR — review before modeling."
        if skewness > 1:
            return "Highly skewed — consider log-transform before using in a regression model."
        return "✅ Looks clean — ready for modeling."

    if dtype == "categorical":
        if n_unique == n_rows:
            return "Every value is unique — this is likely an ID column, exclude from ML features."
        if n_unique <= 2:
            return "Binary column — good candidate for a classification target or boolean feature."
        if n_unique > 50:
            return f"High cardinality ({n_unique} categories) — consider grouping rare values or encoding carefully."
        return "✅ Reasonable cardinality — suitable for one-hot or label encoding."

    if dtype == "datetime":
        return "Parse as datetime and extract day, month, year, weekday for feature engineering."

    if dtype == "text":
        return "Free-text column — consider NLP features or TF-IDF encoding for ML."

    return "✅ No issues detected."


# ─────────────────────────────────────────────────────────────
#  MAIN PUBLIC FUNCTION
# ─────────────────────────────────────────────────────────────

def extract_metadata(df: pd.DataFrame, filename: str = "dataset.csv") -> dict:
    """
    Full metadata extraction for a DataFrame.

    Returns a nested dict that every later phase consumes:
      {
        "overview":  { rows, cols, memory_bytes, ... },
        "quality":   { score, grade, missing_cells, ... },
        "columns":   [ { name, dtype, dtype_category, missing_count,
                         missing_pct, n_unique, sample_values,
                         numeric_stats or categorical_stats,
                         recommendation, total_rows }, ... ]
      }
    """
    n_rows, n_cols = df.shape
    mem_bytes      = int(df.memory_usage(deep=True).sum())
    dup_count      = int(df.duplicated().sum())
    total_cells    = n_rows * n_cols
    missing_cells  = int(df.isnull().sum().sum())
    missing_pct    = round(missing_cells / total_cells * 100, 2) if total_cells else 0.0

    # ── Column metadata ──────────────────────────────────────
    columns_meta = []
    for col_name in df.columns:
        series  = df[col_name]
        dtype_c = _classify_dtype(series)
        miss_n  = int(series.isnull().sum())
        miss_p  = round(miss_n / n_rows * 100, 2) if n_rows else 0.0

        col_meta: dict = {
            "name":          col_name,
            "dtype":         str(series.dtype),
            "dtype_category": dtype_c,
            "missing_count": miss_n,
            "missing_pct":   miss_p,
            "n_unique":      int(series.nunique(dropna=True)),
            "sample_values": _safe_sample(series),
            "total_rows":    n_rows,
        }

        if dtype_c == "numeric":
            col_meta["numeric_stats"] = _numeric_stats(series)
            col_meta["outliers"]  = col_meta["numeric_stats"].get("outliers", 0)
            col_meta["skewness"]  = col_meta["numeric_stats"].get("skewness", 0)
        elif dtype_c == "categorical":
            col_meta["categorical_stats"] = _categorical_stats(series)

        col_meta["recommendation"] = _column_recommendation(col_meta)
        columns_meta.append(col_meta)

    # ── Quality score (0–100) ────────────────────────────────
    # Penalise: missing data, duplicates, high-cardinality text
    missing_penalty   = min(missing_pct * 1.5, 40)
    duplicate_penalty = min(dup_count / max(n_rows, 1) * 100 * 0.5, 20)
    quality_score     = max(0, round(100 - missing_penalty - duplicate_penalty, 1))
    if   quality_score >= 90: grade = "A"
    elif quality_score >= 75: grade = "B"
    elif quality_score >= 60: grade = "C"
    elif quality_score >= 40: grade = "D"
    else:                     grade = "F"

    # ── Breakdowns by dtype category ─────────────────────────
    dtype_counts = {}
    for cm in columns_meta:
        dtype_counts[cm["dtype_category"]] = dtype_counts.get(cm["dtype_category"], 0) + 1

    # ── Potential target columns ─────────────────────────────
    # A column is a candidate target if it's numeric with reasonable
    # variance OR categorical with low cardinality (2–20 unique values).
    target_candidates = []
    for cm in columns_meta:
        if cm["dtype_category"] == "numeric" and cm["n_unique"] > 10:
            target_candidates.append({"name": cm["name"], "reason": "Numeric — regression target"})
        if cm["dtype_category"] == "categorical" and 2 <= cm["n_unique"] <= 20:
            target_candidates.append({"name": cm["name"], "reason": f"Categorical ({cm['n_unique']} classes) — classification target"})

    return {
        "overview": {
            "filename":    filename,
            "n_rows":      n_rows,
            "n_cols":      n_cols,
            "memory_bytes": mem_bytes,
            "dup_count":   dup_count,
            "missing_cells": missing_cells,
            "missing_pct": missing_pct,
            "dtype_counts": dtype_counts,
            "extracted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "quality": {
            "score":    quality_score,
            "grade":    grade,
            "issues": {
                "missing_cells":   missing_cells,
                "duplicate_rows":  dup_count,
                "high_missing_cols": [
                    cm["name"] for cm in columns_meta if cm["missing_pct"] > 30
                ],
                "id_like_cols": [
                    cm["name"] for cm in columns_meta
                    if cm["n_unique"] == n_rows and cm["dtype_category"] == "categorical"
                ],
            },
        },
        "columns":           columns_meta,
        "target_candidates": target_candidates[:6],  # top 6
    }