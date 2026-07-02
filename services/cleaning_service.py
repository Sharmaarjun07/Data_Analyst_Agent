# =============================================================
#  services/cleaning_service.py
#  Phase 4 — Data Cleaning Engine
#
#  Two public functions:
#    analyze_issues(df, metadata)  → what problems exist + recommended fix
#    apply_cleaning(df, plan)      → apply the chosen fixes, return cleaned df + report
#
#  The LLM never touches this layer.  Pure Pandas logic only.
# =============================================================

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional


# ─────────────────────────────────────────────────────────────
#  ISSUE DETECTION
# ─────────────────────────────────────────────────────────────

def analyze_issues(df: pd.DataFrame, metadata: dict) -> dict:
    """
    Scan the DataFrame and return a structured list of every cleaning
    issue found, together with the recommended fix and available options.

    Return shape:
    {
      "summary": { total_issues, missing_issues, duplicate_issues,
                   outlier_issues, type_issues },
      "missing": [ { col, dtype_category, missing_count, missing_pct,
                     recommended_fix, fix_options, fill_value } ],
      "duplicates": { count, pct, recommended_fix },
      "outliers":  [ { col, count, lower_fence, upper_fence,
                        recommended_fix, fix_options } ],
      "type_issues": [ { col, current_dtype, suggested_dtype, reason } ],
    }
    """
    cols_meta = {c["name"]: c for c in metadata.get("columns", [])}
    n_rows    = df.shape[0]

    # ── Missing values ────────────────────────────────────────
    missing_issues = []
    for col in df.columns:
        miss_n = int(df[col].isnull().sum())
        if miss_n == 0:
            continue
        miss_p  = round(miss_n / n_rows * 100, 2)
        dtype_c = cols_meta.get(col, {}).get("dtype_category", "categorical")

        if miss_p > 50:
            rec    = "drop_column"
            opts   = ["drop_column", "fill_median", "fill_mean", "fill_mode", "fill_constant"]
            fill_v = None
        elif dtype_c == "numeric":
            skew = abs(cols_meta.get(col, {}).get("skewness", 0))
            rec  = "fill_median" if skew > 0.5 else "fill_mean"
            opts = ["fill_median", "fill_mean", "fill_constant", "drop_rows"]
            ns   = cols_meta.get(col, {}).get("numeric_stats", {})
            fill_v = ns.get("median", ns.get("mean", 0))
        elif dtype_c == "categorical":
            rec    = "fill_mode"
            opts   = ["fill_mode", "fill_constant", "drop_rows"]
            fill_v = str(df[col].mode().iloc[0]) if not df[col].mode().empty else "Unknown"
        else:
            rec    = "fill_constant"
            opts   = ["fill_constant", "drop_rows"]
            fill_v = "Unknown"

        missing_issues.append({
            "col":             col,
            "dtype_category":  dtype_c,
            "missing_count":   miss_n,
            "missing_pct":     miss_p,
            "recommended_fix": rec,
            "fix_options":     opts,
            "fill_value":      fill_v,
        })

    # ── Duplicates ────────────────────────────────────────────
    dup_count = int(df.duplicated().sum())
    duplicates = {
        "count":           dup_count,
        "pct":             round(dup_count / n_rows * 100, 2),
        "recommended_fix": "drop_duplicates" if dup_count > 0 else "none",
    }

    # ── Outliers (numeric only, IQR method) ──────────────────
    outlier_issues = []
    for col in df.select_dtypes(include="number").columns:
        series = df[col].dropna()
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr    = q3 - q1
        if iqr == 0:
            continue
        lower  = q1 - 1.5 * iqr
        upper  = q3 + 1.5 * iqr
        mask   = (series < lower) | (series > upper)
        count  = int(mask.sum())
        if count == 0:
            continue
        outlier_issues.append({
            "col":             col,
            "count":           count,
            "pct":             round(count / len(series) * 100, 2),
            "lower_fence":     round(float(lower), 4),
            "upper_fence":     round(float(upper), 4),
            "q1":              round(float(q1), 4),
            "q3":              round(float(q3), 4),
            "recommended_fix": "winsorize",
            "fix_options":     ["winsorize", "drop_rows", "keep"],
        })

    # ── Data type issues ──────────────────────────────────────
    type_issues = []
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        if dtype_str != "object":
            continue
        # Try numeric
        numeric_attempt = pd.to_numeric(df[col], errors="coerce")
        n_valid = numeric_attempt.notna().sum()
        n_total = df[col].notna().sum()
        if n_total > 0 and n_valid / n_total > 0.90:
            type_issues.append({
                "col":            col,
                "current_dtype":  dtype_str,
                "suggested_dtype": "float64",
                "reason":         f"{n_valid}/{n_total} values parse as numbers",
                "action":         "convert_numeric",
            })
            continue
        # Try datetime
        try:
            dt_attempt = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
            dt_valid   = dt_attempt.notna().sum()
            if n_total > 0 and dt_valid / n_total > 0.90:
                type_issues.append({
                    "col":            col,
                    "current_dtype":  dtype_str,
                    "suggested_dtype": "datetime64",
                    "reason":         f"{dt_valid}/{n_total} values parse as dates",
                    "action":         "convert_datetime",
                })
        except Exception:
            pass

    total = (
        len(missing_issues)
        + (1 if dup_count > 0 else 0)
        + len(outlier_issues)
        + len(type_issues)
    )

    return {
        "summary": {
            "total_issues":     total,
            "missing_issues":   len(missing_issues),
            "duplicate_issues": 1 if dup_count > 0 else 0,
            "outlier_issues":   len(outlier_issues),
            "type_issues":      len(type_issues),
        },
        "missing":     missing_issues,
        "duplicates":  duplicates,
        "outliers":    outlier_issues,
        "type_issues": type_issues,
    }


# ─────────────────────────────────────────────────────────────
#  APPLY CLEANING
# ─────────────────────────────────────────────────────────────

def apply_cleaning(df: pd.DataFrame, plan: dict) -> tuple[pd.DataFrame, dict]:
    """
    Apply a cleaning plan produced by the UI and return:
      (cleaned_df, cleaning_report)

    plan shape:
    {
      "missing":    { col_name: { "action": "fill_median"|"fill_mean"|
                                            "fill_mode"|"fill_constant"|
                                            "drop_column"|"drop_rows",
                                  "constant": optional_value } },
      "duplicates": { "action": "drop_duplicates"|"keep" },
      "outliers":   { col_name: { "action": "winsorize"|"drop_rows"|"keep" } },
      "types":      { col_name: { "action": "convert_numeric"|"convert_datetime"|"keep" } },
    }
    """
    cleaned     = df.copy()
    report_rows = []          # list of plain-English log entries
    rows_before = len(cleaned)

    # ── 1. Type conversions first (affects later fills) ──────
    for col, cfg in plan.get("types", {}).items():
        action = cfg.get("action", "keep")
        if action == "keep" or col not in cleaned.columns:
            continue
        try:
            if action == "convert_numeric":
                cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
                report_rows.append({
                    "step":   "Type Fix",
                    "column": col,
                    "action": "Converted to numeric (float64)",
                    "impact": f"{cleaned[col].notna().sum()} valid values",
                    "status": "✅",
                })
            elif action == "convert_datetime":
                cleaned[col] = pd.to_datetime(cleaned[col],
                                              infer_datetime_format=True, errors="coerce")
                report_rows.append({
                    "step":   "Type Fix",
                    "column": col,
                    "action": "Converted to datetime",
                    "impact": f"{cleaned[col].notna().sum()} valid dates",
                    "status": "✅",
                })
        except Exception as e:
            report_rows.append({
                "step": "Type Fix", "column": col,
                "action": f"Conversion failed: {e}", "impact": "—", "status": "⚠️",
            })

    # ── 2. Missing values ─────────────────────────────────────
    cols_to_drop = []
    rows_to_drop_mask = pd.Series(False, index=cleaned.index)

    for col, cfg in plan.get("missing", {}).items():
        action = cfg.get("action", "keep")
        if action == "keep" or col not in cleaned.columns:
            continue
        miss_before = int(cleaned[col].isnull().sum())
        if miss_before == 0:
            continue

        if action == "drop_column":
            cols_to_drop.append(col)
            report_rows.append({
                "step": "Missing Values", "column": col,
                "action": "Column dropped",
                "impact": f"{miss_before} missing cells removed with column",
                "status": "🗑️",
            })
        elif action == "drop_rows":
            rows_to_drop_mask |= cleaned[col].isnull()
            report_rows.append({
                "step": "Missing Values", "column": col,
                "action": "Rows with missing value flagged for removal",
                "impact": f"{miss_before} rows flagged",
                "status": "🗑️",
            })
        elif action == "fill_median":
            val = cleaned[col].median()
            cleaned[col].fillna(val, inplace=True)
            report_rows.append({
                "step": "Missing Values", "column": col,
                "action": f"Filled with median ({round(val, 4)})",
                "impact": f"{miss_before} cells filled",
                "status": "✅",
            })
        elif action == "fill_mean":
            val = cleaned[col].mean()
            cleaned[col].fillna(val, inplace=True)
            report_rows.append({
                "step": "Missing Values", "column": col,
                "action": f"Filled with mean ({round(val, 4)})",
                "impact": f"{miss_before} cells filled",
                "status": "✅",
            })
        elif action == "fill_mode":
            val = cleaned[col].mode().iloc[0] if not cleaned[col].mode().empty else "Unknown"
            cleaned[col].fillna(val, inplace=True)
            report_rows.append({
                "step": "Missing Values", "column": col,
                "action": f'Filled with mode ("{val}")',
                "impact": f"{miss_before} cells filled",
                "status": "✅",
            })
        elif action == "fill_constant":
            val = cfg.get("constant", "Unknown")
            cleaned[col].fillna(val, inplace=True)
            report_rows.append({
                "step": "Missing Values", "column": col,
                "action": f'Filled with constant ("{val}")',
                "impact": f"{miss_before} cells filled",
                "status": "✅",
            })

    # Apply column drops
    if cols_to_drop:
        cleaned.drop(columns=cols_to_drop, inplace=True)

    # Apply row drops from missing
    if rows_to_drop_mask.any():
        n_drop = int(rows_to_drop_mask.sum())
        cleaned = cleaned[~rows_to_drop_mask].reset_index(drop=True)
        report_rows.append({
            "step": "Missing Values", "column": "(multiple)",
            "action": "Rows with missing values removed",
            "impact": f"{n_drop} rows removed",
            "status": "🗑️",
        })

    # ── 3. Duplicates ─────────────────────────────────────────
    dup_action = plan.get("duplicates", {}).get("action", "keep")
    if dup_action == "drop_duplicates":
        dup_count = int(cleaned.duplicated().sum())
        if dup_count > 0:
            cleaned = cleaned.drop_duplicates().reset_index(drop=True)
            report_rows.append({
                "step": "Duplicates", "column": "—",
                "action": "Duplicate rows removed",
                "impact": f"{dup_count} rows removed",
                "status": "✅",
            })
    else:
        report_rows.append({
            "step": "Duplicates", "column": "—",
            "action": "Kept (no action)",
            "impact": "—", "status": "⏭️",
        })

    # ── 4. Outliers ───────────────────────────────────────────
    for col, cfg in plan.get("outliers", {}).items():
        action = cfg.get("action", "keep")
        if action == "keep" or col not in cleaned.columns:
            continue
        series = cleaned[col].dropna()
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr    = q3 - q1
        lower  = q1 - 1.5 * iqr
        upper  = q3 + 1.5 * iqr
        n_out  = int(((series < lower) | (series > upper)).sum())

        if action == "winsorize":
            cleaned[col] = cleaned[col].clip(lower=lower, upper=upper)
            report_rows.append({
                "step": "Outliers", "column": col,
                "action": f"Winsorized to [{round(lower,2)}, {round(upper,2)}]",
                "impact": f"{n_out} value(s) clipped",
                "status": "✅",
            })
        elif action == "drop_rows":
            mask = (cleaned[col] < lower) | (cleaned[col] > upper)
            cleaned = cleaned[~mask].reset_index(drop=True)
            report_rows.append({
                "step": "Outliers", "column": col,
                "action": "Rows with outlier values removed",
                "impact": f"{n_out} rows removed",
                "status": "🗑️",
            })

    rows_after = len(cleaned)

    return cleaned, {
        "applied_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rows_before":  rows_before,
        "rows_after":   rows_after,
        "rows_removed": rows_before - rows_after,
        "cols_before":  df.shape[1],
        "cols_after":   cleaned.shape[1],
        "steps":        report_rows,
        "total_actions": len([r for r in report_rows if r["status"] not in ("⏭️",)]),
    }