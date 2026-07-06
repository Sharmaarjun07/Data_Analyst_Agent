"""
frontend/pages/eda.py
EDA page — Phase 5 (built).

Uses st.session_state.cleaned_df, calls run_eda(), and renders the
results with Streamlit + Plotly charts and tables.
"""

import hashlib
import json

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from services.eda_service import run_eda


# ─────────────────────────────────────────────────────────────
#  Module-level cached EDA runner (Streamlit best practice:
#  cached functions should live at module scope, not be
#  redefined inside render() on every rerun).
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Running exploratory data analysis...")
def cached_eda(df_hash_key: str, _df: pd.DataFrame, metadata: dict) -> dict:
    # Leading underscore on _df tells Streamlit not to hash the dataframe
    # itself (it can't hash arbitrary DataFrames reliably) — we already
    # supply a stable df_hash_key above for cache-invalidation purposes.
    return run_eda(_df, metadata)


def _hash_dataframe(df: pd.DataFrame) -> str:
    """
    Stable hash of a dataframe's content for cache-keying.
    Hashing the underlying bytes via md5 avoids the (very small but
    real) collision risk of summing per-row hashes together.
    """
    row_hashes = pd.util.hash_pandas_object(df, index=True).values
    return hashlib.md5(row_hashes.tobytes()).hexdigest()


def _has_statsmodels() -> bool:
    try:
        import statsmodels  # noqa: F401
        return True
    except ImportError:
        return False


def render():
    st.title("📊 Exploratory Data Analysis")

    # ── 1. Guard: make sure cleaned data exists ───────────────
    if "cleaned_df" not in st.session_state or st.session_state.cleaned_df is None:
        st.warning(
            "⚠️ No cleaned dataset found. Please upload and clean your data "
            "on the **Cleaning** page first."
        )
        return

    df = st.session_state.cleaned_df

    if df is None or df.empty:
        st.warning("⚠️ The cleaned dataset is empty. Please check your Cleaning step.")
        return

    metadata = st.session_state.get("metadata", {"columns": []})

    # ── 2. Run EDA (cached so it isn't recomputed on every widget click) ─
    df_hash_key = _hash_dataframe(df)
    eda_result = cached_eda(df_hash_key, _df=df, metadata=metadata)

    overview        = eda_result.get("overview", {})
    numeric_stats   = eda_result.get("numeric_stats", {})
    outliers        = eda_result.get("outliers", {})
    correlation     = eda_result.get("correlation", {})
    distributions   = eda_result.get("distributions", {})
    categorical_eda = eda_result.get("categorical_eda", {})
    time_series     = eda_result.get("time_series", {})
    insights        = eda_result.get("summary_insights", [])

    top_bar_l, top_bar_r = st.columns([4, 1])
    with top_bar_l:
        st.caption(f"Computed at: {eda_result.get('computed_at', '—')}")
    with top_bar_r:
        st.download_button(
            "⬇️ Download EDA Report (JSON)",
            data=json.dumps(eda_result, indent=2, default=str),
            file_name="eda_report.json",
            mime="application/json",
            use_container_width=True,
        )

    tabs = st.tabs([
        "🧾 Overview",
        "📈 Summary Statistics",
        "🚨 Outliers",
        "🔗 Correlations",
        "📊 Distributions",
        "🏷️ Categorical",
        "🕒 Time Series",
        "💡 Insights",
    ])

    # ─────────────────────────────────────────────────────────
    # TAB 1 — Overview (shape, columns, dtypes, missing, duplicates)
    # ─────────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Dataset shape")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", overview.get("shape", {}).get("rows", 0))
        c2.metric("Columns", overview.get("shape", {}).get("columns", 0))
        c3.metric("Duplicate rows", overview.get("duplicate_rows", 0))
        c4.metric("Missing cells", overview.get("total_missing", 0))

        st.subheader("Column details")
        cols_df = pd.DataFrame(overview.get("columns", []))
        if not cols_df.empty:
            st.dataframe(cols_df, use_container_width=True)

            # Missing-value visualization: bar chart of missing % per column
            missing_df = cols_df[cols_df["missing"] > 0].sort_values(
                "missing_pct", ascending=False
            )
            if not missing_df.empty:
                st.subheader("Missing values by column")
                fig = px.bar(
                    missing_df, x="name", y="missing_pct",
                    text="missing",
                    labels={"name": "Column", "missing_pct": "Missing (%)"},
                    title="Percentage of missing values per column",
                )
                fig.update_traces(texttemplate="%{text} cells", textposition="outside")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing values in any column.")
        else:
            st.info("No column metadata available.")

        if overview.get("duplicate_rows", 0) > 0:
            st.warning(
                f"Found {overview['duplicate_rows']} duplicate rows "
                f"({overview.get('duplicate_rows_pct', 0)}% of data)."
            )
        else:
            st.success("No duplicate rows detected.")

    # ─────────────────────────────────────────────────────────
    # TAB 2 — Summary statistics
    # ─────────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Descriptive statistics (numeric columns)")
        if numeric_stats:
            stats_df = pd.DataFrame(numeric_stats).T
            numeric_only_cols = stats_df.select_dtypes(include="number").columns
            stats_df[numeric_only_cols] = stats_df[numeric_only_cols].round(3)
            st.dataframe(stats_df, use_container_width=True)

            selected_col = st.selectbox(
                "Select a numeric column to inspect closely",
                options=list(numeric_stats.keys()),
                key="stats_col_select",
            )
            if selected_col:
                s = numeric_stats[selected_col]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Mean", s.get("mean"))
                c2.metric("Median", s.get("median"))
                c3.metric("Std Dev", s.get("std"))
                c4.metric("Range", s.get("range"))
                c1.metric("Min", s.get("min"))
                c2.metric("Max", s.get("max"))
                c3.metric("Skewness", f"{s.get('skewness')} ({s.get('skew_label')})")
                c4.metric("Kurtosis", f"{s.get('kurtosis')} ({s.get('kurt_label')})")
        else:
            st.info("No numeric columns found in this dataset.")

    # ─────────────────────────────────────────────────────────
    # TAB 3 — Outliers
    # ─────────────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Outlier detection (IQR method)")
        if outliers:
            out_rows = []
            for col, o in outliers.items():
                out_rows.append({
                    "column": col,
                    "outlier_count": o["count"],
                    "outlier_pct": o["pct"],
                    "lower_bound": o["lower_bound"],
                    "upper_bound": o["upper_bound"],
                })
            st.dataframe(pd.DataFrame(out_rows), use_container_width=True)

            out_col = st.selectbox(
                "View boxplot / flagged rows for column",
                options=list(outliers.keys()),
                key="outlier_col_select",
            )
            if out_col and out_col in df.columns:
                fig = px.box(df, y=out_col, points="outliers",
                             title=f"Boxplot — {out_col}")
                st.plotly_chart(fig, use_container_width=True)

                idxs = outliers[out_col].get("sample_indices", [])
                if idxs:
                    st.caption("Sample rows flagged as outliers:")
                    valid_idxs = [i for i in idxs if i in df.index]
                    st.dataframe(df.loc[valid_idxs], use_container_width=True)
        else:
            st.info("No numeric columns available for outlier detection.")

    # ─────────────────────────────────────────────────────────
    # TAB 4 — Correlations (heatmap + scatter + plain-English insights)
    # ─────────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Correlation heatmap")
        cols = correlation.get("columns", [])
        pearson = correlation.get("pearson", [])
        if cols and pearson:
            fig = go.Figure(data=go.Heatmap(
                z=pearson, x=cols, y=cols,
                colorscale="RdBu", zmin=-1, zmax=1,
                colorbar=dict(title="Pearson r"),
                text=pearson,
                texttemplate="%{z:.2f}",
                hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>",
            ))
            fig.update_layout(title="Pearson correlation matrix")
            st.plotly_chart(fig, use_container_width=True)

            top_pairs = correlation.get("top_pairs", [])
            if top_pairs:
                st.subheader("Correlation insights")
                for pair in top_pairs[:8]:
                    emoji = "🔗" if pair["strength"] == "strong" else (
                        "🔸" if pair["strength"] == "moderate" else "▫️"
                    )
                    st.markdown(
                        f"{emoji} **{pair['col_a']}** and **{pair['col_b']}** are "
                        f"**{pair['strength']}ly {pair['direction']}ly** correlated "
                        f"(r = {pair['pearson']:.2f})"
                    )
                with st.expander("View full correlation pairs table"):
                    st.dataframe(pd.DataFrame(top_pairs), use_container_width=True)

            st.subheader("Scatter plot explorer")
            c1, c2 = st.columns(2)
            x_col = c1.selectbox("X axis", options=cols, index=0, key="scatter_x")
            y_col = c2.selectbox(
                "Y axis", options=cols,
                index=min(1, len(cols) - 1), key="scatter_y"
            )
            if x_col and y_col:
                scatter_kwargs = dict(x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                if _has_statsmodels():
                    scatter_kwargs["trendline"] = "ols"
                else:
                    st.caption(
                        "ℹ️ Install `statsmodels` to see an OLS trendline on this scatter plot "
                        "(`pip install statsmodels`)."
                    )
                fig2 = px.scatter(df, **scatter_kwargs)
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns to compute correlations.")

    # ─────────────────────────────────────────────────────────
    # TAB 5 — Distributions (histograms, using backend-computed bins)
    # ─────────────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("Distributions")
        if distributions:
            dist_col = st.selectbox(
                "Select column", options=list(distributions.keys()), key="dist_col_select"
            )
            if dist_col:
                d = distributions[dist_col]
                fig = go.Figure(data=go.Bar(
                    x=d["bin_centers"], y=d["counts"],
                    marker_color="indianred",
                ))
                fig.update_layout(
                    title=f"Histogram — {dist_col} (precomputed bins)",
                    xaxis_title=dist_col, yaxis_title="count",
                    bargap=0.02,
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns available for distribution plots.")

    # ─────────────────────────────────────────────────────────
    # TAB 6 — Categorical EDA (bar charts)
    # ─────────────────────────────────────────────────────────
    with tabs[5]:
        st.subheader("Categorical breakdown")
        if categorical_eda:
            cat_col = st.selectbox(
                "Select categorical column",
                options=list(categorical_eda.keys()),
                key="cat_col_select",
            )
            if cat_col:
                vc_df = pd.DataFrame(categorical_eda[cat_col]["value_counts"])
                st.metric("Unique categories", categorical_eda[cat_col]["n_unique"])
                if not vc_df.empty:
                    fig = px.bar(
                        vc_df, x="label", y="count",
                        color="count", color_continuous_scale="Blues",
                        title=f"Value counts — {cat_col}",
                        text="pct",
                    )
                    fig.update_traces(texttemplate="%{text}%", textposition="outside")
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(vc_df, use_container_width=True)
        else:
            st.info("No categorical columns found in this dataset.")

    # ─────────────────────────────────────────────────────────
    # TAB 7 — Time series (trends)
    # ─────────────────────────────────────────────────────────
    with tabs[6]:
        st.subheader("Trends over time")
        if time_series:
            ts_col = st.selectbox(
                "Select date column", options=list(time_series.keys()), key="ts_col_select"
            )
            if ts_col:
                ts = time_series[ts_col]
                monthly = ts.get("monthly", {})
                if monthly.get("periods"):
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=monthly["periods"], y=monthly["values"],
                        mode="lines+markers", name=ts["value_col"],
                    ))
                    fig.add_trace(go.Scatter(
                        x=monthly["periods"], y=monthly["moving_avg"],
                        mode="lines", name="3-period moving avg",
                        line=dict(dash="dash"),
                    ))
                    fig.update_layout(
                        title=f"Monthly trend — {ts['value_col']} by {ts_col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No datetime columns detected — trend analysis unavailable.")

    # ─────────────────────────────────────────────────────────
    # TAB 8 — Plain-English insights
    # ─────────────────────────────────────────────────────────
    with tabs[7]:
        st.subheader("Key insights")
        if insights:
            for line in insights:
                st.markdown(f"- {line}")
        else:
            st.info("No notable insights were generated for this dataset.")